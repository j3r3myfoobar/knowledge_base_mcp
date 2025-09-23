import logging
import os
import chromadb
import argparse
import nltk
from concurrent.futures import ProcessPoolExecutor, as_completed


from typing import Dict, Any, List
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure NLTK data path is known
nltk.data.path.append("/app/nltk_data")

# Centralize settings for easier management
CONFIG = {
    "chroma_host": "chroma",
    "chroma_port": 8000,
    "collection_name": "knowledge_base",
    "embedding_model": "all-MiniLM-L6-v2",
    "docs_dir": "./documents",
    "chunk_size": 512,  # Optimal size for MiniLM can be smaller
    "chunk_overlap": 50,
    "ingestion_batch_size": 5000,
    "max_workers": os.cpu_count()
    or 4,  # Use available CPU cores for parallel processing
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Global Clients (Initialized once) ---
try:
    client = chromadb.HttpClient(host=CONFIG["chroma_host"], port=CONFIG["chroma_port"])
    embedding_function = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
    vectorstore = Chroma(
        client=client,
        collection_name=CONFIG["collection_name"],
        embedding_function=embedding_function,
    )
    logging.info("Successfully connected to ChromaDB.")
except Exception as e:
    logging.critical(f"Failed to connect to ChromaDB or initialize embeddings: {e}")
    exit(1)


def group_elements_into_chunks(
    elements: List[Any], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Instead of arbitrarily splitting text, this function groups semantic elements
    (like paragraphs, titles, list items) from `unstructured` into coherent chunks.
    It now correctly handles different object types returned by the loader.
    """
    chunks = []
    current_chunk_text = ""

    for el in elements:
        element_text = getattr(el, "text", getattr(el, "page_content", ""))
        if not element_text:
            continue

        # If adding the next element fits, append it
        if len(current_chunk_text) + len(element_text) <= chunk_size:
            current_chunk_text += "\n\n" + element_text
        else:
            # If the chunk is full, create a Document and start a new one
            if current_chunk_text:
                chunks.append(
                    Document(
                        page_content=current_chunk_text.strip(), metadata=el.metadata
                    )
                )

            # Start the new chunk with an overlap from the end of the last one
            overlap = current_chunk_text[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk_text = overlap + element_text

    # Add the last remaining chunk
    if current_chunk_text:
        # Use the metadata from the last processed element
        last_metadata = elements[-1].metadata if elements else {}
        chunks.append(
            Document(page_content=current_chunk_text.strip(), metadata=last_metadata)
        )

    return chunks


def process_file(file_item: tuple) -> List[Document]:
    """Worker function to load, semantically chunk, and prepare a single file."""
    filepath, last_modified = file_item
    logging.info(f"--- Starting processing for: {filepath} ---")
    try:
        # `unstructured` identifies elements like 'Title', 'NarrativeText', 'ListItem'
        loader = UnstructuredLoader(filepath, mode="elements", strategy="auto")
        elements = loader.load()

        if not elements:
            logging.warning(f"--- No content extracted from {filepath}, skipping. ---")
            return []

        # Group elements into meaningful chunks
        splits = group_elements_into_chunks(
            elements, CONFIG["chunk_size"], CONFIG["chunk_overlap"]
        )

        # Enrich metadata for all chunks from this file
        for split in splits:
            split.metadata["source"] = filepath
            split.metadata["last_modified"] = int(last_modified)

        # Filter out complex metadata types that ChromaDB might not handle
        final_splits = filter_complex_metadata(splits)

        logging.info(
            f"--- Finished splitting {os.path.basename(filepath)} into {len(final_splits)} semantic chunks. ---"
        )
        return final_splits
    except Exception as e:
        logging.error(f"--- Error processing {filepath}: {e} ---")
        return []


def ingest_documents(files_to_add: Dict[str, float]):
    """Loads, splits, and ingests documents in parallel."""
    if not files_to_add:
        logging.info("No new or modified documents to ingest.")
        return

    logging.info(
        f"Ingesting {len(files_to_add)} documents using up to {CONFIG['max_workers']} processes..."
    )

    all_splits = []
    # Use a process pool to handle files in parallel, significantly speeding up ingestion
    with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        future_to_file = {
            executor.submit(process_file, item): item for item in files_to_add.items()
        }
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                if result:
                    all_splits.extend(result)
            except Exception as e:
                logging.error(f"A worker process failed: {e}")

    if not all_splits:
        logging.warning("No content could be extracted from the documents.")
        return

    logging.info(f"Total new chunks to add: {len(all_splits)}")
    logging.info("Adding chunks to the vector store in batches...")

    batch_size = CONFIG["ingestion_batch_size"]
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i : i + batch_size]
        logging.info(f"Adding batch of {len(batch)} chunks...")
        vectorstore.add_documents(documents=batch)

    logging.info("Finished adding all chunks to the vector store.")


def delete_documents(files_to_delete: List[str]):
    """
    Deletes all chunks associated with a list of file paths from the vector store.
    This version is much more efficient as it minimizes database calls.
    """
    if not files_to_delete:
        logging.info("No documents to delete.")
        return

    logging.info(f"Deleting {len(files_to_delete)} documents from the vector store...")

    # Build a filter to find all chunks from the specified source files in one go
    # Note: ChromaDB's `where` filter syntax may vary. This is a common pattern.
    where_filter = {"source": {"$in": files_to_delete}}

    # Get all document IDs that match the filter
    existing_docs = vectorstore.get(where=where_filter, include=["metadatas"])
    ids_to_delete = existing_docs.get("ids", [])

    if ids_to_delete:
        logging.info(
            f"Found {len(ids_to_delete)} chunks across {len(files_to_delete)} files to delete."
        )
        vectorstore.delete(ids=ids_to_delete)
    else:
        logging.warning("Did not find any chunks to delete for the specified files.")


def synchronize_vectorstore(docs_dir: str):
    """Main function to synchronize the vector store with the documents directory."""
    logging.info("\n--- Starting Vector Store Synchronization ---")

    # 1. Get known documents from DB
    existing_docs = vectorstore.get(include=["metadatas"])
    known_docs = {}
    if existing_docs and existing_docs.get("metadatas"):
        for metadata in existing_docs["metadatas"]:
            source = metadata.get("source")
            if source:
                last_mod = metadata.get("last_modified", 0)
                if source not in known_docs or last_mod > known_docs[source]:
                    known_docs[source] = last_mod
    logging.info(f"Found metadata for {len(known_docs)} documents in the database.")

    # 2. Get documents on filesystem
    fs_docs = {}
    for root, _, files in os.walk(docs_dir):
        for filename in files:
            if not filename.startswith("."):
                filepath = os.path.join(root, filename)
                fs_docs[filepath] = os.path.getmtime(filepath)
    logging.info(f"Found {len(fs_docs)} documents on the filesystem.")

    # 3. Determine changes
    known_paths = set(known_docs.keys())
    fs_paths = set(fs_docs.keys())

    paths_to_delete = list(known_paths - fs_paths)
    delete_documents(paths_to_delete)

    files_to_add = {}
    # Add brand new files
    for path in fs_paths - known_paths:
        files_to_add[path] = fs_docs[path]

    # Add modified files (and re-ingest)
    for path in fs_paths.intersection(known_paths):
        if int(fs_docs[path]) > int(known_docs.get(path, 0)):
            logging.info(
                f"Document '{path}' has been modified. Scheduling for re-ingestion."
            )
            # First delete the old versions, then add the new one
            delete_documents([path])
            files_to_add[path] = fs_docs[path]

    # 4. Ingest new and modified files
    ingest_documents(files_to_add)

    logging.info("\n--- Synchronization Complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Chroma vector store."
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=CONFIG["docs_dir"],
        help="Directory with documents.",
    )
    parser.add_argument(
        "--re-ingest",
        action="store_true",
        help="Force re-ingestion by deleting the existing collection.",
    )
    args = parser.parse_args()

    if args.re_ingest:
        logging.warning(
            f"--- Deleting existing collection '{CONFIG['collection_name']}' as requested. ---"
        )
        try:
            client.delete_collection(name=CONFIG["collection_name"])
        except Exception as e:
            logging.error(f"Could not delete collection (it might not exist): {e}")

    synchronize_vectorstore(args.docs_dir)
