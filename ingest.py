import logging
import os
import chromadb
import argparse
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect, LangDetectException
from langchain_community.vectorstores.utils import filter_complex_metadata

# Configure logging to output debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
COLLECTION_NAME = "knowledge_base"

# --- ChromaDB Client ---
# Connect to ChromaDB running in Docker
client = chromadb.HttpClient(host="chroma", port=8000)

# --- Embeddings Model ---
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- LangChain Chroma Integration ---
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


def get_known_documents_from_db():
    """Retrieves metadata for all documents currently in the vector store."""
    print("Querying database for existing document metadata...")
    try:
        existing_docs = vectorstore.get(include=["metadatas"])
        known_docs = {}
        if existing_docs and existing_docs.get("metadatas"):
            for metadata in existing_docs["metadatas"]:
                source = metadata.get("source")
                last_modified = metadata.get("last_modified")
                if source:
                    if source not in known_docs or last_modified > known_docs[source]:
                        known_docs[source] = last_modified
        print(f"Found {len(known_docs)} known documents in the database.")
        return known_docs
    except chromadb.errors.NotFoundError:
        print("Collection not found. Returning empty list of documents.")
        return {}


def get_filesystem_documents(docs_dir):
    """Scans the documents directory and returns a dict of {filepath: last_modified_time}."""
    print(f"Scanning for documents in '{docs_dir}'...")
    fs_docs = {}
    for root, _, files in os.walk(docs_dir):
        for filename in files:
            if filename.startswith("."):
                continue
            filepath = os.path.join(root, filename)
            try:
                fs_docs[filepath] = os.path.getmtime(filepath)
            except OSError:
                print(f"Could not read metadata for {filepath}, skipping.")
    print(f"Found {len(fs_docs)} documents on the filesystem.")
    return fs_docs


def process_file(file_item):
    """Worker function to load, split, and prepare a single file."""
    filepath, last_modified = file_item
    print(f"--- Starting processing for: {filepath} with strategy: fast ---")
    try:
        loader = UnstructuredLoader(filepath, strategy="auto", mode="elements")
        documents = loader.load()

        if not documents:
            print(f"--- No content extracted from {filepath}, skipping. ---")
            return []

        for doc in documents:
            doc.metadata = {
                "source": doc.metadata["source"],
                "last_modified": int(last_modified), # Store as integer
            }

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        print(
            f"--- Finished splitting {os.path.basename(filepath)} into {len(splits)} chunks. ---"
        )
        return splits
    except Exception as e:
        print(f"--- Error processing {filepath}: {e} ---")
        return []


def ingest_new_or_modified_documents(files_to_add):
    """Loads, splits, and ingests a list of new or modified documents serially."""
    if not files_to_add:
        print("No new or modified documents to ingest.")
        return

    print(
        f"Ingesting {len(files_to_add)} new or modified documents in a single process..."
    )

    all_splits = []
    for file_item in files_to_add.items():
        result = process_file(file_item)
        if result:
            all_splits.extend(result)

    if not all_splits:
        print("No content could be extracted from the documents.")
        return

    print(f"\nTotal new chunks to add: {len(all_splits)}")
    print("Adding chunks to the vector store...")
    batch_size = 5000
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i : i + batch_size]
        vectorstore.add_documents(documents=batch)
        print(
            f"  - Adding batch {i // batch_size + 1}/{(len(all_splits) + batch_size - 1) // batch_size} ({len(batch)} chunks)..."
        )


def delete_documents(files_to_delete):
    """Deletes all chunks associated with a list of file paths from the vector store."""
    if not files_to_delete:
        print("No documents to delete.")
        return

    print(f"Deleting {len(files_to_delete)} documents from the vector store...")

    for filepath in files_to_delete:
        print(f"  - Deleting chunks for: {filepath}")
        all_docs = vectorstore.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"])
            if metadata.get("source") == filepath
        ]

        if ids_to_delete:
            print(f"    - Found {len(ids_to_delete)} chunks to delete.")
            vectorstore.delete(ids=ids_to_delete)
        else:
            print(f"    - No chunks found for {filepath} (this might be unexpected).")


def synchronize_vectorstore(docs_dir):
    """Main function to synchronize the vector store with the documents directory."""
    print("\n--- Starting Vector Store Synchronization ---")

    known_docs = get_known_documents_from_db()
    fs_docs = get_filesystem_documents(docs_dir)

    known_paths = set(known_docs.keys())
    fs_paths = set(fs_docs.keys())

    paths_to_delete = known_paths - fs_paths
    if paths_to_delete:
        delete_documents(list(paths_to_delete))

    files_to_add = {}
    new_paths = fs_paths - known_paths
    for path in new_paths:
        files_to_add[path] = fs_docs[path]

    potentially_modified_paths = fs_paths.intersection(known_paths)
    for path in potentially_modified_paths:
        fs_mtime = int(fs_docs[path])
        db_mtime = int(known_docs.get(path, 0))
        if fs_mtime > db_mtime:
            print(f"Document '{path}' has been modified. Re-ingesting.")
            delete_documents([path])
            files_to_add[path] = fs_docs[path]

    ingest_new_or_modified_documents(files_to_add)

    print("\n--- Synchronization Complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Chroma vector store."
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="./documents",
        help="The directory where the documents are stored.",
    )
    parser.add_argument(
        "--re-ingest",
        action="store_true",
        help="Force a full re-ingestion by deleting the existing collection.",
    )
    args = parser.parse_args()

    if args.re_ingest:
        print("Performing a full reset of the collection...")
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception as e:
            print(f"Could not delete collection (it might not exist): {e}")

        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )

    synchronize_vectorstore(args.docs_dir)
