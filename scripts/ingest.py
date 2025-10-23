#!/usr/bin/env python3
"""
Ingest documents into baseline_kb collection.
Works with current simplified system (src/retrieval.py).

Usage:
    python scripts/ingest.py --docs_dir ./documents
    python scripts/ingest.py --docs_dir ./documents --re-ingest

SOLID Principles Applied:
- Depends on Retriever interface, not concrete implementation (Dependency Inversion)
- Can work with any Retriever implementation
"""
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document

from src.core.interfaces import Retriever
from src.factories import create_baseline_retriever
from src.core.database import get_chroma_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Handles document ingestion into baseline_kb collection.

    SOLID: Depends on Retriever interface (Dependency Inversion Principle)
    """

    def __init__(
        self,
        docs_dir: str,
        retriever: Optional[Retriever] = None,
        collection_name: str = "baseline_kb"
    ):
        """
        Initialize the document ingester.

        Args:
            docs_dir: Directory containing documents to ingest
            retriever: Optional Retriever implementation (if None, will create BaselineRetriever)
            collection_name: ChromaDB collection name

        Note: If retriever is provided, it will be used directly.
              If None, initialize_retriever() must be called before ingestion.
        """
        self.docs_dir = Path(docs_dir)
        self.collection_name = collection_name
        self.retriever = retriever  # Can be any Retriever implementation

    def handle_reingestion(self) -> None:
        """Delete existing collection if re-ingestion is requested."""
        logger.info("⚠️  Re-ingestion requested - deleting collection...")
        try:
            get_chroma_client().delete_collection(self.collection_name)
            logger.info("✅ Collection deleted successfully")
        except Exception as e:
            logger.warning(f"Could not delete collection (may not exist): {e}")

    def initialize_retriever(
        self,
        chroma_host: str = "chroma",
        chroma_port: int = 8000
    ) -> None:
        """
        Initialize the BaselineRetriever if not already provided.

        Args:
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port

        Note: This method uses the factory function to create retriever.
              For full control, pass a Retriever instance to __init__ instead.
        """
        if self.retriever is not None:
            logger.info("Retriever already initialized (dependency injection)")
            return

        logger.info("Initializing BaselineRetriever using factory...")
        self.retriever = create_baseline_retriever(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=self.collection_name
        )

    def find_documents(self) -> Tuple[List[Path], List[Path]]:
        """
        Find markdown and PDF files in the documents directory.

        Returns:
            Tuple of (markdown_files, pdf_files)

        Raises:
            FileNotFoundError: If documents directory doesn't exist
        """
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.docs_dir}")

        # Find all markdown and PDF files
        md_files = list(self.docs_dir.rglob("*.md"))
        pdf_files = list(self.docs_dir.rglob("*.pdf"))

        # Filter out documentation files from project root
        md_files = [
            f for f in md_files
            if f.name not in ['README.md', 'CLAUDE.md'] or 'documents' in str(f)
        ]

        logger.info(f"Found {len(md_files)} markdown files")
        logger.info(f"Found {len(pdf_files)} PDF files")
        logger.info(f"Total: {len(md_files) + len(pdf_files)} files to process")

        return md_files, pdf_files

    def load_documents(
        self,
        md_files: List[Path],
        pdf_files: List[Path]
    ) -> List[Document]:
        """
        Load markdown and PDF files into Document objects.

        Args:
            md_files: List of markdown file paths
            pdf_files: List of PDF file paths

        Returns:
            List of loaded Document objects
        """
        logger.info("-" * 60)
        logger.info("Loading documents...")
        documents = []

        # Load markdown files
        for md_file in md_files:
            try:
                loader = TextLoader(str(md_file), encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = str(md_file)
                    doc.metadata['filename'] = md_file.name
                    doc.metadata['type'] = 'personal_note'
                documents.extend(docs)
                logger.info(f"  ✓ Loaded {md_file.name}")
            except Exception as e:
                logger.error(f"  ✗ Error loading {md_file}: {e}")

        # Load PDF files
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = str(pdf_file)
                    doc.metadata['filename'] = pdf_file.name
                    doc.metadata['type'] = 'technical_doc'
                documents.extend(docs)
                logger.info(f"  ✓ Loaded {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                logger.error(f"  ✗ Error loading {pdf_file}: {e}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using the retriever's chunking method.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents

        Raises:
            ValueError: If retriever not initialized
            AttributeError: If retriever doesn't support chunking

        Note: This method requires the retriever to have a chunk_documents() method.
              This is a limitation of the current design - ideally chunking would be
              separated into a DocumentChunker interface (see interfaces.py).
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call initialize_retriever() first.")

        # Check if retriever supports chunking (not part of Retriever interface)
        if not hasattr(self.retriever, 'chunk_documents'):
            raise AttributeError(
                f"Retriever {type(self.retriever).__name__} doesn't support chunking. "
                "Consider using a DocumentChunker implementation."
            )

        logger.info("-" * 60)
        logger.info("Chunking documents (512 chars, 50 overlap)...")
        chunks = self.retriever.chunk_documents(documents)
        logger.info(f"✅ Created {len(chunks)} chunks")
        return chunks

    def build_bm25_index(self, chunks: List[Document]) -> None:
        """
        Build BM25 index for keyword search.

        Args:
            chunks: List of document chunks

        Raises:
            ValueError: If retriever not initialized
            AttributeError: If retriever doesn't support BM25 indexing

        Note: This method requires the retriever to have a build_bm25_index() method.
              Not all Retriever implementations need BM25 (e.g., vector-only retrievers).
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call initialize_retriever() first.")

        # Check if retriever supports BM25 (not part of Retriever interface)
        if not hasattr(self.retriever, 'build_bm25_index'):
            logger.warning(
                f"Retriever {type(self.retriever).__name__} doesn't support BM25 indexing. Skipping..."
            )
            return

        logger.info("-" * 60)
        logger.info("Building BM25 index for keyword search...")
        self.retriever.build_bm25_index(chunks)
        logger.info("✅ BM25 index built")

    def add_to_vectorstore(self, chunks: List[Document]) -> None:
        """
        Add chunks to the vector store via the Retriever interface.

        Uses the Retriever.add_documents() method to maintain abstraction
        and avoid directly accessing the vectorstore implementation.

        Args:
            chunks: List of document chunks to add

        Raises:
            ValueError: If retriever not initialized
            RuntimeError: If adding documents fails (propagated from retriever)

        Example:
            ingester = DocumentIngester(docs_dir="./documents")
            ingester.initialize_retriever()
            chunks = ingester.process_documents(md_files, pdf_files)
            ingester.add_to_vectorstore(chunks)  # Uses Retriever interface
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call initialize_retriever() first.")

        logger.info("-" * 60)
        logger.info("Adding chunks to vector store (this may take a while)...")
        try:
            # Use Retriever interface instead of direct vectorstore access
            self.retriever.add_documents(chunks)
            logger.info("✅ Chunks added to vector store")
        except RuntimeError as e:
            # RuntimeError from retriever.add_documents() - already logged there
            logger.error(f"❌ Error adding to vector store: {e}")
            raise
        except Exception as e:
            # Unexpected error - log and wrap
            logger.error(f"❌ Unexpected error adding to vector store: {e}")
            raise RuntimeError(f"Failed to add chunks to vectorstore: {e}") from e

    def verify_ingestion(self) -> int:
        """
        Verify ingestion by counting total chunks in collection.

        Returns:
            Total number of chunks in the collection
        """
        logger.info("-" * 60)
        logger.info("Verifying ingestion...")
        collection = get_chroma_client().get_collection(self.collection_name)
        total_chunks = collection.count()
        logger.info(f"Total chunks in {self.collection_name}: {total_chunks}")
        return total_chunks

    def ingest(
        self,
        chroma_host: str = "chroma",
        chroma_port: int = 8000,
        re_ingest: bool = False
    ) -> Dict:
        """
        Run the full ingestion pipeline.

        Args:
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
            re_ingest: Whether to delete and recreate collection

        Returns:
            Dictionary with ingestion statistics

        Raises:
            FileNotFoundError: If documents directory not found
            ValueError: If no documents found
        """
        logger.info("=" * 60)
        logger.info("Document Ingestion - Baseline + Hybrid Search")
        logger.info("=" * 60)
        logger.info(f"Documents directory: {self.docs_dir}")
        logger.info(f"Re-ingest mode: {re_ingest}")
        logger.info("=" * 60)

        # Step 1: Handle re-ingestion
        if re_ingest:
            self.handle_reingestion()

        # Step 2: Initialize retriever
        self.initialize_retriever(chroma_host, chroma_port)

        # Step 3: Find documents
        md_files, pdf_files = self.find_documents()
        if len(md_files) + len(pdf_files) == 0:
            raise ValueError("No documents found to ingest!")

        # Step 4: Load documents
        documents = self.load_documents(md_files, pdf_files)
        if len(documents) == 0:
            raise ValueError("No documents were successfully loaded!")

        # Step 5: Chunk documents
        chunks = self.chunk_documents(documents)

        # Step 6: Build BM25 index
        self.build_bm25_index(chunks)

        # Step 7: Add to vector store
        self.add_to_vectorstore(chunks)

        # Step 8: Verify ingestion
        total_chunks = self.verify_ingestion()

        # Summary
        logger.info("=" * 60)
        logger.info("✅ INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Chunks created: {len(chunks)}")
        logger.info(f"Total in collection: {total_chunks}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info("=" * 60)

        return {
            'documents_processed': len(documents),
            'chunks_created': len(chunks),
            'total_in_collection': total_chunks,
            'collection_name': self.collection_name
        }


def main():
    """
    CLI entry point for document ingestion.

    Demonstrates two patterns:
    1. Dependency Injection: Create retriever and inject it
    2. Factory Pattern: Let DocumentIngester create it

    Currently using pattern #2 (factory) for simplicity.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into baseline_kb collection"
    )
    parser.add_argument(
        "--docs_dir",
        default="./documents",
        help="Directory containing documents to ingest (default: ./documents)"
    )
    parser.add_argument(
        "--re-ingest",
        action="store_true",
        help="Delete collection and re-ingest all documents from scratch"
    )
    args = parser.parse_args()

    # Option 1: Full Dependency Injection (more explicit)
    # retriever = create_baseline_retriever(
    #     chroma_host="chroma",
    #     chroma_port=8000,
    #     collection_name="baseline_kb"
    # )
    # ingester = DocumentIngester(
    #     docs_dir=args.docs_dir,
    #     retriever=retriever,
    #     collection_name="baseline_kb"
    # )

    # Option 2: Let ingester create retriever (simpler for CLI)
    ingester = DocumentIngester(
        docs_dir=args.docs_dir,
        collection_name="baseline_kb"
    )

    try:
        ingester.ingest(
            chroma_host="chroma",
            chroma_port=8000,
            re_ingest=args.re_ingest
        )
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise


if __name__ == '__main__':
    main()
