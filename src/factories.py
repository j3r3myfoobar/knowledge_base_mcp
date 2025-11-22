"""
Factory functions for creating system components.

Centralizes object creation and dependency wiring, implementing the Factory pattern.
This makes it easy to create properly configured objects without repeating setup code.
"""

import logging
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.core.config import (
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.core.interfaces import Retriever, DocumentChunker, SearchIndex
from src.retrieval import BaselineRetriever
from src.components.chunker import FixedSizeChunker
from src.components.bm25_index import BM25Index

logger = logging.getLogger(__name__)


def create_chunker(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> DocumentChunker:
    """
    Factory function to create document chunker.

    Args:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap

    Returns:
        FixedSizeChunker instance
    """
    logger.info(f"Creating chunker: size={chunk_size}, overlap={chunk_overlap}")
    return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_bm25_index() -> SearchIndex:
    """
    Factory function to create BM25 search index.

    Returns:
        BM25Index instance (empty, needs to be built with documents)
    """
    logger.info("Creating BM25 index")
    return BM25Index()


def create_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Factory function to create embedding function.

    Args:
        model_name: Name of the HuggingFace model to use

    Returns:
        HuggingFaceEmbeddings instance
    """
    logger.info(f"Creating embeddings with model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


def create_chroma_client(
    host: str = CHROMA_HOST,
    port: int = CHROMA_PORT
):
    """
    Factory function to create ChromaDB client.

    Args:
        host: ChromaDB host
        port: ChromaDB port

    Returns:
        ChromaDB HttpClient instance
    """
    logger.info(f"Creating ChromaDB client: {host}:{port}")
    return chromadb.HttpClient(host=host, port=port)


def create_vectorstore(
    client=None,
    collection_name: str = COLLECTION_NAME,
    embedding_function=None
):
    """
    Factory function to create Chroma vectorstore.

    Args:
        client: Optional ChromaDB client (creates one if not provided)
        collection_name: Name of the collection
        embedding_function: Optional embedding function (creates one if not provided)

    Returns:
        Chroma vectorstore instance
    """
    if client is None:
        client = create_chroma_client()

    if embedding_function is None:
        embedding_function = create_embeddings()

    logger.info(f"Creating vectorstore: {collection_name}")
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function
    )


def create_baseline_retriever(
    chroma_host: str = CHROMA_HOST,
    chroma_port: int = CHROMA_PORT,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> Retriever:
    """
    Factory function to create BaselineRetriever with all dependencies.

    Creates and injects all component dependencies:
    - DocumentChunker for chunking
    - BM25Index for keyword search
    - VectorStore for semantic search

    Args:
        chroma_host: ChromaDB host
        chroma_port: ChromaDB port
        collection_name: ChromaDB collection name
        embedding_model: HuggingFace model name
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Configured BaselineRetriever instance with all components

    Example (production use):
        retriever = create_baseline_retriever(
            chroma_host="chroma",
            chroma_port=8000,
            collection_name="baseline_kb"
        )

    Example (testing with mocks):
        mock_chunker = Mock(spec=DocumentChunker)
        mock_bm25 = Mock(spec=SearchIndex)
        mock_vectorstore = Mock()

        retriever = BaselineRetriever(
            vectorstore=mock_vectorstore,
            chunker=mock_chunker,
            bm25_index=mock_bm25
        )
    """
    logger.info("Creating BaselineRetriever with all components...")

    # Create core dependencies
    embeddings = create_embeddings(model_name=embedding_model)
    chroma_client = create_chroma_client(host=chroma_host, port=chroma_port)
    vectorstore = create_vectorstore(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    # Create component dependencies
    chunker = create_chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    bm25_index = create_bm25_index()

    # Inject ALL dependencies into retriever
    retriever = BaselineRetriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
        chroma_client=chroma_client,
        chunker=chunker,          # NEW
        bm25_index=bm25_index,    # NEW
        collection_name=collection_name
    )

    logger.info("BaselineRetriever created with all components")
    return retriever


def create_retriever_for_testing():
    """
    Factory function to create a retriever suitable for testing.

    This is useful for integration tests that need a real retriever
    but with test-specific configuration.

    Returns:
        BaselineRetriever configured for testing
    """
    return create_baseline_retriever(
        chroma_host="localhost",
        chroma_port=8000,
        collection_name="test_kb"
    )


# Example usage (for documentation):
"""
# Production usage:
from src.factories import create_baseline_retriever

retriever = create_baseline_retriever()  # Uses default config
app = MCPApp(retriever=retriever)


# Testing usage with dependency injection:
from unittest.mock import Mock
from src.retrieval import BaselineRetriever

mock_vectorstore = Mock()
mock_embeddings = Mock()
mock_client = Mock()

retriever = BaselineRetriever(
    vectorstore=mock_vectorstore,
    embeddings=mock_embeddings,
    chroma_client=mock_client
)

# Now retriever uses mocks - no real ChromaDB needed!
ingester = DocumentIngester(retriever=retriever)
ingester.ingest()  # Works without real database


# Custom configuration:
retriever = create_baseline_retriever(
    chroma_host="my-chroma-server",
    chroma_port=9000,
    collection_name="my_collection",
    embedding_model="BAAI/bge-m3"  # Use BGE-M3 instead
)
"""
