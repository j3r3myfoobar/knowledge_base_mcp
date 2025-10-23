"""
PATTERN: Modular RAG - Centralized database connections and model initialization.
Handles the connection to ChromaDB and the initialization of embedding models.
This module ensures that clients are initialized only once and shared across the application.
Implements singleton pattern for expensive resources (embeddings, cross-encoder models).
"""

import logging
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Import centralized configuration
from src.core.config import (
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
)

logger = logging.getLogger(__name__)


# --- Client Store ---
# PATTERN: Modular RAG - Singleton pattern for expensive model initialization

_clients = {}


def get_chroma_client():
    """Returns a thread-safe ChromaDB client."""
    if "client" not in _clients:
        try:
            _clients["client"] = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            logger.info(f"Successfully connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
        except Exception as e:
            logger.critical(f"Failed to connect to ChromaDB: {e}")
            raise
    return _clients["client"]


def get_embedding_function():
    """Initializes and returns the HuggingFace embedding model."""
    if "embedding_function" not in _clients:
        try:
            _clients["embedding_function"] = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.critical(f"Failed to load embedding model: {e}")
            raise
    return _clients["embedding_function"]


def get_vectorstore():
    """Initializes and returns the Chroma vector store."""
    if "vectorstore" not in _clients:
        try:
            _clients["vectorstore"] = Chroma(
                client=get_chroma_client(),
                collection_name=COLLECTION_NAME,
                embedding_function=get_embedding_function(),
            )
            logger.info(f"Successfully connected to vector store: {COLLECTION_NAME}")
        except Exception as e:
            logger.critical(f"Failed to connect to vector store: {e}")
            raise
    return _clients["vectorstore"]


def get_cross_encoder():
    """
    PATTERN: Corrective RAG - CrossEncoder model for document re-ranking.
    Initializes and returns the CrossEncoder model with graceful error handling.
    """
    if "cross_encoder" not in _clients:
        try:
            _clients["cross_encoder"] = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info(f"Successfully loaded CrossEncoder model: {CROSS_ENCODER_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            # PATTERN: Corrective RAG - Graceful fallback when model fails to load
            _clients["cross_encoder"] = None
    return _clients["cross_encoder"]
