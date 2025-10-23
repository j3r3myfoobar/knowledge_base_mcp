"""
Centralized configuration for the knowledge base.

This configuration file contains settings for the baseline hybrid search system.
Only includes settings actually used by the current system.
"""

import os

# --- ChromaDB Settings ---
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000
COLLECTION_NAME = "baseline_kb"  # Current collection name

# --- Embedding Model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Alternative: "BAAI/bge-m3" for better accuracy (see docs/BGE_M3_UPGRADE_GUIDE.md)

# --- Document Processing ---
CHUNK_SIZE = 512         # Characters per chunk
CHUNK_OVERLAP = 50       # Overlap between chunks to preserve context

# --- Retrieval Settings ---
BM25_WEIGHT = 0.3        # Weight for BM25 keyword search in hybrid fusion
VECTOR_WEIGHT = 0.7      # Weight for vector semantic search in hybrid fusion
