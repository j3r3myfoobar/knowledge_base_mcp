"""
Reusable components for the knowledge base system.

This module contains focused, single-responsibility classes that can be
composed together to build retrieval systems.
"""

from src.components.chunker import FixedSizeChunker
from src.components.bm25_index import BM25Index

__all__ = [
    'FixedSizeChunker',
    'BM25Index',
]
