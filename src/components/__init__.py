"""
Reusable components for the knowledge base system.

This module contains focused, single-responsibility classes that can be
composed together to build retrieval systems.

SOLID Principles Applied:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Easy to extend by creating new implementations
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Classes depend on abstractions
"""

from src.components.chunker import FixedSizeChunker
from src.components.bm25_index import BM25Index

__all__ = [
    'FixedSizeChunker',
    'BM25Index',
]
