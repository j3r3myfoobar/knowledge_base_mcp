"""
BM25 search index component.

Implements the SearchIndex interface for keyword-based search using BM25.
"""

import logging
from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document

from src.core.interfaces import SearchIndex

logger = logging.getLogger(__name__)


class BM25Index(SearchIndex):
    """
    BM25 keyword search index.

    Uses BM25Okapi algorithm for probabilistic keyword ranking.
    Provides fast keyword-based retrieval complementing semantic search.

    Single Responsibility: Only handles BM25 indexing and keyword search.
    """

    def __init__(self):
        """Initialize empty BM25 index."""
        self.index = None
        self.documents = []
        logger.info("BM25Index initialized (empty)")

    def build_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents to index

        Example:
            index = BM25Index()
            index.build_index(documents)
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")

        self.documents = documents

        # Tokenize documents for BM25 (simple whitespace tokenization)
        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        # Build index
        self.index = BM25Okapi(tokenized_docs)

        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        Search the BM25 index for relevant documents.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (Document, score) tuples, sorted by relevance

        Example:
            results = index.search("Python classes", top_k=5)
            for doc, score in results:
                print(f"Score: {score}, Content: {doc.page_content[:100]}")
        """
        if self.index is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        if not query.strip():
            logger.warning("Empty query provided")
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if idx < len(self.documents) and scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    def get_index_size(self) -> int:
        """
        Get the number of documents in the index.

        Returns:
            Number of indexed documents
        """
        return len(self.documents)

    def is_built(self) -> bool:
        """
        Check if index has been built.

        Returns:
            True if index is built and ready to search
        """
        return self.index is not None


class TFIDFIndex(SearchIndex):
    """
    TF-IDF search index (placeholder for future implementation).

    Would use Term Frequency-Inverse Document Frequency for ranking.

    Single Responsibility: Only handles TF-IDF indexing and search.
    """

    def __init__(self):
        """Initialize TF-IDF index."""
        logger.info("TFIDFIndex initialized (not yet implemented)")

    def build_index(self, documents: List[Document]) -> None:
        """Build TF-IDF index (not yet implemented)."""
        raise NotImplementedError(
            "TFIDFIndex is a placeholder. Use BM25Index for now."
        )

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """Search TF-IDF index (not yet implemented)."""
        raise NotImplementedError(
            "TFIDFIndex is a placeholder. Use BM25Index for now."
        )


# Example usage (for documentation):
"""
# Create and build index
index = BM25Index()
documents = [
    Document(page_content="Python programming language", metadata={}),
    Document(page_content="Java programming language", metadata={}),
]
index.build_index(documents)

# Search
results = index.search("Python classes", top_k=5)
for doc, score in results:
    print(f"Score: {score:.3f}, Content: {doc.page_content}")

# Use with retriever
retriever = BaselineRetriever(bm25_index=index, ...)

# Or inject via factory
from src.factories import create_bm25_index
index = create_bm25_index(documents)
retriever = BaselineRetriever(bm25_index=index, ...)
"""
