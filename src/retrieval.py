#!/usr/bin/env python3
"""
Baseline RAG Retriever with Hybrid Search (BM25 + Vector)

This is a simplified RAG system that combines keyword matching (BM25) with
semantic search (vector similarity) for robust retrieval across diverse document types.

Architecture:
- Composed of smaller, focused components
- Hybrid search (BM25 + Vector fusion)
- Metadata filtering (type: personal_note vs technical_doc)
- Simple confidence scoring (hybrid score)
- Dependency injection for all components
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import numpy as np

from src.core.config import (
    EMBEDDING_MODEL,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    BATCH_SIZE,
)
from src.core.interfaces import Retriever, DocumentChunker, SearchIndex
from src.components.chunker import FixedSizeChunker
from src.components.bm25_index import BM25Index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineRetriever(Retriever):
    """
    Baseline RAG with Hybrid Search (BM25 + Vector).

    Composed of specialized components:
    - DocumentChunker handles chunking
    - BM25Index handles keyword search
    - VectorStore handles semantic search
    - BaselineRetriever orchestrates hybrid search
    """

    def __init__(
        self,
        vectorstore=None,
        embeddings=None,
        chroma_client=None,
        chunker: Optional[DocumentChunker] = None,
        bm25_index: Optional[SearchIndex] = None,
        collection_name: str = "baseline_kb",
        # Backward compatibility: if dependencies not provided, create them
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
    ):
        """
        Initialize baseline retriever with dependency injection.

        Args:
            vectorstore: Optional vectorstore instance
            embeddings: Optional embedding function
            chroma_client: Optional ChromaDB client
            chunker: Optional DocumentChunker instance
            bm25_index: Optional SearchIndex instance
            collection_name: ChromaDB collection name
            chroma_host: ChromaDB host (used only if not provided)
            chroma_port: ChromaDB port (used only if not provided)

        Example (with full DI - recommended):
            chunker = FixedSizeChunker(chunk_size=512)
            bm25_index = BM25Index()
            retriever = BaselineRetriever(
                vectorstore=vectorstore,
                chunker=chunker,
                bm25_index=bm25_index
            )

        Example (backward compatible - still works):
            retriever = BaselineRetriever(
                chroma_host="chroma",
                chroma_port=8000
            )
        """
        # Initialize embeddings (DI or create)
        if embeddings is not None:
            self.embeddings = embeddings
            logger.info("Using injected embeddings")
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info(f"Created embeddings: {EMBEDDING_MODEL}")

        # Initialize ChromaDB client (DI or create)
        if chroma_client is not None:
            self.chroma_client = chroma_client
            logger.info("Using injected ChromaDB client")
        else:
            import chromadb
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
            logger.info(f"Created ChromaDB client: {chroma_host}:{chroma_port}")

        # Initialize vectorstore (DI or create)
        if vectorstore is not None:
            self.vectorstore = vectorstore
            logger.info("Using injected vectorstore")
        else:
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Created vectorstore: {collection_name}")

        # NEW: Initialize chunker component (DI or create)
        if chunker is not None:
            self.chunker = chunker
            logger.info("Using injected chunker")
        else:
            self.chunker = FixedSizeChunker()
            logger.info("Created default FixedSizeChunker")

        # NEW: Initialize BM25 index component (DI or create)
        if bm25_index is not None:
            self.bm25_index = bm25_index
            logger.info("Using injected BM25 index")
        else:
            self.bm25_index = BM25Index()
            logger.info("Created default BM25Index")

        logger.info("Baseline retriever initialized with components")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using the injected chunker component.

        Args:
            documents: Raw documents to chunk

        Returns:
            Chunked documents with metadata
        """
        return self.chunker.chunk_documents(documents)

    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index using the injected BM25 index component.

        Args:
            documents: Documents to index
        """
        self.bm25_index.build_index(documents)

    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        Keyword search using BM25 index component.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        return self.bm25_index.search(query, top_k=top_k)

    def vector_search(
        self,
        query: str,
        top_k: int = 20,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Semantic search using vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
            filter=filter_metadata
        )

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and vector similarity.

        Uses weighted score fusion to combine normalized scores from both methods.
        BM25 scores are normalized by max score, vector distances are converted to
        similarities, then combined using configurable weights (default: 0.3 BM25 + 0.7 vector).

        Note: This is NOT Reciprocal Rank Fusion (RRF), which would use rank positions
        instead of raw scores. This approach preserves score magnitudes for better ranking.

        Args:
            query: Search query
            top_k: Number of final results
            bm25_weight: Weight for BM25 scores (0.0-1.0), default 0.3
            vector_weight: Weight for vector scores (0.0-1.0), default 0.7
            filter_metadata: Optional metadata filter

        Returns:
            List of documents with hybrid scores (weighted combination of BM25 and vector)
        """
        # Get BM25 results (top 20)
        bm25_results = self.bm25_search(query, top_k=20)

        # Get vector results (top 20)
        vector_results = self.vector_search(query, top_k=20, filter_metadata=filter_metadata)

        # Normalize scores and combine
        hybrid_scores = {}

        # Add BM25 scores
        max_bm25 = max([score for _, score in bm25_results]) if bm25_results else 1.0
        for doc, score in bm25_results:
            # Use global_chunk_id as stable identifier (set by chunker)
            # Fall back to chunk_id if global_chunk_id not present (backward compatibility)
            doc_id = doc.metadata.get('global_chunk_id', doc.metadata.get('chunk_id'))
            if doc_id is None:
                # If no chunk IDs at all, use source+content hash as last resort
                source = doc.metadata.get('source', 'unknown')
                content_hash = hash(doc.page_content)
                doc_id = f"{source}:{content_hash}"
                logger.warning(f"Document missing chunk IDs, using source+content hash: {doc_id}")

            normalized_score = score / max_bm25 if max_bm25 > 0 else 0.0
            hybrid_scores[doc_id] = {
                'document': doc,
                'score': bm25_weight * normalized_score
            }

        # Add vector scores
        # Note: Chroma returns distance (lower is better), convert to similarity
        max_dist = max([score for _, score in vector_results]) if vector_results else 1.0
        for doc, dist in vector_results:
            # Use global_chunk_id as stable identifier (set by chunker)
            # Fall back to chunk_id if global_chunk_id not present (backward compatibility)
            doc_id = doc.metadata.get('global_chunk_id', doc.metadata.get('chunk_id'))
            if doc_id is None:
                # If no chunk IDs at all, use source+content hash as last resort
                source = doc.metadata.get('source', 'unknown')
                content_hash = hash(doc.page_content)
                doc_id = f"{source}:{content_hash}"
                logger.warning(f"Document missing chunk IDs, using source+content hash: {doc_id}")

            # Convert distance to similarity (1 / (1 + distance))
            similarity = 1.0 / (1.0 + dist)

            if doc_id in hybrid_scores:
                hybrid_scores[doc_id]['score'] += vector_weight * similarity
            else:
                hybrid_scores[doc_id] = {
                    'document': doc,
                    'score': vector_weight * similarity
                }

        # Sort by hybrid score and return top-k
        sorted_results = sorted(
            hybrid_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        # Format results
        results = []
        for item in sorted_results:
            doc = item['document']
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'confidence': float(item['score'])
            })

        logger.info(f"Hybrid search returned {len(results)} results")
        return results

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Query the knowledge base.

        Args:
            query: Search query
            top_k: Number of results
            use_hybrid: Use hybrid search (True) or vector only (False)
            filter_type: Filter by document type ('personal_note', 'technical_doc')

        Returns:
            List of results with content, metadata, and confidence
        """
        # Prepare metadata filter
        filter_metadata = None
        if filter_type:
            filter_metadata = {'type': filter_type}

        if use_hybrid:
            return self.hybrid_search(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
        else:
            # Simple vector search
            results = self.vector_search(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata
            )

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'confidence': float(1.0 / (1.0 + score))  # Convert distance to confidence
                })

            return formatted_results

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store in batches.

        Implements the Retriever interface by delegating to the underlying
        vectorstore. This abstraction allows clients to add documents without
        directly accessing the vectorstore implementation.

        Documents are processed in batches to avoid exceeding ChromaDB's
        maximum batch size limit.

        Args:
            documents: List of documents to add to the knowledge base

        Raises:
            RuntimeError: If adding documents to vectorstore fails

        Example:
            retriever = create_baseline_retriever()
            docs = [Document(page_content="test", metadata={"source": "test.md"})]
            retriever.add_documents(docs)
        """
        try:
            total_docs = len(documents)
            logger.info(f"Adding {total_docs} documents to vectorstore in batches of {BATCH_SIZE}...")

            # Process documents in batches
            for i in range(0, total_docs, BATCH_SIZE):
                batch = documents[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
                self.vectorstore.add_documents(batch)
                logger.info(f"✅ Batch {batch_num}/{total_batches} added successfully")

            logger.info(f"✅ Successfully added all {total_docs} documents")
        except Exception as e:
            error_msg = f"Failed to add documents to vectorstore: {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def initialize(self) -> None:
        """
        Initialize retriever by building BM25 index from existing documents.

        Implements the Retriever interface by fetching documents from the
        internal vectorstore and building the BM25 index. This abstracts
        the implementation details from clients.

        Raises:
            RuntimeError: If initialization fails

        Example:
            retriever = create_baseline_retriever()
            retriever.initialize()  # Builds BM25 index from existing docs
            results = retriever.query("test query")  # Now ready to use
        """
        try:
            logger.info("Initializing retriever: building BM25 index from ChromaDB...")

            # Access internal collection to fetch existing documents
            collection = self.chroma_client.get_collection(name=self.vectorstore._collection.name)
            results = collection.get(include=['documents', 'metadatas'])

            if results['documents']:
                # Convert ChromaDB documents back to LangChain Documents
                documents = []
                for content, metadata in zip(results['documents'], results['metadatas']):
                    doc = Document(page_content=content, metadata=metadata or {})
                    documents.append(doc)

                # Build BM25 index using the SearchIndex component
                self.build_bm25_index(documents)
                logger.info(f"✅ BM25 index built with {len(documents)} documents")
            else:
                logger.warning("No documents found in collection, BM25 index will be empty")

        except Exception as e:
            error_msg = f"Failed to initialize retriever: {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e


def main():
    """Example usage."""
    retriever = BaselineRetriever()

    # Example: Query with hybrid search
    results = retriever.query(
        query="AWS Aurora failover",
        top_k=5,
        use_hybrid=True
    )

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Confidence: {result['confidence']:.3f}")
        print(f"   Source: {result['metadata'].get('source', 'unknown')}")
        print(f"   Type: {result['metadata'].get('type', 'unknown')}")
        print(f"   Content: {result['content'][:150]}...")


if __name__ == '__main__':
    main()
