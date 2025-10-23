"""
Core interfaces for the knowledge base system.

These abstract base classes define the contracts that implementations must follow,
enabling dependency inversion and making the system more testable and extensible.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from langchain.docstore.document import Document


class Retriever(ABC):
    """
    Interface for document retrieval systems.

    This is the main public interface that clients should depend on.
    Different retrieval strategies (baseline, HyDE, ColBERT) can implement this.
    """

    @abstractmethod
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Query the knowledge base and return relevant documents.

        Args:
            query: Search query or question
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search
            filter_type: Optional filter by document type

        Returns:
            List of dictionaries with 'content', 'metadata', 'confidence'
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever's underlying vector store.

        This method abstracts the vector store implementation, allowing
        clients to add documents without directly accessing the vectorstore.

        Args:
            documents: List of documents to add to the knowledge base

        Raises:
            RuntimeError: If adding documents fails

        Example:
            retriever = create_baseline_retriever()
            docs = [Document(page_content="test", metadata={})]
            retriever.add_documents(docs)  # Abstracts vectorstore access
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the retriever by building necessary indexes.

        This method should be called once after the retriever is created
        to build any indexes (e.g., BM25) from existing documents in the
        vector store. This abstracts the implementation details of index
        construction from clients.

        Raises:
            RuntimeError: If initialization fails

        Example:
            retriever = create_baseline_retriever()
            retriever.initialize()  # Builds BM25 index from existing docs
            results = retriever.query("test query")  # Now ready to use
        """
        pass


class DocumentChunker(ABC):
    """
    Interface for document chunking strategies.

    Separates chunking concern from retrieval, allowing different strategies
    (fixed-size, semantic, recursive, etc.) to be swapped easily.
    """

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with metadata
        """
        pass


class SearchIndex(ABC):
    """
    Interface for search index implementations.

    Abstracts different search backends (BM25, TF-IDF, Elasticsearch, etc.)
    """

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """
        Build search index from documents.

        Args:
            documents: List of documents to index
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        Search the index for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        pass


class VectorStore(ABC):
    """
    Interface for vector database implementations.

    Abstracts different vector stores (ChromaDB, Qdrant, Pinecone, Weaviate, etc.)
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (document, distance) tuples
        """
        pass


class EmbeddingFunction(ABC):
    """
    Interface for embedding models.

    Abstracts different embedding implementations (HuggingFace, OpenAI, Cohere, etc.)
    """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        pass


class SearchStrategy(ABC):
    """
    Interface for different search strategies.

    Enables Strategy pattern for search methods (hybrid, vector-only, HyDE, ColBERT, etc.)
    """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Execute search using this strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of result dictionaries with content, metadata, confidence
        """
        pass


# Example Usage (for documentation):
"""
# Before (tightly coupled):
retriever = BaselineRetriever(
    chroma_host="chroma",
    chroma_port=8000,
    collection_name="baseline_kb"
)
app = MCPApp(retriever)  # Tightly coupled to BaselineRetriever

# After (loosely coupled via interface):
retriever: Retriever = BaselineRetriever(...)  # Type hint to interface
app = MCPApp(retriever)  # Depends on Retriever interface

# Can easily swap implementations:
retriever: Retriever = HyDERetriever(...)
retriever: Retriever = ColBERTRetriever(...)
retriever: Retriever = MockRetriever(...)  # For testing
"""
