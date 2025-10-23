"""
Document chunking components.

Implements the DocumentChunker interface with various chunking strategies.

SOLID Principles Applied:
- Single Responsibility: Only handles document chunking
- Open/Closed: Easy to add new chunking strategies
- Interface Segregation: Implements focused DocumentChunker interface
"""

import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from src.core.interfaces import DocumentChunker
from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class FixedSizeChunker(DocumentChunker):
    """
    Fixed-size document chunker using RecursiveCharacterTextSplitter.

    Splits documents into chunks of fixed character length with overlap,
    preserving document structure where possible.

    Single Responsibility: Only handles document chunking.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to split on (in order of preference)
        """
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

        logger.info(
            f"FixedSizeChunker initialized: size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into fixed-size pieces.

        Args:
            documents: Raw documents to chunk

        Returns:
            Chunked documents with metadata

        Metadata added to each chunk:
            - global_chunk_id: Sequential ID across all chunks (0, 1, 2, ...)
            - chunk_id: Position within source document (0, 1, 2, ...)
            - document_id: Index of source document in the input list
            - type: Inferred document type based on file extension

        Example:
            chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=50)
            chunks = chunker.chunk_documents(documents)
            # First chunk of second document:
            # chunk.metadata = {
            #     'global_chunk_id': 5,  # 6th chunk overall
            #     'chunk_id': 0,          # 1st chunk of its document
            #     'document_id': 1,       # From 2nd document
            #     'source': 'doc2.md',
            #     'type': 'personal_note'
            # }
        """
        logger.info(f"Chunking {len(documents)} documents...")

        # Split documents (preserves original metadata including source)
        chunks = self.text_splitter.split_documents(documents)

        # Track current document source and counters
        current_source = None
        per_doc_chunk_id = 0
        document_id = 0

        # Add chunk metadata with both global and per-document IDs
        for global_chunk_id, chunk in enumerate(chunks):
            # Detect when we move to a new document (source changes)
            source = chunk.metadata.get('source', '')
            if source != current_source:
                current_source = source
                per_doc_chunk_id = 0  # Reset per-document counter
                document_id += 1 if global_chunk_id > 0 else 0  # Increment document_id

            # Add both global and per-document chunk IDs
            chunk.metadata['global_chunk_id'] = global_chunk_id
            chunk.metadata['chunk_id'] = per_doc_chunk_id
            chunk.metadata['document_id'] = document_id

            # Infer document type from file extension
            if source.endswith('.md'):
                chunk.metadata['type'] = 'personal_note'
            elif source.endswith('.pdf'):
                chunk.metadata['type'] = 'technical_doc'
            else:
                chunk.metadata['type'] = 'unknown'

            # Increment per-document counter
            per_doc_chunk_id += 1

        logger.info(
            f"Created {len(chunks)} chunks from {len(documents)} documents"
        )
        return chunks


class SemanticChunker(DocumentChunker):
    """
    Semantic document chunker (placeholder for future implementation).

    Would use sentence embeddings and similarity to create semantically
    coherent chunks instead of fixed-size chunks.

    Single Responsibility: Only handles semantic chunking.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize semantic chunker.

        Args:
            similarity_threshold: Minimum similarity to keep sentences together
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"SemanticChunker initialized (not yet implemented)")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents semantically (not yet implemented).

        Args:
            documents: Raw documents to chunk

        Returns:
            Semantically chunked documents

        Raises:
            NotImplementedError: This is a placeholder
        """
        raise NotImplementedError(
            "SemanticChunker is a placeholder for future implementation. "
            "Use FixedSizeChunker for now."
        )


# Example usage (for documentation):
"""
# Create chunker
chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=50)

# Chunk documents
documents = [Document(page_content="Long text...", metadata={})]
chunks = chunker.chunk_documents(documents)

# Use in retriever
retriever = BaselineRetriever(chunker=chunker, ...)

# Or inject via factory
from src.factories import create_chunker
chunker = create_chunker(chunk_size=1024)  # Custom size
retriever = BaselineRetriever(chunker=chunker, ...)
"""
