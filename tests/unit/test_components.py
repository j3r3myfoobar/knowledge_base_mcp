"""
Unit tests for component classes using TDD approach with Osherove naming convention.

Naming Convention: [UnitOfWork_StateUnderTest_ExpectedBehavior]
- UnitOfWork: The method or behavior being tested
- StateUnderTest: The condition or input scenario
- ExpectedBehavior: What should happen

Example: ChunkDocuments_LongDocument_SplitsIntoMultipleChunks

Tests for FixedSizeChunker and BM25Index components that follow TDD principles.
"""
import pytest
from langchain.docstore.document import Document

from src.components.chunker import FixedSizeChunker
from src.components.bm25_index import BM25Index
from tests.unit.helpers import (
    create_retriever_with_real_components,
    create_fully_mocked_retriever,
)


class TestFixedSizeChunker:
    """
    Test the FixedSizeChunker component.

    This chunker splits documents into fixed-size pieces with overlap.
    It's the simplest chunking strategy and works well for most documents.
    """

    def test_ChunkDocuments_LongDocument_SplitsIntoMultipleChunks(self, long_document):
        """Long documents should be split into multiple smaller chunks."""
        # Arrange
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)

        # Act
        chunks = chunker.chunk_documents([long_document])

        # Assert
        assert len(chunks) > 1, "Long document should produce multiple chunks"
        assert all(isinstance(c, Document) for c in chunks), "All chunks should be Documents"

    def test_ChunkDocuments_ConfiguredSize_RespectsChunkSizeParameter(self):
        """Chunker should respect the configured chunk_size."""
        # Arrange
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        long_text = "x" * 500  # 500 characters
        doc = Document(page_content=long_text, metadata={'source': 'test.md'})

        # Act
        chunks = chunker.chunk_documents([doc])

        # Assert
        for chunk in chunks:
            # Chunks should be roughly chunk_size (±overlap)
            assert len(chunk.page_content) <= 110, \
                f"Chunk too long: {len(chunk.page_content)} chars"

    def test_ChunkDocuments_MarkdownFile_InfersPersonalNoteType(self, markdown_document):
        """Markdown files should be classified as personal_note type."""
        # Arrange
        chunker = FixedSizeChunker()

        # Act
        chunks = chunker.chunk_documents([markdown_document])

        # Assert
        assert all(c.metadata['type'] == 'personal_note' for c in chunks), \
            ".md files should be typed as personal_note"

    def test_ChunkDocuments_PDFFile_InfersTechnicalDocType(self, pdf_document):
        """PDF files should be classified as technical_doc type."""
        # Arrange
        chunker = FixedSizeChunker()

        # Act
        chunks = chunker.chunk_documents([pdf_document])

        # Assert
        assert all(c.metadata['type'] == 'technical_doc' for c in chunks), \
            ".pdf files should be typed as technical_doc"

    def test_ChunkDocuments_AnyDocument_AddsAllRequiredChunkIds(self):
        """Each chunk should have chunk_id, global_chunk_id, and document_id."""
        # Arrange
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=5)
        doc = Document(page_content="x" * 200, metadata={'source': 'test.md'})

        # Act
        chunks = chunker.chunk_documents([doc])

        # Assert
        assert all('chunk_id' in c.metadata for c in chunks), \
            "Each chunk needs chunk_id"
        assert all('global_chunk_id' in c.metadata for c in chunks), \
            "Each chunk needs global_chunk_id"
        assert all('document_id' in c.metadata for c in chunks), \
            "Each chunk needs document_id"

    def test_ChunkDocuments_MultipleDocuments_ResetsChunkIdPerDocument(self):
        """chunk_id should restart at 0 for each new document."""
        # Arrange
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=5)
        docs = [
            Document(page_content="A" * 150, metadata={'source': 'doc1.md'}),
            Document(page_content="B" * 150, metadata={'source': 'doc2.md'}),
            Document(page_content="C" * 100, metadata={'source': 'doc3.pdf'}),
        ]

        # Act
        chunks = chunker.chunk_documents(docs)

        # Assert: global_chunk_id should be sequential
        global_ids = [c.metadata['global_chunk_id'] for c in chunks]
        assert global_ids == list(range(len(chunks))), \
            "global_chunk_id should be sequential across all chunks"

        # Assert: chunk_id should restart for each document
        doc1_chunks = [c for c in chunks if c.metadata['source'] == 'doc1.md']
        doc1_chunk_ids = [c.metadata['chunk_id'] for c in doc1_chunks]
        assert doc1_chunk_ids == list(range(len(doc1_chunks))), \
            "chunk_id should restart at 0 for each document"

        doc2_chunks = [c for c in chunks if c.metadata['source'] == 'doc2.md']
        doc2_chunk_ids = [c.metadata['chunk_id'] for c in doc2_chunks]
        assert doc2_chunk_ids == list(range(len(doc2_chunks))), \
            "chunk_id should restart at 0 for doc2"

    def test_ChunkDocuments_MultipleDocuments_AssignsIncrementalDocumentIds(self):
        """document_id should increment for each new source document."""
        # Arrange
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=5)
        docs = [
            Document(page_content="A" * 150, metadata={'source': 'doc1.md'}),
            Document(page_content="B" * 150, metadata={'source': 'doc2.md'}),
        ]

        # Act
        chunks = chunker.chunk_documents(docs)

        # Assert
        doc1_chunks = [c for c in chunks if c.metadata['source'] == 'doc1.md']
        doc2_chunks = [c for c in chunks if c.metadata['source'] == 'doc2.md']

        assert all(c.metadata['document_id'] == 0 for c in doc1_chunks), \
            "First document should have document_id=0"
        assert all(c.metadata['document_id'] == 1 for c in doc2_chunks), \
            "Second document should have document_id=1"


class TestBM25Index:
    """
    Test the BM25Index component.

    BM25 is a keyword-based search algorithm used for finding documents
    that contain specific terms. It complements vector search in hybrid retrieval.
    """

    def test_BuildIndex_ValidDocuments_BuildsIndexSuccessfully(self, sample_documents):
        """Should build BM25 index from provided documents."""
        # Arrange
        index = BM25Index()

        # Act
        index.build_index(sample_documents)

        # Assert
        assert index.is_built(), "Index should report as built"
        assert index.get_index_size() == len(sample_documents), \
            f"Index should contain {len(sample_documents)} documents"

    def test_Constructor_NewIndex_ReturnsEmptyIndex(self):
        """Newly created index should be empty."""
        # Arrange & Act
        index = BM25Index()

        # Assert
        assert not index.is_built(), "New index should not be built"
        assert index.get_index_size() == 0, "New index should be empty"

    def test_Search_MatchingQuery_ReturnsRelevantDocuments(self):
        """Search should return documents matching the query."""
        # Arrange
        index = BM25Index()
        docs = [
            Document(page_content="python programming language", metadata={}),
            Document(page_content="java programming language", metadata={}),
            Document(page_content="javascript web development", metadata={}),
        ]
        index.build_index(docs)

        # Act
        results = index.search("python", top_k=2)

        # Assert
        assert len(results) > 0, "Should return results for matching query"
        assert all(isinstance(r, tuple) for r in results), \
            "Results should be (doc, score) tuples"
        assert all(len(r) == 2 for r in results), \
            "Each result should be (doc, score)"

    def test_Search_WithTopK_RespectsTopKLimit(self):
        """Search should not return more than top_k results."""
        # Arrange
        index = BM25Index()
        docs = [
            Document(page_content=f"python document {i}", metadata={})
            for i in range(10)
        ]
        index.build_index(docs)

        # Act
        results = index.search("python", top_k=3)

        # Assert
        assert len(results) <= 3, "Should respect top_k limit"

    def test_Search_UnbuiltIndex_ReturnsEmptyList(self):
        """Searching unbuilt index should return empty list."""
        # Arrange
        index = BM25Index()

        # Act
        results = index.search("python", top_k=5)

        # Assert
        assert results == [], "Unbuilt index should return empty results"

    def test_Search_EmptyQuery_ReturnsEmptyList(self):
        """Empty query should return empty results."""
        # Arrange
        index = BM25Index()
        docs = [Document(page_content="test", metadata={})]
        index.build_index(docs)

        # Act
        results = index.search("", top_k=5)

        # Assert
        assert results == [], "Empty query should return no results"


class TestComponentIntegration:
    """
    Test that components integrate correctly with BaselineRetriever.

    These tests verify the Dependency Inversion Principle - the retriever
    depends on abstract interfaces, not concrete implementations.
    """

    def test_Retriever_InjectedChunker_UsesInjectedComponent(self, sample_documents):
        """Retriever should use the chunker passed via dependency injection."""
        # Arrange
        from unittest.mock import Mock
        mock_chunker = Mock()
        mock_chunker.chunk_documents.return_value = sample_documents

        retriever = create_fully_mocked_retriever(chunker=mock_chunker)

        # Act
        result = retriever.chunk_documents(sample_documents)

        # Assert
        mock_chunker.chunk_documents.assert_called_once_with(sample_documents)
        assert result == sample_documents, "Should return chunker's result"

    def test_Retriever_InjectedBM25Index_UsesInjectedComponent(self):
        """Retriever should use the BM25 index passed via dependency injection."""
        # Arrange
        from unittest.mock import Mock
        mock_bm25 = Mock()
        expected_results = [(Mock(), 0.8)]
        mock_bm25.search.return_value = expected_results

        retriever = create_fully_mocked_retriever(bm25_index=mock_bm25)

        # Act
        results = retriever.bm25_search("query", top_k=5)

        # Assert
        mock_bm25.search.assert_called_once_with("query", top_k=5)
        assert results == expected_results, "Should return BM25's results"

    def test_Retriever_DifferentChunkerSizes_ProducesDifferentChunkCounts(self):
        """Should be able to swap different chunker implementations."""
        # Arrange
        chunker1 = FixedSizeChunker(chunk_size=100)
        chunker2 = FixedSizeChunker(chunk_size=200)

        retriever1 = create_retriever_with_real_components(chunker=chunker1)
        retriever2 = create_retriever_with_real_components(chunker=chunker2)

        long_doc = Document(page_content="x" * 500, metadata={'source': 'test.md'})

        # Act
        chunks1 = retriever1.chunk_documents([long_doc])
        chunks2 = retriever2.chunk_documents([long_doc])

        # Assert
        # Smaller chunk size should produce more chunks
        assert len(chunks1) > len(chunks2), \
            "Smaller chunk_size should produce more chunks"
