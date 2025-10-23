"""
Unit tests for BaselineRetriever using TDD approach with Osherove naming convention.

Naming Convention: [UnitOfWork_StateUnderTest_ExpectedBehavior]
- UnitOfWork: The method or behavior being tested
- StateUnderTest: The condition or input scenario
- ExpectedBehavior: What should happen

Example: ChunkDocuments_LongDocument_SplitsIntoMultipleChunks

These tests follow TDD principles:
1. Clear naming that documents the requirement
2. Arrange-Act-Assert structure
3. No advanced Python features
4. Uses shared fixtures and helpers
5. Each test verifies ONE specific behavior
"""
import pytest
from unittest.mock import Mock
from langchain.docstore.document import Document

from src.retrieval import BaselineRetriever
from tests.unit.helpers import (
    create_retriever_with_real_components,
    create_fully_mocked_retriever,
    create_mock_chroma_collection_with_docs,
    create_sample_chunks,
)


class TestBaselineRetrieverInitialization:
    """Test retriever initialization and configuration."""

    def test_Initialization_NoParameters_HasRequiredMethods(self):
        """Retriever class should have all required public methods."""
        # This is a smoke test - just verify the interface exists
        assert hasattr(BaselineRetriever, 'query')
        assert hasattr(BaselineRetriever, 'add_documents')
        assert hasattr(BaselineRetriever, 'initialize')
        assert hasattr(BaselineRetriever, 'chunk_documents')
        assert hasattr(BaselineRetriever, 'build_bm25_index')

    def test_Constructor_MockedDependencies_AcceptsAllDependencies(self, mock_retriever_dependencies):
        """Retriever should accept all dependencies via dependency injection."""
        # Arrange & Act
        retriever = BaselineRetriever(**mock_retriever_dependencies)

        # Assert
        assert retriever.vectorstore is not None
        assert retriever.embeddings is not None
        assert retriever.chroma_client is not None
        assert retriever.chunker is not None
        assert retriever.bm25_index is not None


class TestDocumentChunking:
    """
    Test document chunking behavior.

    These tests verify that the retriever correctly delegates to its chunker component.
    """

    def test_ChunkDocuments_LongDocument_SplitsIntoMultipleChunks(self, long_document):
        """Long documents should be split into multiple smaller chunks."""
        # Arrange
        retriever = create_retriever_with_real_components()

        # Act
        chunks = retriever.chunk_documents([long_document])

        # Assert
        assert len(chunks) > 1, "Long document should produce multiple chunks"
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all(hasattr(chunk, 'page_content') for chunk in chunks)

    def test_ChunkDocuments_LongDocument_AddsChunkIdMetadata(self, long_document):
        """Chunker should add chunk_id metadata to each chunk."""
        # Arrange
        retriever = create_retriever_with_real_components()

        # Act
        chunks = retriever.chunk_documents([long_document])

        # Assert
        assert all('chunk_id' in chunk.metadata for chunk in chunks), \
            "Each chunk should have a chunk_id"
        assert all('global_chunk_id' in chunk.metadata for chunk in chunks), \
            "Each chunk should have a global_chunk_id"

    def test_ChunkDocuments_MarkdownFile_InfersPersonalNoteType(self, markdown_document, pdf_document):
        """Chunker should infer document type from file extension."""
        # Arrange
        retriever = create_retriever_with_real_components()

        # Act
        md_chunks = retriever.chunk_documents([markdown_document])
        pdf_chunks = retriever.chunk_documents([pdf_document])

        # Assert
        assert md_chunks[0].metadata['type'] == 'personal_note', \
            ".md files should be typed as personal_note"
        assert pdf_chunks[0].metadata['type'] == 'technical_doc', \
            ".pdf files should be typed as technical_doc"

    def test_ChunkDocuments_MockedChunker_DelegatesToComponent(self, sample_documents):
        """Retriever should delegate to its injected chunker component."""
        # Arrange
        mock_chunker = Mock()
        mock_chunker.chunk_documents.return_value = sample_documents
        retriever = create_fully_mocked_retriever(chunker=mock_chunker)

        # Act
        result = retriever.chunk_documents(sample_documents)

        # Assert
        mock_chunker.chunk_documents.assert_called_once_with(sample_documents)
        assert result == sample_documents


class TestBM25IndexBuilding:
    """
    Test BM25 index building behavior.

    BM25 is used for keyword-based search in hybrid retrieval.
    """

    def test_BuildBM25Index_ValidDocuments_BuildsIndexSuccessfully(self, sample_documents):
        """Should build BM25 index from provided documents."""
        # Arrange
        retriever = create_retriever_with_real_components()

        # Act
        retriever.build_bm25_index(sample_documents)

        # Assert
        assert retriever.bm25_index.is_built(), "Index should be built"
        assert retriever.bm25_index.get_index_size() == len(sample_documents)

    def test_BuildBM25Index_MockedComponent_DelegatesToBM25Index(self, sample_documents):
        """Retriever should delegate to its injected BM25 index component."""
        # Arrange
        mock_bm25 = Mock()
        retriever = create_fully_mocked_retriever(bm25_index=mock_bm25)

        # Act
        retriever.build_bm25_index(sample_documents)

        # Assert
        mock_bm25.build_index.assert_called_once_with(sample_documents)


class TestBM25Search:
    """Test BM25 keyword search functionality."""

    def test_BM25Search_MatchingQuery_ReturnsRelevantDocuments(self):
        """BM25 search should return documents matching query keywords."""
        # Arrange
        retriever = create_retriever_with_real_components()
        docs = [
            Document(page_content="Python programming language", metadata={}),
            Document(page_content="Java programming language", metadata={}),
            Document(page_content="Cooking recipes", metadata={}),
        ]
        retriever.build_bm25_index(docs)

        # Act
        results = retriever.bm25_search("Python programming", top_k=2)

        # Assert
        assert len(results) <= 2, "Should respect top_k limit"
        assert all(isinstance(r, tuple) for r in results), "Should return (doc, score) tuples"
        assert all(len(r) == 2 for r in results), "Each result should be (doc, score)"

    def test_BM25Search_MockedComponent_DelegatesToBM25Index(self):
        """Retriever should delegate search to its BM25 index component."""
        # Arrange
        mock_bm25 = Mock()
        expected_results = [(Mock(), 0.8)]
        mock_bm25.search.return_value = expected_results
        retriever = create_fully_mocked_retriever(bm25_index=mock_bm25)

        # Act
        results = retriever.bm25_search("test query", top_k=5)

        # Assert
        mock_bm25.search.assert_called_once_with("test query", top_k=5)
        assert results == expected_results


class TestVectorSearch:
    """Test semantic vector search functionality."""

    def test_VectorSearch_ValidQuery_DelegatesToVectorstore(self, sample_documents):
        """Vector search should delegate to vectorstore component."""
        # Arrange
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.5)
        ]
        retriever = create_fully_mocked_retriever(vectorstore=mock_vectorstore)

        # Act
        results = retriever.vector_search("test query", top_k=3)

        # Assert
        mock_vectorstore.similarity_search_with_score.assert_called_once()
        assert len(results) == 1

    def test_VectorSearch_WithMetadataFilter_PassesFilterToVectorstore(self):
        """Vector search should pass metadata filter to vectorstore."""
        # Arrange
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = []
        retriever = create_fully_mocked_retriever(vectorstore=mock_vectorstore)

        # Act
        retriever.vector_search(
            "test query",
            top_k=5,
            filter_metadata={'type': 'technical_doc'}
        )

        # Assert
        call_args = mock_vectorstore.similarity_search_with_score.call_args
        assert call_args.kwargs['filter'] == {'type': 'technical_doc'}


class TestQueryMethod:
    """
    Test the main query() method that orchestrates hybrid search.

    This is the primary public interface used by clients.
    """

    def test_Query_HybridSearchEnabled_UsesHybridSearch(self):
        """Query should use hybrid search (BM25 + vector) by default."""
        # Arrange
        mock_results = [{'content': 'test', 'metadata': {}, 'confidence': 0.9}]
        retriever = create_fully_mocked_retriever()
        retriever.hybrid_search = Mock(return_value=mock_results)

        # Act
        results = retriever.query("test query", top_k=5, use_hybrid=True)

        # Assert
        retriever.hybrid_search.assert_called_once()
        assert results == mock_results

    def test_Query_VectorOnlyMode_UsesVectorSearchOnly(self, mock_vector_search_results):
        """Query should support vector-only search when use_hybrid=False."""
        # Arrange
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = mock_vector_search_results
        retriever = create_fully_mocked_retriever(vectorstore=mock_vectorstore)

        # Act
        results = retriever.query("test query", top_k=5, use_hybrid=False)

        # Assert
        mock_vectorstore.similarity_search_with_score.assert_called_once()
        assert len(results) == len(mock_vector_search_results)
        assert all('confidence' in r for r in results)

    def test_Query_EmptyQueryString_ReturnsEmptyList(self):
        """Query should handle empty query string gracefully."""
        # Arrange
        retriever = create_fully_mocked_retriever()
        retriever.hybrid_search = Mock(return_value=[])

        # Act
        results = retriever.query("", top_k=5)

        # Assert
        assert results == []


class TestAddDocuments:
    """Test adding documents to the vectorstore."""

    def test_AddDocuments_ValidDocuments_DelegatesToVectorstore(self, sample_documents):
        """Should delegate document addition to vectorstore."""
        # Arrange
        mock_vectorstore = Mock()
        retriever = create_fully_mocked_retriever(vectorstore=mock_vectorstore)

        # Act
        retriever.add_documents(sample_documents)

        # Assert
        mock_vectorstore.add_documents.assert_called_once_with(sample_documents)

    def test_AddDocuments_VectorstoreFailure_RaisesRuntimeError(self, sample_documents):
        """Should raise RuntimeError if vectorstore add fails."""
        # Arrange
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents.side_effect = Exception("DB connection failed")
        retriever = create_fully_mocked_retriever(vectorstore=mock_vectorstore)

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to add documents"):
            retriever.add_documents(sample_documents)


class TestInitializeMethod:
    """
    Test the initialize() method that builds indexes from existing data.

    This method is called at server startup to build BM25 index from documents
    already in ChromaDB.
    """

    def test_Initialize_ExistingDocuments_BuildsBM25IndexFromChromaDB(self):
        """Initialize should fetch existing docs and build BM25 index."""
        # Arrange
        existing_docs = [
            {'content': 'Python programming', 'metadata': {'source': 'py.md'}},
            {'content': 'Java programming', 'metadata': {'source': 'java.md'}},
            {'content': 'Rust programming', 'metadata': {'source': 'rust.md'}},
        ]
        mock_client, mock_collection = create_mock_chroma_collection_with_docs(existing_docs)

        mock_vectorstore = Mock()
        mock_vectorstore._collection.name = "test_collection"

        mock_bm25 = Mock()
        retriever = create_fully_mocked_retriever(
            chroma_client=mock_client,
            vectorstore=mock_vectorstore,
            bm25_index=mock_bm25
        )

        # Act
        retriever.initialize()

        # Assert
        mock_client.get_collection.assert_called_once_with(name="test_collection")
        mock_collection.get.assert_called_once_with(include=['documents', 'metadatas'])
        mock_bm25.build_index.assert_called_once()

        # Verify documents were converted correctly
        built_docs = mock_bm25.build_index.call_args[0][0]
        assert len(built_docs) == 3
        assert built_docs[0].page_content == 'Python programming'
        assert built_docs[0].metadata == {'source': 'py.md'}

    def test_Initialize_EmptyCollection_DoesNotBuildIndex(self):
        """Initialize should not fail if collection is empty."""
        # Arrange
        mock_client, mock_collection = create_mock_chroma_collection_with_docs([])

        mock_vectorstore = Mock()
        mock_vectorstore._collection.name = "empty_collection"

        mock_bm25 = Mock()
        retriever = create_fully_mocked_retriever(
            chroma_client=mock_client,
            vectorstore=mock_vectorstore,
            bm25_index=mock_bm25
        )

        # Act - should not raise
        retriever.initialize()

        # Assert
        mock_bm25.build_index.assert_not_called()

    def test_Initialize_ChromaDBFailure_RaisesRuntimeError(self):
        """Initialize should raise RuntimeError if ChromaDB access fails."""
        # Arrange
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Connection failed")

        mock_vectorstore = Mock()
        mock_vectorstore._collection.name = "test_collection"

        retriever = create_fully_mocked_retriever(
            chroma_client=mock_client,
            vectorstore=mock_vectorstore
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to initialize retriever"):
            retriever.initialize()


# --- Module-level tests (not class-based) ---

def test_BaselineRetriever_Constructor_AcceptsCollectionNameParameter():
    """Retriever should accept collection_name parameter."""
    # This is a smoke test for the interface
    import inspect
    sig = inspect.signature(BaselineRetriever.__init__)
    assert 'collection_name' in sig.parameters
