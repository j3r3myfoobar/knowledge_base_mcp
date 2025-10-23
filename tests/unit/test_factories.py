"""
Unit tests for src/factories.py

Tests factory functions and dependency injection patterns.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.factories import (
    create_embeddings,
    create_chroma_client,
    create_vectorstore,
    create_baseline_retriever,
    create_retriever_for_testing,
)
from src.retrieval import BaselineRetriever


class TestCreateEmbeddings:
    """Test create_embeddings factory."""

    @patch('src.factories.HuggingFaceEmbeddings')
    def test_create_embeddings_default_model(self, mock_embeddings_class):
        """Test creating embeddings with default model."""
        mock_instance = Mock()
        mock_embeddings_class.return_value = mock_instance

        result = create_embeddings()

        mock_embeddings_class.assert_called_once()
        assert result == mock_instance

    @patch('src.factories.HuggingFaceEmbeddings')
    def test_create_embeddings_custom_model(self, mock_embeddings_class):
        """Test creating embeddings with custom model."""
        mock_instance = Mock()
        mock_embeddings_class.return_value = mock_instance

        result = create_embeddings(model_name="BAAI/bge-m3")

        mock_embeddings_class.assert_called_once_with(model_name="BAAI/bge-m3")
        assert result == mock_instance


class TestCreateChromaClient:
    """Test create_chroma_client factory."""

    @patch('src.factories.chromadb.HttpClient')
    def test_create_chroma_client_default(self, mock_client_class):
        """Test creating ChromaDB client with defaults."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance

        result = create_chroma_client()

        mock_client_class.assert_called_once()
        assert result == mock_instance

    @patch('src.factories.chromadb.HttpClient')
    def test_create_chroma_client_custom(self, mock_client_class):
        """Test creating ChromaDB client with custom host/port."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance

        result = create_chroma_client(host="myhost", port=9000)

        mock_client_class.assert_called_once_with(host="myhost", port=9000)
        assert result == mock_instance


class TestCreateVectorstore:
    """Test create_vectorstore factory."""

    @patch('src.factories.create_chroma_client')
    @patch('src.factories.create_embeddings')
    @patch('src.factories.Chroma')
    def test_create_vectorstore_no_deps(
        self,
        mock_chroma_class,
        mock_create_embeddings,
        mock_create_client
    ):
        """Test creating vectorstore without providing dependencies."""
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_vectorstore = Mock()

        mock_create_client.return_value = mock_client
        mock_create_embeddings.return_value = mock_embeddings
        mock_chroma_class.return_value = mock_vectorstore

        result = create_vectorstore(collection_name="test_kb")

        # Should create dependencies
        mock_create_client.assert_called_once()
        mock_create_embeddings.assert_called_once()

        # Should create vectorstore with them
        mock_chroma_class.assert_called_once_with(
            client=mock_client,
            collection_name="test_kb",
            embedding_function=mock_embeddings
        )

        assert result == mock_vectorstore

    @patch('src.factories.Chroma')
    def test_create_vectorstore_with_deps(self, mock_chroma_class):
        """Test creating vectorstore with provided dependencies."""
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        result = create_vectorstore(
            client=mock_client,
            collection_name="test_kb",
            embedding_function=mock_embeddings
        )

        # Should use provided dependencies
        mock_chroma_class.assert_called_once_with(
            client=mock_client,
            collection_name="test_kb",
            embedding_function=mock_embeddings
        )

        assert result == mock_vectorstore


class TestCreateBaselineRetriever:
    """Test create_baseline_retriever factory."""

    @patch('src.factories.BaselineRetriever')
    @patch('src.factories.create_vectorstore')
    @patch('src.factories.create_chroma_client')
    @patch('src.factories.create_embeddings')
    def test_create_baseline_retriever_full_wiring(
        self,
        mock_create_embeddings,
        mock_create_client,
        mock_create_vectorstore,
        mock_retriever_class
    ):
        """Test that factory wires up all dependencies correctly."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_client = Mock()
        mock_vectorstore = Mock()
        mock_retriever = Mock()

        mock_create_embeddings.return_value = mock_embeddings
        mock_create_client.return_value = mock_client
        mock_create_vectorstore.return_value = mock_vectorstore
        mock_retriever_class.return_value = mock_retriever

        # Call factory
        result = create_baseline_retriever(
            chroma_host="testhost",
            chroma_port=7000,
            collection_name="test_collection",
            embedding_model="test-model"
        )

        # Verify dependency creation
        mock_create_embeddings.assert_called_once_with(model_name="test-model")
        mock_create_client.assert_called_once_with(host="testhost", port=7000)
        mock_create_vectorstore.assert_called_once_with(
            client=mock_client,
            collection_name="test_collection",
            embedding_function=mock_embeddings
        )

        # Verify retriever creation with injected dependencies (including chunker and bm25_index)
        call_args = mock_retriever_class.call_args
        assert call_args.kwargs['vectorstore'] == mock_vectorstore
        assert call_args.kwargs['embeddings'] == mock_embeddings
        assert call_args.kwargs['chroma_client'] == mock_client
        assert call_args.kwargs['collection_name'] == "test_collection"
        assert 'chunker' in call_args.kwargs
        assert 'bm25_index' in call_args.kwargs

        assert result == mock_retriever

    @patch('src.factories.create_baseline_retriever')
    def test_create_retriever_for_testing(self, mock_create_baseline):
        """Test factory for creating test retrievers."""
        mock_retriever = Mock()
        mock_create_baseline.return_value = mock_retriever

        result = create_retriever_for_testing()

        # Should call create_baseline_retriever with test config
        mock_create_baseline.assert_called_once_with(
            chroma_host="localhost",
            chroma_port=8000,
            collection_name="test_kb"
        )

        assert result == mock_retriever


class TestDependencyInjectionPatterns:
    """Test that dependency injection works in practice."""

    def test_retriever_accepts_mocked_dependencies(self):
        """Test that BaselineRetriever can accept mocked dependencies."""
        # Create mocks
        mock_vectorstore = Mock()
        mock_embeddings = Mock()
        mock_client = Mock()

        # Inject mocks into retriever
        retriever = BaselineRetriever(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
            chroma_client=mock_client,
            collection_name="test"
        )

        # Verify mocks are used
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.embeddings == mock_embeddings
        assert retriever.chroma_client == mock_client

    def test_retriever_can_query_with_mocked_vectorstore(self):
        """Test that retriever can execute query with mocked vectorstore."""
        # Setup mocks
        mock_vectorstore = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "test content"
        mock_doc.metadata = {'source': 'test.md', 'type': 'personal_note'}

        # Mock vectorstore response
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_doc, 0.5)
        ]

        mock_embeddings = Mock()
        mock_client = Mock()

        # Create retriever with mocks
        retriever = BaselineRetriever(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
            chroma_client=mock_client
        )

        # Execute query (vector-only to avoid BM25 complexity)
        results = retriever.query(
            query="test query",
            top_k=5,
            use_hybrid=False  # Vector only
        )

        # Verify query was executed
        mock_vectorstore.similarity_search_with_score.assert_called_once()
        assert len(results) == 1
        assert results[0]['content'] == "test content"

    @patch('src.factories.create_embeddings')
    @patch('src.factories.create_chroma_client')
    @patch('src.factories.create_vectorstore')
    def test_factory_allows_partial_mocking(
        self,
        mock_create_vectorstore,
        mock_create_client,
        mock_create_embeddings
    ):
        """Test that we can mock some dependencies and use real others."""
        # Mock only vectorstore, let others be created
        mock_vectorstore = Mock()
        mock_create_vectorstore.return_value = mock_vectorstore

        real_embeddings = Mock(spec=['embed_query', 'embed_documents'])
        real_client = Mock(spec=['get_collection'])
        mock_create_embeddings.return_value = real_embeddings
        mock_create_client.return_value = real_client

        # Create retriever using factory
        retriever = create_baseline_retriever()

        # Verify mixed dependencies
        assert retriever.vectorstore == mock_vectorstore  # Mocked
        assert retriever.embeddings == real_embeddings    # Real
        assert retriever.chroma_client == real_client     # Real


class TestBackwardCompatibility:
    """Test that old code still works (no breaking changes)."""

    def test_retriever_with_injected_dependencies(self):
        """Test that new DI pattern works."""
        # New style: dependency injection
        mock_vectorstore = Mock()
        mock_embeddings = Mock()
        mock_client = Mock()

        retriever = BaselineRetriever(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
            chroma_client=mock_client,
            collection_name="baseline_kb"
        )

        # Should use injected dependencies
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.embeddings == mock_embeddings
        assert retriever.chroma_client == mock_client

    def test_retriever_without_dependencies_uses_defaults(self):
        """Test that retriever can still be created without DI (backward compatible)."""
        # This test doesn't mock, so we can't actually instantiate
        # Just verify the interface accepts the old parameters
        import inspect
        sig = inspect.signature(BaselineRetriever.__init__)

        # Verify backward-compatible parameters exist
        assert 'chroma_host' in sig.parameters
        assert 'chroma_port' in sig.parameters
        assert 'collection_name' in sig.parameters

        # Verify new DI parameters exist
        assert 'vectorstore' in sig.parameters
        assert 'embeddings' in sig.parameters
        assert 'chroma_client' in sig.parameters
