"""
Unit tests for database connection management.

Tests the singleton pattern for database connections and model initialization
including ChromaDB client, embeddings, and cross-encoder.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core import database


class TestDatabaseConnections:
    """Test suite for database connection management."""

    def setup_method(self):
        """Reset the client store before each test."""
        database._clients.clear()

    @patch("chromadb.HttpClient")
    def test_get_chroma_client_success(self, mock_http_client):
        """Test successful ChromaDB client initialization and singleton behavior."""
        mock_client = MagicMock()
        mock_http_client.return_value = mock_client

        client = database.get_chroma_client()
        client2 = database.get_chroma_client()  # Test singleton

        assert client == mock_client
        assert client2 == mock_client
        # Should only be called once due to singleton
        mock_http_client.assert_called_once_with(host="chroma", port=8000)

    @patch("chromadb.HttpClient")
    def test_get_chroma_client_failure(self, mock_http_client):
        """Test ChromaDB client initialization failure."""
        mock_http_client.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            database.get_chroma_client()

    @patch("src.core.database.HuggingFaceEmbeddings")
    def test_get_embedding_function_success(self, mock_embeddings):
        """Test successful embedding function initialization."""
        mock_embedding = MagicMock()
        mock_embeddings.return_value = mock_embedding

        embedding_func = database.get_embedding_function()

        assert embedding_func == mock_embedding
        mock_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")

    @patch("src.core.database.HuggingFaceEmbeddings")
    def test_get_embedding_function_singleton(self, mock_embeddings):
        """Test embedding function singleton behavior."""
        mock_embedding = MagicMock()
        mock_embeddings.return_value = mock_embedding

        embedding1 = database.get_embedding_function()
        embedding2 = database.get_embedding_function()

        assert embedding1 == embedding2
        # Should only initialize once
        mock_embeddings.assert_called_once()

    @patch("src.core.database.CrossEncoder")
    def test_get_cross_encoder_success(self, mock_cross_encoder):
        """Test successful cross-encoder initialization."""
        mock_encoder = MagicMock()
        mock_cross_encoder.return_value = mock_encoder

        encoder = database.get_cross_encoder()

        assert encoder == mock_encoder
        mock_cross_encoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    @patch("src.core.database.CrossEncoder")
    def test_get_cross_encoder_fallback(self, mock_cross_encoder):
        """Test CrossEncoder failure returns None for graceful fallback."""
        mock_cross_encoder.side_effect = Exception("Model loading failed")

        encoder = database.get_cross_encoder()

        assert encoder is None

    @patch("src.core.database.CrossEncoder")
    def test_get_cross_encoder_singleton(self, mock_cross_encoder):
        """Test cross-encoder singleton behavior."""
        mock_encoder = MagicMock()
        mock_cross_encoder.return_value = mock_encoder

        encoder1 = database.get_cross_encoder()
        encoder2 = database.get_cross_encoder()

        assert encoder1 == encoder2
        # Should only initialize once
        mock_cross_encoder.assert_called_once()

    @patch("src.core.database.Chroma")
    @patch("src.core.database.get_embedding_function")
    @patch("src.core.database.get_chroma_client")
    def test_get_vectorstore_success(
        self,
        mock_get_client,
        mock_get_embeddings,
        mock_chroma
    ):
        """Test successful vector store initialization."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_vectorstore = MagicMock()

        mock_get_client.return_value = mock_client
        mock_get_embeddings.return_value = mock_embeddings
        mock_chroma.return_value = mock_vectorstore

        vectorstore = database.get_vectorstore()

        assert vectorstore == mock_vectorstore
        mock_chroma.assert_called_once_with(
            client=mock_client,
            collection_name="baseline_kb",  # Updated to match current config
            embedding_function=mock_embeddings
        )

    @patch("src.core.database.Chroma")
    @patch("src.core.database.get_embedding_function")
    @patch("src.core.database.get_chroma_client")
    def test_get_vectorstore_singleton(
        self,
        mock_get_client,
        mock_get_embeddings,
        mock_chroma
    ):
        """Test vectorstore singleton behavior."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_vectorstore = MagicMock()

        mock_get_client.return_value = mock_client
        mock_get_embeddings.return_value = mock_embeddings
        mock_chroma.return_value = mock_vectorstore

        vs1 = database.get_vectorstore()
        vs2 = database.get_vectorstore()

        assert vs1 == vs2
        # Should only initialize once
        mock_chroma.assert_called_once()

    def test_client_store_isolation(self):
        """Test that different resource types are isolated in client store."""
        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value = MagicMock()
            database.get_chroma_client()

        with patch("src.core.database.HuggingFaceEmbeddings") as mock_emb:
            mock_emb.return_value = MagicMock()
            database.get_embedding_function()

        # Different keys in the store
        assert "client" in database._clients
        assert "embedding_function" in database._clients
        assert database._clients["client"] != database._clients["embedding_function"]

    @patch("src.core.database.HuggingFaceEmbeddings")
    def test_embedding_function_configuration(self, mock_embeddings):
        """Test that embedding function uses correct model configuration."""
        from src.core.config import EMBEDDING_MODEL

        mock_embeddings.return_value = MagicMock()
        database.get_embedding_function()

        # Verify correct model name from config
        mock_embeddings.assert_called_once_with(model_name=EMBEDDING_MODEL)

    @patch("src.core.database.CrossEncoder")
    def test_cross_encoder_configuration(self, mock_cross_encoder):
        """Test that cross-encoder uses correct model configuration."""
        from src.core.config import CROSS_ENCODER_MODEL

        mock_cross_encoder.return_value = MagicMock()
        database.get_cross_encoder()

        # Verify correct model name from config
        mock_cross_encoder.assert_called_once_with(CROSS_ENCODER_MODEL)
