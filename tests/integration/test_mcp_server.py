import pytest
from unittest.mock import MagicMock, patch
from src.server.mcp_app import query_knowledge_base
from src.core.models import Document, KnowledgeBaseOutput
import langchain.docstore.document


class TestQueryKnowledgeBaseIntegration:
    """Integration tests for the enhanced query_knowledge_base tool."""


@patch("src.server.mcp_app.get_vectorstore")
@patch("src.server.mcp_app.get_cross_encoder")
async def test_query_knowledge_base_with_reranking(mock_get_cross_encoder, mock_get_vectorstore):
    """Test the query_knowledge_base tool with re-ranking."""
    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        langchain.docstore.document.Document(page_content="This is a test document.", metadata={}),
    ]
    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.return_value = [0.9]

    mock_get_vectorstore.return_value = mock_vectorstore
    mock_get_cross_encoder.return_value = mock_cross_encoder

    result = await query_knowledge_base.run({"query": "test query"})

    import json
    output = json.loads(result.content[0].text)
    output = KnowledgeBaseOutput(**output)

    assert isinstance(output, KnowledgeBaseOutput)
    assert len(output.documents) == 1
    assert output.documents[0].content == "This is a test document."
    # Check that confidence score was added
    assert hasattr(output.documents[0], 'confidence')
    assert 0.0 <= output.documents[0].confidence <= 1.0
    mock_vectorstore.as_retriever.assert_called_once()
    mock_cross_encoder.predict.assert_called_once()


@pytest.mark.asyncio
@patch("src.server.mcp_app.get_vectorstore")
@patch("src.server.mcp_app.get_cross_encoder")
async def test_query_knowledge_base_without_reranking(mock_get_cross_encoder, mock_get_vectorstore):
    """Test the query_knowledge_base tool without re-ranking (when cross-encoder is not available)."""
    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [
        langchain.docstore.document.Document(page_content="This is a test document.", metadata={}),
    ]

    mock_get_vectorstore.return_value = mock_vectorstore
    mock_get_cross_encoder.return_value = None

    result = await query_knowledge_base.run({"query": "test query"})

    import json
    output = json.loads(result.content[0].text)
    output = KnowledgeBaseOutput(**output)

    assert isinstance(output, KnowledgeBaseOutput)
    assert len(output.documents) == 1
    assert output.documents[0].content == "This is a test document."
    # Check that confidence score was added (even without cross-encoder)
    assert hasattr(output.documents[0], 'confidence')
    assert 0.0 <= output.documents[0].confidence <= 1.0
    mock_vectorstore.as_retriever.assert_called_once()
