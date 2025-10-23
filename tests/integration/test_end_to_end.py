import pytest
from unittest.mock import MagicMock, patch
from src.server.mcp_app import query_knowledge_base
from src.core.models import Document, KnowledgeBaseOutput
import langchain.docstore.document


class TestQueryKnowledgeBaseEnhancements:
    """Integration tests for the enhanced query_knowledge_base functionality."""

    @pytest.mark.asyncio
    @patch("src.server.mcp_app.get_vectorstore")
    @patch("src.server.mcp_app.get_cross_encoder")
    async def test_query_enhancement_integration(self, mock_get_cross_encoder, mock_get_vectorstore):
        """Test that query enhancement is applied during retrieval."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            langchain.docstore.document.Document(page_content="Python classes tutorial", metadata={"source": "python.py"}),
        ]
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.8]

        mock_get_vectorstore.return_value = mock_vectorstore
        mock_get_cross_encoder.return_value = mock_cross_encoder

        # Test with a statement that should be enhanced to a question
        result = await query_knowledge_base.run({"query": "Python classes"})

        # Verify that the enhanced query was used for retrieval
        mock_retriever.invoke.assert_called_once()
        called_query = mock_retriever.invoke.call_args[0][0]
        assert called_query == "what is Python classes?"

        # Verify that enhanced query was also used for cross-encoder
        mock_cross_encoder.predict.assert_called_once()
        cross_encoder_pairs = mock_cross_encoder.predict.call_args[0][0]
        assert cross_encoder_pairs[0][0] == "what is Python classes?"

    @pytest.mark.asyncio
    @patch("src.server.mcp_app.get_vectorstore")
    @patch("src.server.mcp_app.get_cross_encoder")
    async def test_confidence_scoring_integration(self, mock_get_cross_encoder, mock_get_vectorstore):
        """Test that confidence scoring is properly integrated."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            langchain.docstore.document.Document(
                page_content="Python programming language tutorial",
                metadata={"source": "tutorial.py"}
            ),
        ]
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9]

        mock_get_vectorstore.return_value = mock_vectorstore
        mock_get_cross_encoder.return_value = mock_cross_encoder

        result = await query_knowledge_base.run({"query": "Python programming"})

        import json
        output = json.loads(result.content[0].text)
        output = KnowledgeBaseOutput(**output)

        # Check confidence score properties
        doc = output.documents[0]
        assert hasattr(doc, 'confidence')
        assert 0.0 <= doc.confidence <= 1.0

        # Should have high confidence due to:
        # - High cross-encoder score (0.9)
        # - Good query overlap ("Python programming" in content)
        # - Metadata present
        assert doc.confidence > 0.5  # Should be reasonably high

    @pytest.mark.asyncio
    @patch("src.server.mcp_app.get_vectorstore")
    @patch("src.server.mcp_app.get_cross_encoder")
    async def test_multiple_documents_confidence_ordering(self, mock_get_cross_encoder, mock_get_vectorstore):
        """Test that documents are returned with proper confidence scores."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            langchain.docstore.document.Document(
                page_content="Python programming advanced tutorial",
                metadata={"source": "advanced.py"}
            ),
            langchain.docstore.document.Document(
                page_content="JavaScript frameworks overview",
                metadata={}  # No source metadata
            ),
        ]
        mock_cross_encoder = MagicMock()
        # Higher score for first doc, lower for second
        mock_cross_encoder.predict.return_value = [0.9, 0.3]

        mock_get_vectorstore.return_value = mock_vectorstore
        mock_get_cross_encoder.return_value = mock_cross_encoder

        result = await query_knowledge_base.run({"query": "Python programming"})

        import json
        output = json.loads(result.content[0].text)
        output = KnowledgeBaseOutput(**output)

        assert len(output.documents) == 2

        # First document should have higher confidence
        # (better cross-encoder score, query overlap, has metadata)
        doc1 = output.documents[0]
        doc2 = output.documents[1]

        assert doc1.confidence > doc2.confidence
        assert doc1.content == "Python programming advanced tutorial"
        assert doc2.content == "JavaScript frameworks overview"

    @pytest.mark.asyncio
    @patch("src.server.mcp_app.get_vectorstore")
    @patch("src.server.mcp_app.get_cross_encoder")
    async def test_fallback_path_with_enhancements(self, mock_get_cross_encoder, mock_get_vectorstore):
        """Test that enhancements work in fallback mode (no cross-encoder)."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            langchain.docstore.document.Document(
                page_content="Python classes and objects tutorial",
                metadata={"source": "oop.py"}
            ),
        ]

        mock_get_vectorstore.return_value = mock_vectorstore
        mock_get_cross_encoder.return_value = None  # No cross-encoder

        result = await query_knowledge_base.run({"query": "Python classes"})

        # Verify enhanced query was used even in fallback mode
        mock_retriever.invoke.assert_called_once()
        called_query = mock_retriever.invoke.call_args[0][0]
        assert called_query == "what is Python classes?"

        import json
        output = json.loads(result.content[0].text)
        output = KnowledgeBaseOutput(**output)

        # Should still have confidence score (without cross-encoder component)
        doc = output.documents[0]
        assert hasattr(doc, 'confidence')
        assert 0.0 <= doc.confidence <= 1.0

    @pytest.mark.asyncio
    @patch("src.server.mcp_app.get_vectorstore")
    @patch("src.server.mcp_app.get_cross_encoder")
    async def test_existing_question_not_modified(self, mock_get_cross_encoder, mock_get_vectorstore):
        """Test that queries already in question format are not modified."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            langchain.docstore.document.Document(page_content="Python tutorial content", metadata={}),
        ]
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.7]

        mock_get_vectorstore.return_value = mock_vectorstore
        mock_get_cross_encoder.return_value = mock_cross_encoder

        # Query already in question format
        original_question = "What are Python classes?"
        result = await query_knowledge_base.run({"query": original_question})

        # Should not be modified
        mock_retriever.invoke.assert_called_once()
        called_query = mock_retriever.invoke.call_args[0][0]
        assert called_query == original_question

    @pytest.mark.asyncio
    @patch("src.server.mcp_app.get_vectorstore")
    @patch("src.server.mcp_app.get_cross_encoder")
    async def test_confidence_score_components(self, mock_get_cross_encoder, mock_get_vectorstore):
        """Test that confidence score reflects different quality signals."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever

        # Document with perfect query match and metadata
        perfect_doc = langchain.docstore.document.Document(
            page_content="Python programming tutorial",
            metadata={"source": "python_tutorial.py"}
        )

        # Document with poor match and no metadata
        poor_doc = langchain.docstore.document.Document(
            page_content="JavaScript advanced concepts and frameworks",
            metadata={}
        )

        mock_retriever.invoke.return_value = [perfect_doc, poor_doc]
        mock_vectorstore.as_retriever.return_value = mock_retriever

        mock_cross_encoder = MagicMock()
        # Same cross-encoder scores to isolate other factors
        mock_cross_encoder.predict.return_value = [0.5, 0.5]

        mock_get_vectorstore.return_value = mock_vectorstore
        mock_get_cross_encoder.return_value = mock_cross_encoder

        result = await query_knowledge_base.run({"query": "Python programming"})

        import json
        output = json.loads(result.content[0].text)
        output = KnowledgeBaseOutput(**output)

        assert len(output.documents) == 2

        # Perfect match document should have higher confidence
        # due to better query overlap and metadata presence
        doc1 = output.documents[0]  # Perfect doc (re-ranked first)
        doc2 = output.documents[1]  # Poor doc

        # Find which document is which based on content
        if "Python" in doc1.content:
            python_doc = doc1
            js_doc = doc2
        else:
            python_doc = doc2
            js_doc = doc1

        # Python doc should have higher confidence due to:
        # - Better query word overlap
        # - Has source metadata
        assert python_doc.confidence > js_doc.confidence