"""
Unit tests for Pydantic models.

Tests the API data models used by the MCP server for input/output validation.
"""

import pytest
from pydantic import ValidationError

from src.core.models import Document, KnowledgeBaseOutput


class TestDocumentModel:
    """Test cases for the Document model."""

    def test_document_with_all_fields(self):
        """Test Document creation with all fields."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.py", "page": 1},
            confidence=0.85
        )

        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test.py", "page": 1}
        assert doc.confidence == 0.85

    def test_document_default_confidence(self):
        """Test Document creation with default confidence."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.py"}
        )

        assert doc.confidence == 0.0

    def test_document_minimum_fields(self):
        """Test Document with only required fields."""
        doc = Document(
            content="Test content",
            metadata={}
        )

        assert doc.content == "Test content"
        assert doc.metadata == {}
        assert doc.confidence == 0.0

    def test_document_confidence_validation_valid(self):
        """Test that valid confidence values are accepted."""
        # Boundary values
        doc1 = Document(content="test", metadata={}, confidence=0.0)
        doc2 = Document(content="test", metadata={}, confidence=1.0)
        doc3 = Document(content="test", metadata={}, confidence=0.5)

        assert doc1.confidence == 0.0
        assert doc2.confidence == 1.0
        assert doc3.confidence == 0.5

    def test_document_confidence_validation_too_low(self):
        """Test that confidence below 0.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Document(content="test", metadata={}, confidence=-0.1)

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_document_confidence_validation_too_high(self):
        """Test that confidence above 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Document(content="test", metadata={}, confidence=1.1)

        assert "less than or equal to 1" in str(exc_info.value)

    def test_document_missing_required_field_content(self):
        """Test that missing 'content' field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Document(metadata={})

        assert "content" in str(exc_info.value)

    def test_document_missing_required_field_metadata(self):
        """Test that missing 'metadata' field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Document(content="Test content")

        assert "metadata" in str(exc_info.value)

    def test_document_empty_content(self):
        """Test that empty content is allowed."""
        doc = Document(content="", metadata={})
        assert doc.content == ""

    def test_document_complex_metadata(self):
        """Test Document with complex nested metadata."""
        metadata = {
            "source": "test.pdf",
            "page": 42,
            "chapter": "Introduction",
            "tags": ["python", "tutorial"],
            "nested": {"key": "value"}
        }
        doc = Document(content="Test", metadata=metadata)

        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["page"] == 42
        assert doc.metadata["tags"] == ["python", "tutorial"]
        assert doc.metadata["nested"]["key"] == "value"

    def test_document_serialization(self):
        """Test that Document can be serialized to dict."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.py"},
            confidence=0.75
        )

        doc_dict = doc.model_dump()

        assert doc_dict["content"] == "Test content"
        assert doc_dict["metadata"] == {"source": "test.py"}
        assert doc_dict["confidence"] == 0.75

    def test_document_json_serialization(self):
        """Test that Document can be serialized to JSON."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.py"},
            confidence=0.75
        )

        json_str = doc.model_dump_json()

        assert "Test content" in json_str
        assert "test.py" in json_str
        assert "0.75" in json_str


class TestKnowledgeBaseOutput:
    """Test cases for the KnowledgeBaseOutput model."""

    def test_knowledge_base_output_single_document(self):
        """Test KnowledgeBaseOutput with single document."""
        doc = Document(
            content="Test",
            metadata={"source": "test.txt"},
            confidence=0.8
        )
        output = KnowledgeBaseOutput(documents=[doc])

        assert len(output.documents) == 1
        assert output.documents[0].content == "Test"

    def test_knowledge_base_output_multiple_documents(self):
        """Test KnowledgeBaseOutput with multiple documents."""
        docs = [
            Document(content="First", metadata={}, confidence=0.9),
            Document(content="Second", metadata={}, confidence=0.7),
            Document(content="Third", metadata={}, confidence=0.5),
        ]
        output = KnowledgeBaseOutput(documents=docs)

        assert len(output.documents) == 3
        assert output.documents[0].content == "First"
        assert output.documents[1].content == "Second"
        assert output.documents[2].content == "Third"

    def test_knowledge_base_output_empty_list(self):
        """Test KnowledgeBaseOutput with empty document list."""
        output = KnowledgeBaseOutput(documents=[])

        assert len(output.documents) == 0

    def test_knowledge_base_output_validation(self):
        """Test that invalid documents are rejected."""
        with pytest.raises(ValidationError):
            # Dict missing required 'metadata' field - Pydantic validation should fail
            KnowledgeBaseOutput(documents=[
                {"content": "Test"}  # Invalid document structure: missing required 'metadata' field
            ])

    def test_knowledge_base_output_serialization(self):
        """Test KnowledgeBaseOutput serialization."""
        docs = [
            Document(content="Test1", metadata={"source": "a.txt"}, confidence=0.9),
            Document(content="Test2", metadata={"source": "b.txt"}, confidence=0.7),
        ]
        output = KnowledgeBaseOutput(documents=docs)

        output_dict = output.model_dump()

        assert len(output_dict["documents"]) == 2
        assert output_dict["documents"][0]["content"] == "Test1"
        assert output_dict["documents"][1]["content"] == "Test2"

    def test_knowledge_base_output_json_serialization(self):
        """Test KnowledgeBaseOutput JSON serialization."""
        docs = [
            Document(content="Test", metadata={"source": "test.txt"}, confidence=0.8)
        ]
        output = KnowledgeBaseOutput(documents=docs)

        json_str = output.model_dump_json()

        assert "Test" in json_str
        assert "test.txt" in json_str
        assert "0.8" in json_str

    def test_knowledge_base_output_preserves_order(self):
        """Test that document order is preserved."""
        docs = [
            Document(content=f"Doc{i}", metadata={}, confidence=i/10)
            for i in range(5)
        ]
        output = KnowledgeBaseOutput(documents=docs)

        for i, doc in enumerate(output.documents):
            assert doc.content == f"Doc{i}"
            assert doc.confidence == i/10
