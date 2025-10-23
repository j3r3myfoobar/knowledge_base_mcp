"""
Shared pytest fixtures for all tests.

These fixtures provide standard test data and mocked components that can be
reused across all test files. This follows the DRY principle and makes tests
more maintainable.
"""
import pytest
from unittest.mock import Mock
from langchain.docstore.document import Document


# --- Document Fixtures ---

@pytest.fixture
def sample_documents():
    """
    Standard test documents representing a realistic knowledge base.

    Use this for tests that need a small set of varied documents with
    different types and content.
    """
    return [
        Document(
            page_content="Python is a programming language used for data science and web development.",
            metadata={'source': 'python.md', 'type': 'technical_doc'}
        ),
        Document(
            page_content="AWS Aurora is a cloud database service that provides high availability.",
            metadata={'source': 'aws.pdf', 'type': 'technical_doc'}
        ),
        Document(
            page_content="Terraform manages infrastructure as code using declarative configuration.",
            metadata={'source': 'terraform.md', 'type': 'personal_note'}
        ),
    ]


@pytest.fixture
def long_document():
    """
    A long document that will be split into multiple chunks.

    Use this for testing chunking behavior - should produce 2+ chunks
    with default chunk size of 512 characters.
    """
    return Document(
        page_content="This is a test document with repeated content. " * 50,  # ~2350 chars
        metadata={'source': 'long_doc.md', 'type': 'technical_doc'}
    )


@pytest.fixture
def markdown_document():
    """Simple markdown document for type inference tests."""
    return Document(
        page_content="# Test Document\n\nThis is a markdown document.",
        metadata={'source': 'test.md'}
    )


@pytest.fixture
def pdf_document():
    """Simple PDF document for type inference tests."""
    return Document(
        page_content="This is a PDF document with technical content.",
        metadata={'source': 'test.pdf'}
    )


# --- Mock Component Fixtures ---

@pytest.fixture
def mock_vectorstore():
    """Mocked Chroma vectorstore with common behaviors."""
    mock = Mock()
    mock.add_documents = Mock(return_value=None)
    mock.similarity_search_with_score = Mock(return_value=[])
    mock._collection = Mock()
    mock._collection.name = "test_collection"
    return mock


@pytest.fixture
def mock_embeddings():
    """Mocked HuggingFace embeddings."""
    mock = Mock()
    mock.embed_query = Mock(return_value=[0.1] * 384)
    mock.embed_documents = Mock(return_value=[[0.1] * 384])
    return mock


@pytest.fixture
def mock_chroma_client():
    """Mocked ChromaDB HTTP client."""
    mock = Mock()
    mock.delete_collection = Mock(return_value=None)
    mock.get_collection = Mock()
    return mock


@pytest.fixture
def mock_chunker():
    """Mocked DocumentChunker that returns input unchanged."""
    mock = Mock()
    # By default, return documents unchanged (no chunking)
    mock.chunk_documents = Mock(side_effect=lambda docs: docs)
    return mock


@pytest.fixture
def mock_bm25_index():
    """Mocked BM25 search index."""
    mock = Mock()
    mock.build_index = Mock(return_value=None)
    mock.search = Mock(return_value=[])
    mock.is_built = Mock(return_value=False)
    mock.get_index_size = Mock(return_value=0)
    return mock


@pytest.fixture
def mock_retriever_dependencies(
    mock_vectorstore,
    mock_embeddings,
    mock_chroma_client,
    mock_chunker,
    mock_bm25_index
):
    """
    Complete set of mocked dependencies for BaselineRetriever.

    Use this when you want to test the retriever's orchestration logic
    without actually running real components.

    Example:
        def test_something(mock_retriever_dependencies):
            retriever = BaselineRetriever(**mock_retriever_dependencies)
            # Test retriever behavior...
    """
    return {
        'vectorstore': mock_vectorstore,
        'embeddings': mock_embeddings,
        'chroma_client': mock_chroma_client,
        'chunker': mock_chunker,
        'bm25_index': mock_bm25_index,
    }


# --- Search Result Fixtures ---

@pytest.fixture
def mock_search_results():
    """Sample search results for testing query methods."""
    return [
        {
            'content': 'Python is a programming language.',
            'metadata': {'source': 'python.md', 'type': 'technical_doc'},
            'confidence': 0.95
        },
        {
            'content': 'AWS Aurora is a database service.',
            'metadata': {'source': 'aws.pdf', 'type': 'technical_doc'},
            'confidence': 0.87
        },
    ]


@pytest.fixture
def mock_vector_search_results(sample_documents):
    """
    Mock vector search results in ChromaDB format: List[(Document, distance)].

    Lower distance = better match (0.0 = perfect match).
    """
    return [
        (sample_documents[0], 0.15),  # Good match
        (sample_documents[1], 0.42),  # Medium match
        (sample_documents[2], 0.68),  # Weak match
    ]


@pytest.fixture
def mock_bm25_search_results(sample_documents):
    """
    Mock BM25 search results: List[(Document, score)].

    Higher score = better match.
    """
    return [
        (sample_documents[0], 8.5),  # Best match
        (sample_documents[1], 4.2),  # Medium match
        (sample_documents[2], 1.1),  # Weak match
    ]
