"""
Test helper functions for creating test objects.

These helpers make it easy to create properly configured objects for testing
without repeating setup code.
"""
from unittest.mock import Mock
from src.retrieval import BaselineRetriever
from src.components.chunker import FixedSizeChunker
from src.components.bm25_index import BM25Index


def create_retriever_with_real_components(**overrides):
    """
    Create BaselineRetriever with real chunker and BM25 index for integration testing.

    This is useful when you want to test the retriever's actual behavior with
    real components, but don't want to deal with ChromaDB or embeddings.

    By default:
    - Chunker: Real FixedSizeChunker (actually splits documents)
    - BM25 Index: Real BM25Index (actually searches)
    - Vectorstore: Mocked (no real ChromaDB needed)
    - Embeddings: Mocked (no real model needed)
    - ChromaDB Client: Mocked (no real DB needed)

    Args:
        **overrides: Override any component by passing it as a keyword argument.
                     E.g., chunker=MyCustomChunker()

    Returns:
        BaselineRetriever: Configured retriever ready for testing

    """
    defaults = {
        'vectorstore': Mock(),
        'embeddings': Mock(),
        'chroma_client': Mock(),
        'chunker': FixedSizeChunker(),
        'bm25_index': BM25Index(),
    }
    defaults.update(overrides)
    return BaselineRetriever(**defaults)


def create_fully_mocked_retriever(**overrides):
    """
    Create BaselineRetriever with all dependencies mocked.

    This is useful for pure unit tests where you only want to test the
    retriever's orchestration logic, not the actual component implementations.

    All components return empty/default values unless you configure them.

    Args:
        **overrides: Override any mock by passing it as a keyword argument

    Returns:
        BaselineRetriever: Retriever with all mocked dependencies

    """
    defaults = {
        'vectorstore': Mock(),
        'embeddings': Mock(),
        'chroma_client': Mock(),
        'chunker': Mock(),
        'bm25_index': Mock(),
    }
    defaults.update(overrides)
    return BaselineRetriever(**defaults)


def create_mock_chroma_collection_with_docs(documents, collection_name="test_collection"):
    """
    Create a mocked ChromaDB collection that returns specific documents.

    This is useful for testing the initialize() method which fetches
    existing documents from ChromaDB.

    Args:
        documents: List of dicts with 'content' and 'metadata' keys
        collection_name: Name of the collection

    Returns:
        tuple: (mock_chroma_client, mock_collection)

    Example (TDD):
        # Test: "Initialize should build BM25 index from existing docs"
        def test_initialize_builds_index():
            # Arrange
            existing_docs = [
                {'content': 'doc1', 'metadata': {'source': 'test.md'}},
                {'content': 'doc2', 'metadata': {'source': 'test2.md'}},
            ]
            mock_client, mock_collection = create_mock_chroma_collection_with_docs(existing_docs)

            # Act
            retriever = create_fully_mocked_retriever(chroma_client=mock_client)
            retriever.initialize()

            # Assert
            assert mock_collection.get.called
    """
    mock_collection = Mock()
    mock_collection.get.return_value = {
        'documents': [doc['content'] for doc in documents],
        'metadatas': [doc['metadata'] for doc in documents]
    }

    mock_chroma_client = Mock()
    mock_chroma_client.get_collection.return_value = mock_collection

    return mock_chroma_client, mock_collection


def assert_documents_equal(doc1, doc2, check_metadata=True):
    """
    Assert that two LangChain Documents are equal.

    Useful for comparing expected vs actual documents in tests.

    Args:
        doc1: First Document
        doc2: Second Document
        check_metadata: Whether to compare metadata (default True)

    Raises:
        AssertionError: If documents are not equal

    Example (TDD):
        # Test: "Chunker should preserve document content"
        def test_preserves_content():
            # Arrange
            original = Document(page_content="test", metadata={})

            # Act
            chunks = chunker.chunk_documents([original])

            # Assert
            assert_documents_equal(chunks[0], original, check_metadata=False)
    """
    assert doc1.page_content == doc2.page_content, \
        f"Content mismatch:\nExpected: {doc2.page_content}\nActual: {doc1.page_content}"

    if check_metadata:
        assert doc1.metadata == doc2.metadata, \
            f"Metadata mismatch:\nExpected: {doc2.metadata}\nActual: {doc1.metadata}"


def create_sample_chunks(num_chunks=3, source="test.md"):
    """
    Create sample document chunks for testing.

    Args:
        num_chunks: Number of chunks to create
        source: Source filename for metadata

    Returns:
        List of Documents with proper chunk metadata

    Example (TDD):
        # Test: "BM25 index should accept chunks"
        def test_builds_index_with_chunks():
            # Arrange
            chunks = create_sample_chunks(5)
            index = BM25Index()

            # Act
            index.build_index(chunks)

            # Assert
            assert index.get_index_size() == 5
    """
    from langchain.docstore.document import Document

    chunks = []
    for i in range(num_chunks):
        chunk = Document(
            page_content=f"This is chunk {i} of the document with some test content.",
            metadata={
                'source': source,
                'chunk_id': i,
                'global_chunk_id': i,
                'document_id': 0,
                'type': 'technical_doc' if source.endswith('.pdf') else 'personal_note'
            }
        )
        chunks.append(chunk)

    return chunks
