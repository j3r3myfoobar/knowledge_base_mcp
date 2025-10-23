"""
Unit tests for scripts/ingest.py

Tests all methods of the DocumentIngester class.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from langchain.docstore.document import Document

from scripts.ingest import DocumentIngester


@pytest.fixture
def temp_docs_dir(tmp_path):
    """Create a temporary documents directory with test files."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    # Create test markdown files
    (docs_dir / "test1.md").write_text("# Test Document 1\n\nThis is a test.")
    (docs_dir / "test2.md").write_text("# Test Document 2\n\nAnother test.")

    # Create a README that should be filtered
    (tmp_path / "README.md").write_text("# Project README")

    return docs_dir


@pytest.fixture
def ingester(temp_docs_dir):
    """Create a DocumentIngester instance for testing."""
    return DocumentIngester(
        docs_dir=str(temp_docs_dir),
        collection_name="test_collection"
    )


class TestDocumentIngesterInit:
    """Test DocumentIngester initialization."""

    def test_init_with_defaults(self, temp_docs_dir):
        """Test initialization with default collection name."""
        ingester = DocumentIngester(docs_dir=str(temp_docs_dir))

        assert ingester.docs_dir == Path(temp_docs_dir)
        assert ingester.collection_name == "baseline_kb"
        assert ingester.retriever is None

    def test_init_with_custom_collection(self, temp_docs_dir):
        """Test initialization with custom collection name."""
        ingester = DocumentIngester(
            docs_dir=str(temp_docs_dir),
            collection_name="custom_collection"
        )

        assert ingester.collection_name == "custom_collection"

    def test_init_converts_path(self, temp_docs_dir):
        """Test that string path is converted to Path object."""
        ingester = DocumentIngester(docs_dir=str(temp_docs_dir))

        assert isinstance(ingester.docs_dir, Path)


class TestHandleReingestion:
    """Test handle_reingestion method."""

    @patch('scripts.ingest.get_chroma_client')
    def test_handle_reingestion_success(self, mock_get_client, ingester):
        """Test successful collection deletion."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        ingester.handle_reingestion()

        mock_client.delete_collection.assert_called_once_with("test_collection")

    @patch('scripts.ingest.get_chroma_client')
    def test_handle_reingestion_collection_not_exists(self, mock_get_client, ingester):
        """Test deletion when collection doesn't exist (should warn, not fail)."""
        mock_client = Mock()
        mock_client.delete_collection.side_effect = Exception("Collection not found")
        mock_get_client.return_value = mock_client

        # Should not raise, just log warning
        ingester.handle_reingestion()

        mock_client.delete_collection.assert_called_once()


class TestInitializeRetriever:
    """Test initialize_retriever method."""

    @patch('scripts.ingest.create_baseline_retriever')
    def test_initialize_retriever_default_params(self, mock_factory, ingester):
        """Test retriever initialization with default parameters."""
        mock_retriever = Mock()
        mock_factory.return_value = mock_retriever

        ingester.initialize_retriever()

        mock_factory.assert_called_once_with(
            chroma_host="chroma",
            chroma_port=8000,
            collection_name="test_collection"
        )
        assert ingester.retriever == mock_retriever

    @patch('scripts.ingest.create_baseline_retriever')
    def test_initialize_retriever_custom_params(self, mock_factory, ingester):
        """Test retriever initialization with custom parameters."""
        mock_retriever = Mock()
        mock_factory.return_value = mock_retriever

        ingester.initialize_retriever(chroma_host="localhost", chroma_port=9000)

        mock_factory.assert_called_once_with(
            chroma_host="localhost",
            chroma_port=9000,
            collection_name="test_collection"
        )


class TestFindDocuments:
    """Test find_documents method."""

    def test_find_documents_success(self, ingester):
        """Test finding markdown and PDF files."""
        md_files, pdf_files = ingester.find_documents()

        assert len(md_files) == 2
        assert all(f.suffix == ".md" for f in md_files)
        assert len(pdf_files) == 0

    def test_find_documents_filters_readme(self, tmp_path):
        """Test that README.md in root is filtered out."""
        # Create README in root
        (tmp_path / "README.md").write_text("# README")

        # Create documents dir with another README
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text("# Docs README")

        ingester = DocumentIngester(docs_dir=str(docs_dir))
        md_files, pdf_files = ingester.find_documents()

        # Should include documents/README.md but not root README.md
        assert len(md_files) == 1
        assert "documents" in str(md_files[0])

    def test_find_documents_nonexistent_dir(self, tmp_path):
        """Test error when documents directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        ingester = DocumentIngester(docs_dir=str(nonexistent))

        with pytest.raises(FileNotFoundError, match="Directory not found"):
            ingester.find_documents()

    def test_find_documents_with_pdfs(self, tmp_path):
        """Test finding PDF files."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create test files
        (docs_dir / "test.md").write_text("# Test")
        (docs_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake pdf")

        ingester = DocumentIngester(docs_dir=str(docs_dir))
        md_files, pdf_files = ingester.find_documents()

        assert len(md_files) == 1
        assert len(pdf_files) == 1
        assert pdf_files[0].suffix == ".pdf"


class TestLoadDocuments:
    """Test load_documents method."""

    def test_load_markdown_documents(self, ingester, temp_docs_dir):
        """Test loading markdown files."""
        md_files = list(temp_docs_dir.glob("*.md"))
        pdf_files = []

        documents = ingester.load_documents(md_files, pdf_files)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.metadata['type'] == 'personal_note' for doc in documents)
        assert all('filename' in doc.metadata for doc in documents)
        assert all('source' in doc.metadata for doc in documents)

    @patch('scripts.ingest.PyPDFLoader')
    def test_load_pdf_documents(self, mock_pdf_loader, ingester, tmp_path):
        """Test loading PDF files."""
        # Create fake PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        # Mock PDF loader
        mock_loader_instance = Mock()
        mock_doc = Document(page_content="PDF content", metadata={})
        mock_loader_instance.load.return_value = [mock_doc]
        mock_pdf_loader.return_value = mock_loader_instance

        documents = ingester.load_documents([], [pdf_file])

        assert len(documents) == 1
        assert documents[0].metadata['type'] == 'technical_doc'
        assert documents[0].metadata['filename'] == 'test.pdf'

    def test_load_documents_handles_errors(self, ingester, tmp_path):
        """Test that loading errors don't crash, just log."""
        # Create file with encoding issues
        bad_file = tmp_path / "bad.md"
        bad_file.write_bytes(b'\xff\xfe invalid utf-8')

        # Should not raise, just log error
        documents = ingester.load_documents([bad_file], [])

        # May be empty or have partial results depending on loader behavior
        assert isinstance(documents, list)


class TestChunkDocuments:
    """Test chunk_documents method."""

    def test_chunk_documents_success(self, ingester):
        """Test chunking documents."""
        mock_retriever = Mock()
        mock_chunks = [
            Document(page_content="chunk1", metadata={'chunk_id': 0}),
            Document(page_content="chunk2", metadata={'chunk_id': 1}),
        ]
        mock_retriever.chunk_documents.return_value = mock_chunks
        ingester.retriever = mock_retriever

        documents = [Document(page_content="test content", metadata={})]
        chunks = ingester.chunk_documents(documents)

        assert chunks == mock_chunks
        mock_retriever.chunk_documents.assert_called_once_with(documents)

    def test_chunk_documents_no_retriever(self, ingester):
        """Test error when retriever not initialized."""
        documents = [Document(page_content="test", metadata={})]

        with pytest.raises(ValueError, match="Retriever not initialized"):
            ingester.chunk_documents(documents)


class TestBuildBM25Index:
    """Test build_bm25_index method."""

    def test_build_bm25_index_success(self, ingester):
        """Test building BM25 index."""
        mock_retriever = Mock()
        ingester.retriever = mock_retriever

        chunks = [
            Document(page_content="chunk1", metadata={}),
            Document(page_content="chunk2", metadata={}),
        ]

        ingester.build_bm25_index(chunks)

        mock_retriever.build_bm25_index.assert_called_once_with(chunks)

    def test_build_bm25_index_no_retriever(self, ingester):
        """Test error when retriever not initialized."""
        chunks = [Document(page_content="chunk", metadata={})]

        with pytest.raises(ValueError, match="Retriever not initialized"):
            ingester.build_bm25_index(chunks)


class TestAddToVectorstore:
    """Test add_to_vectorstore method."""

    def test_add_to_vectorstore_success(self, ingester):
        """Test adding chunks to vectorstore via Retriever interface."""
        mock_retriever = Mock()
        mock_retriever.add_documents = Mock()  # Use Retriever interface method
        ingester.retriever = mock_retriever

        chunks = [Document(page_content="chunk", metadata={})]

        ingester.add_to_vectorstore(chunks)

        # Verify the Retriever interface method was called
        mock_retriever.add_documents.assert_called_once_with(chunks)

    def test_add_to_vectorstore_no_retriever(self, ingester):
        """Test error when retriever not initialized."""
        chunks = [Document(page_content="chunk", metadata={})]

        with pytest.raises(ValueError, match="Retriever not initialized"):
            ingester.add_to_vectorstore(chunks)

    def test_add_to_vectorstore_failure(self, ingester):
        """Test handling of retriever add_documents failures."""
        mock_retriever = Mock()
        # Simulate RuntimeError from retriever.add_documents()
        mock_retriever.add_documents.side_effect = RuntimeError("Failed to add documents to vectorstore: Database error")
        ingester.retriever = mock_retriever

        chunks = [Document(page_content="chunk", metadata={})]

        with pytest.raises(RuntimeError, match="Failed to add documents to vectorstore"):
            ingester.add_to_vectorstore(chunks)


class TestVerifyIngestion:
    """Test verify_ingestion method."""

    @patch('scripts.ingest.get_chroma_client')
    def test_verify_ingestion_success(self, mock_get_client, ingester):
        """Test verification of ingestion."""
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        total = ingester.verify_ingestion()

        assert total == 42
        mock_client.get_collection.assert_called_once_with("test_collection")
        mock_collection.count.assert_called_once()


class TestIngestPipeline:
    """Test the complete ingest() pipeline."""

    @patch('scripts.ingest.get_chroma_client')
    @patch('scripts.ingest.create_baseline_retriever')
    def test_ingest_full_pipeline(self, mock_factory, mock_get_client, tmp_path):
        """Test complete ingestion pipeline."""
        # Setup
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nContent here.")

        ingester = DocumentIngester(
            docs_dir=str(docs_dir),
            collection_name="test_kb"
        )

        # Mock retriever
        mock_retriever = MagicMock()
        mock_chunks = [
            Document(page_content="chunk1", metadata={'chunk_id': 0}),
            Document(page_content="chunk2", metadata={'chunk_id': 1}),
        ]
        mock_retriever.chunk_documents.return_value = mock_chunks
        mock_factory.return_value = mock_retriever

        # Mock collection
        mock_collection = Mock()
        mock_collection.count.return_value = 2
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        result = ingester.ingest(
            chroma_host="localhost",
            chroma_port=8000,
            re_ingest=False
        )

        # Verify result
        assert result['documents_processed'] == 1
        assert result['chunks_created'] == 2
        assert result['total_in_collection'] == 2
        assert result['collection_name'] == "test_kb"

        # Verify calls
        mock_retriever.chunk_documents.assert_called_once()
        mock_retriever.build_bm25_index.assert_called_once_with(mock_chunks)
        mock_retriever.add_documents.assert_called_once_with(mock_chunks)  # Use Retriever interface

    @patch('scripts.ingest.get_chroma_client')
    @patch('scripts.ingest.create_baseline_retriever')
    def test_ingest_with_reingestion(self, mock_factory, mock_get_client, tmp_path):
        """Test ingestion with re-ingest flag."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test")

        ingester = DocumentIngester(docs_dir=str(docs_dir))

        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 1
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        mock_retriever = MagicMock()
        mock_retriever.chunk_documents.return_value = [
            Document(page_content="chunk", metadata={})
        ]
        mock_factory.return_value = mock_retriever

        ingester.ingest(re_ingest=True)

        # Verify collection was deleted
        mock_client.delete_collection.assert_called_once()

    @patch('scripts.ingest.create_baseline_retriever')
    def test_ingest_no_documents_found(self, mock_factory, tmp_path):
        """Test error when no documents found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        ingester = DocumentIngester(docs_dir=str(empty_dir))

        with pytest.raises(ValueError, match="No documents found"):
            ingester.ingest()

    @patch('scripts.ingest.create_baseline_retriever')
    def test_ingest_nonexistent_directory(self, mock_factory, tmp_path):
        """Test error when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        ingester = DocumentIngester(docs_dir=str(nonexistent))

        with pytest.raises(FileNotFoundError):
            ingester.ingest()

    @patch('scripts.ingest.create_baseline_retriever')
    @patch('scripts.ingest.get_chroma_client')
    def test_ingest_load_failure_raises(self, mock_get_client, mock_factory, tmp_path):
        """Test that if all documents fail to load, ingestion raises error."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create a file that will fail to load (mock will handle this)
        (docs_dir / "test.md").write_text("# Test")

        ingester = DocumentIngester(docs_dir=str(docs_dir))

        # Mock TextLoader to fail
        with patch('scripts.ingest.TextLoader') as mock_loader:
            mock_loader.side_effect = Exception("Load failed")

            with pytest.raises(ValueError, match="No documents were successfully loaded"):
                ingester.ingest()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_document_list(self, ingester):
        """Test handling of empty document list."""
        ingester.retriever = Mock()

        # Should handle empty list gracefully
        documents = ingester.load_documents([], [])
        assert documents == []

    def test_large_number_of_files(self, tmp_path):
        """Test handling of many files."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create 100 test files
        for i in range(100):
            (docs_dir / f"test{i}.md").write_text(f"# Test {i}")

        ingester = DocumentIngester(docs_dir=str(docs_dir))
        md_files, pdf_files = ingester.find_documents()

        assert len(md_files) == 100

    def test_nested_directories(self, tmp_path):
        """Test finding files in nested directories."""
        docs_dir = tmp_path / "documents"
        subdir1 = docs_dir / "subdir1"
        subdir2 = docs_dir / "subdir1" / "subdir2"

        docs_dir.mkdir()
        subdir1.mkdir()
        subdir2.mkdir()

        (docs_dir / "root.md").write_text("# Root")
        (subdir1 / "sub1.md").write_text("# Sub1")
        (subdir2 / "sub2.md").write_text("# Sub2")

        ingester = DocumentIngester(docs_dir=str(docs_dir))
        md_files, pdf_files = ingester.find_documents()

        # rglob should find all nested files
        assert len(md_files) == 3
