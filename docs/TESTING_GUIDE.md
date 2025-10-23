# Testing Guide - TDD Approach

**Target Audience:** Developers who want to write tests they could have written themselves using TDD

**Philosophy:** Tests should be simple, clear, and follow the "Red-Green-Refactor" cycle. No advanced Python features required.

---

## Table of Contents

1. [TDD Principles](#tdd-principles)
2. [Test Structure](#test-structure)
3. [Using Shared Fixtures](#using-shared-fixtures)
4. [Using Test Helpers](#using-test-helpers)
5. [Common Test Patterns](#common-test-patterns)
6. [Running Tests](#running-tests)
7. [Examples by Test Type](#examples-by-test-type)

---

## TDD Principles

### The TDD Cycle

```
1. RED: Write a failing test
2. GREEN: Write minimal code to make it pass
3. REFACTOR: Clean up the code while keeping tests green
```

### Our Testing Philosophy

✅ **DO:**
- Write tests that describe WHAT the code should do
- Use simple, clear English in test names
- Follow Arrange-Act-Assert structure
- Use shared fixtures and helpers
- Make tests independent (can run in any order)

❌ **DON'T:**
- Use advanced Python features (`__get__`, descriptors, etc.)
- Repeat test setup code
- Test implementation details
- Write tests that depend on each other

---

## Test Structure

### Standard Test Class Structure

```python
class TestComponentName:
    """Test suite for ComponentName.

    Brief description of what component does.
    """

    def test_MethodName_InputCondition_ExpectedOutcome(self, fixture_name):
        """Component should do X when Y happens."""
        # Arrange: Set up test data and dependencies
        component = ComponentName(config="test")
        test_data = "input"

        # Act: Execute the behavior being tested
        result = component.do_something(test_data)

        # Assert: Verify expected outcome
        assert result == "expected", "Description of what should happen"
```

### Test Naming Convention

We use **Roy Osherove's naming convention**: `[UnitOfWork_StateUnderTest_ExpectedBehavior]`

- **UnitOfWork**: The method or behavior being tested
- **StateUnderTest**: The condition or input scenario
- **ExpectedBehavior**: What should happen

✅ **Good:**
- `test_ChunkDocuments_LongDocument_SplitsIntoMultipleChunks`
- `test_Search_EmptyQuery_ReturnsEmptyList`
- `test_AddDocuments_VectorstoreFailure_RaisesRuntimeError`

❌ **Bad:**
- `test_1`
- `test_chunker`
- `test_edge_case`

**Reference**: https://osherove.com/blog/2005/4/3/naming-standards-for-unit-tests.html

---

## Using Shared Fixtures

### Available Fixtures (from `tests/conftest.py`)

#### Document Fixtures

```python
def test_example(sample_documents):
    """All fixtures are automatically available in tests."""
    # sample_documents: List[Document] - 3 varied documents
    # long_document: Document - Will be chunked into 2+ pieces
    # markdown_document: Document - For testing .md type inference
    # pdf_document: Document - For testing .pdf type inference
    pass
```

#### Mock Component Fixtures

```python
def test_with_mocks(
    mock_vectorstore,
    mock_embeddings,
    mock_chroma_client,
    mock_chunker,
    mock_bm25_index
):
    """Use pre-configured mocks instead of creating your own."""
    # All mocks have sensible default behaviors
    assert mock_vectorstore is not None
```

#### Combined Mock Fixture

```python
def test_with_all_mocks(mock_retriever_dependencies):
    """Get all mocked dependencies at once."""
    retriever = BaselineRetriever(**mock_retriever_dependencies)
    # Ready to test!
```

---

## Using Test Helpers

### Helper Functions (from `tests/unit/helpers.py`)

#### 1. `create_retriever_with_real_components()`

Use when you want to test actual behavior (integration test):

```python
def test_ChunkDocuments_LongDocument_SplitsIntoMultipleChunks(long_document):
    """Test real chunking with real FixedSizeChunker."""
    # Arrange
    retriever = create_retriever_with_real_components()

    # Act
    chunks = retriever.chunk_documents([long_document])

    # Assert
    assert len(chunks) > 1, "Long doc should be chunked"
```

#### 2. `create_fully_mocked_retriever()`

Use when you want to test orchestration logic (unit test):

```python
def test_ChunkDocuments_MockedChunker_DelegatesToComponent(sample_documents):
    """Test that retriever calls its components correctly."""
    # Arrange
    mock_chunker = Mock()
    mock_chunker.chunk_documents.return_value = sample_documents
    retriever = create_fully_mocked_retriever(chunker=mock_chunker)

    # Act
    result = retriever.chunk_documents(sample_documents)

    # Assert
    mock_chunker.chunk_documents.assert_called_once_with(sample_documents)
```

#### 3. `create_mock_chroma_collection_with_docs()`

Use when testing `initialize()` method:

```python
def test_Initialize_ExistingDocuments_BuildsBM25IndexFromChromaDB():
    """Test building BM25 index from existing documents."""
    # Arrange
    existing_docs = [
        {'content': 'doc1', 'metadata': {'source': 'test.md'}},
        {'content': 'doc2', 'metadata': {'source': 'test2.md'}},
    ]
    mock_client, _ = create_mock_chroma_collection_with_docs(existing_docs)

    # Act & Assert
    retriever = create_fully_mocked_retriever(chroma_client=mock_client)
    retriever.initialize()  # Should not raise
```

---

## Common Test Patterns

### Pattern 1: Testing a Method Returns Expected Output

```python
def test_Search_MatchingQuery_ReturnsRelevantDocuments():
    """Search should return documents that match the query."""
    # Arrange
    index = BM25Index()
    docs = [
        Document(page_content="Python programming", metadata={}),
        Document(page_content="Java programming", metadata={}),
    ]
    index.build_index(docs)

    # Act
    results = index.search("Python", top_k=1)

    # Assert
    assert len(results) == 1
    assert "Python" in results[0][0].page_content
```

### Pattern 2: Testing Error Handling

```python
def test_Process_NegativeInput_RaisesValueError():
    """Should raise ValueError when input is negative."""
    # Arrange
    component = MyComponent()

    # Act & Assert
    with pytest.raises(ValueError, match="must be positive"):
        component.process(-1)
```

### Pattern 3: Testing Delegation (Dependency Injection)

```python
def test_Execute_InjectedComponent_DelegatesToComponent():
    """Should call the injected component's method."""
    # Arrange
    mock_component = Mock()
    mock_component.do_work.return_value = "result"
    orchestrator = Orchestrator(component=mock_component)

    # Act
    result = orchestrator.execute()

    # Assert
    mock_component.do_work.assert_called_once()
    assert result == "result"
```

### Pattern 4: Testing State Changes

```python
def test_BuildIndex_ValidDocuments_UpdatesIndexSize():
    """Adding documents should update the index size."""
    # Arrange
    index = BM25Index()
    initial_size = index.get_index_size()

    # Act
    index.build_index([Document(page_content="test", metadata={})])

    # Assert
    assert index.get_index_size() > initial_size
    assert index.is_built()
```

---

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/unit/test_retrieval.py
```

### Run Specific Test Class

```bash
pytest tests/unit/test_retrieval.py::TestDocumentChunking
```

### Run Specific Test

```bash
pytest tests/unit/test_retrieval.py::TestDocumentChunking::test_splits_long_document_into_multiple_chunks
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Tests Matching Pattern

```bash
pytest -k "chunk"  # Runs all tests with "chunk" in the name
```

---

## Examples by Test Type

### Example 1: Testing a Pure Function

```python
def test_NormalizeScore_ValidDistance_ReturnsConfidenceInRange():
    """normalize_score should convert distance to confidence score."""
    # Arrange
    distance = 0.5

    # Act
    confidence = normalize_score(distance)

    # Assert
    assert 0.0 <= confidence <= 1.0
    assert isinstance(confidence, float)
```

### Example 2: Testing a Component (Real Behavior)

```python
def test_ChunkDocuments_LongDocument_SplitsIntoMultipleChunks(long_document):
    """Chunker should split long documents into multiple chunks."""
    # Arrange
    chunker = FixedSizeChunker(chunk_size=100)

    # Act
    chunks = chunker.chunk_documents([long_document])

    # Assert
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 110 for c in chunks)
```

### Example 3: Testing Integration (Multiple Components)

```python
def test_Query_HybridSearchEnabled_CombinesBM25AndVector():
    """Hybrid search should use both BM25 and vector search."""
    # Arrange
    retriever = create_retriever_with_real_components()
    docs = [Document(page_content="Python programming", metadata={})]
    retriever.build_bm25_index(docs)

    # Mock vector search to return known results
    retriever.vectorstore.similarity_search_with_score = Mock(
        return_value=[(docs[0], 0.3)]
    )

    # Act
    results = retriever.query("Python", top_k=1, use_hybrid=True)

    # Assert
    assert len(results) > 0
    assert 'confidence' in results[0]
```

### Example 4: Testing with Mocks (Orchestration Logic)

```python
def test_Query_HybridEnabled_CallsHybridSearch():
    """query() should call hybrid_search when use_hybrid=True."""
    # Arrange
    retriever = create_fully_mocked_retriever()
    retriever.hybrid_search = Mock(return_value=[])

    # Act
    retriever.query("test", use_hybrid=True)

    # Assert
    retriever.hybrid_search.assert_called_once()
```

---

## Writing Your First Test (TDD Style)

### Step 1: Write the Test (RED)

```python
def test_CalculateAverage_ValidNumbers_ReturnsMean():
    """calculate_average should return mean of numbers."""
    # Arrange
    numbers = [1, 2, 3, 4, 5]

    # Act
    result = calculate_average(numbers)

    # Assert
    assert result == 3.0
```

**Run:** `pytest tests/unit/test_mymodule.py`
**Result:** ❌ FAIL - `calculate_average` doesn't exist

### Step 2: Write Minimal Code (GREEN)

```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

**Run:** `pytest tests/unit/test_mymodule.py`
**Result:** ✅ PASS

### Step 3: Add Edge Cases

```python
def test_CalculateAverage_EmptyList_ReturnsZero():
    """calculate_average should return 0 for empty list."""
    assert calculate_average([]) == 0

def test_CalculateAverage_SingleNumber_ReturnsNumber():
    """calculate_average should return the number itself."""
    assert calculate_average([42]) == 42.0
```

### Step 4: Refactor

```python
def calculate_average(numbers):
    """Calculate the arithmetic mean of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
```

**Run:** `pytest tests/unit/test_mymodule.py`
**Result:** ✅ All tests pass

---

## Best Practices Checklist

Before committing your tests, verify:

- [ ] Test name describes expected behavior
- [ ] Test follows Arrange-Act-Assert structure
- [ ] Test uses fixtures/helpers instead of repeating setup
- [ ] Test has assertion messages for failures
- [ ] Test is independent (doesn't depend on other tests)
- [ ] Test doesn't use advanced Python features
- [ ] Test runs quickly (< 1 second)
- [ ] Test has a clear docstring

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Testing Implementation Details

```python
# BAD: Testing internal method
def test_internal_cache_structure():
    retriever = BaselineRetriever()
    assert retriever._cache == {}  # Implementation detail!
```

```python
# GOOD: Testing observable behavior
def test_Query_SameQuery_UsesCachedResults():
    retriever = create_retriever_with_real_components()
    retriever.query("test", top_k=5)
    retriever.query("test", top_k=5)  # Should use cache
    # Assert on behavior, not internals
```

### ❌ Mistake 2: Not Using Fixtures

```python
# BAD: Repeating setup
def test_1():
    doc = Document(page_content="test", metadata={'source': 'test.md'})
    ...

def test_2():
    doc = Document(page_content="test", metadata={'source': 'test.md'})
    ...
```

```python
# GOOD: Use fixture
def test_1(markdown_document):
    ...

def test_2(markdown_document):
    ...
```

### ❌ Mistake 3: Complex Test Logic

```python
# BAD: Too much logic in test
def test_complex():
    for i in range(10):
        if i % 2 == 0:
            result = process(i)
            assert result > 0
        else:
            ...
```

```python
# GOOD: Simple, focused tests
def test_Process_EvenNumber_ReturnsPositiveResult():
    result = process(2)
    assert result > 0

def test_Process_OddNumber_ReturnsPositiveResult():
    result = process(3)
    assert result > 0
```

---

## Getting Help

- **Read existing tests:** `tests/unit/test_retrieval.py` is a good reference
- **Check fixtures:** `tests/conftest.py` for available test data
- **Use helpers:** `tests/unit/helpers.py` for common setup patterns
- **Ask questions:** Add comments in your test PRs

---

## Quick Reference Card

```python
# Import what you need
import pytest
from unittest.mock import Mock
from langchain.docstore.document import Document

# Use fixtures
def test_with_fixtures(sample_documents, long_document):
    pass

# Create test objects
from tests.unit.helpers import create_retriever_with_real_components
retriever = create_retriever_with_real_components()

# Test structure
def test_MethodName_InputState_ExpectedResult(fixture):
    # Arrange
    component = Component()

    # Act
    result = component.method()

    # Assert
    assert result == expected, "Why this should be true"

# Run tests
pytest                           # All tests
pytest -v                        # Verbose
pytest -k "keyword"              # Match pattern
pytest tests/unit/test_file.py   # Specific file
```

---

**Remember:** If you can't explain your test to a junior developer in 30 seconds, it's too complex. Simplify it!
