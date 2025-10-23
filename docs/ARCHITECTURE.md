# System Architecture

## Overview

This document explains the **architectural decisions** behind the Baseline + Hybrid Search system. For usage instructions, setup, and testing, see the main [README.md](../README.md).

**Performance**: 100% R@5 on test queries, 23ms average latency

## Core Design Philosophy

**Simplicity over sophistication**: Through rigorous testing, we found that a simple hybrid search (BM25 + Vector) outperformed complex pipelines with query enhancement, cross-encoder re-ranking, and multi-signal scoring.

- **200 lines** of core retrieval logic vs 500+ in previous version
- **100% R@5** accuracy vs 80% with complex pipeline
- **23ms** latency vs 381ms with complex pipeline (16x faster)

## Architecture Diagram

```
Documents → Ingestion → ChromaDB (BM25 + Vector Indices)
                              ↓
User Query → MCP Server → Hybrid Search → Ranked Results
```

### Ingestion Pipeline
```
Filesystem (Markdown, PDF)
    ↓
TextLoader / PyPDFLoader
    ↓
Fixed-Size Chunking (512 chars, 50 overlap)
    ↓
Vector Embeddings (all-MiniLM-L6-v2)
    ↓
BM25 Index (rank-bm25)
    ↓
ChromaDB Storage
```

### Retrieval Pipeline
```
User Query
    ↓
BM25 Keyword Search (top 20)
    ↓
Vector Semantic Search (top 20)
    ↓
Hybrid Fusion (0.3*BM25 + 0.7*Vector)
    ↓
Top-K Results with Confidence Scores
```

## Key Design Decisions

### 1. Why Hybrid Search?

Combines strengths of two complementary approaches:

**BM25 (30% weight)**:
- Catches exact keywords: "AWS Aurora", "terraform", "Kubernetes"
- Fast: ~5-10ms for top 20 documents
- No model loading required

**Vector Similarity (70% weight)**:
- Captures semantic meaning and concepts
- Handles synonyms and related terms
- ~10-15ms for top 20 documents

**Result**: Perfect balance for technical documentation where both exact terms and concepts matter.

### 2. Why Fixed-Size Chunking?

**Previous approach**: Semantic chunking with UnstructuredLoader
- Grouped by paragraphs, titles, list items
- Complex logic to maintain document structure
- Parallel processing overhead

**Current approach**: Fixed 512-character chunks with 50-character overlap
- **Simpler**: One RecursiveCharacterTextSplitter call
- **Faster**: No element grouping logic
- **More predictable**: Consistent chunk sizes
- **Works well**: Tested on documents from 7 words to 7,220 words

### 3. Why No Cross-Encoder?

We tested cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`):

**Results**:
- Added 380ms latency overhead
- No accuracy improvement (80% R@5 with and without)
- Model trained on MS MARCO (web search), not personal notes

**Decision**: Remove cross-encoder. Simple hybrid fusion achieved 100% R@5 with 23ms latency.

### 4. Why No Query Enhancement?

We tested converting queries to questions:

**Example**:
- Original: "AWS Aurora failover time"
- Enhanced: "what is AWS Aurora failover time?"

**Results**:
- Diluted keyword precision
- Reduced R@5 from potential 100% to 80%
- Added processing overhead

**Decision**: Raw queries perform better for keyword-dense technical documentation.

## Component Architecture

### Dependency Injection Pattern

All components use constructor injection for testability:

```python
# Create dependencies
chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=50)
bm25_index = BM25Index()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Inject into retriever
retriever = BaselineRetriever(
    vectorstore=vectorstore,
    embeddings=embeddings,
    chunker=chunker,
    bm25_index=bm25_index
)
```

**Benefits**:
- Easy to mock for unit tests (107 tests pass)
- Can swap implementations without changing code
- Factory pattern (`create_baseline_retriever()`) simplifies production use

### Interface-Based Design

Abstract interfaces in `src/core/interfaces.py`:

- **`Retriever`**: Query and document management
- **`DocumentChunker`**: Chunking strategies
- **`SearchIndex`**: Search implementations (BM25, TF-IDF, etc.)

**Benefits**:
- Components depend on abstractions, not concrete classes
- Easy to add new implementations
- Enforces consistent APIs

### Singleton Pattern for Expensive Resources

`src/core/database.py` uses singleton pattern for:
- ChromaDB client
- Embedding model
- Vectorstore

**Why**: These resources are expensive to initialize (model loading, network connections). Initialize once, reuse everywhere.

## Data Flow Details

### Document Ingestion

1. **Find files**: Scan `documents/` for `.md` and `.pdf` files
2. **Load**: Use LangChain loaders (TextLoader, PyPDFLoader)
3. **Chunk**: FixedSizeChunker creates 512-char chunks with metadata
4. **Embed**: all-MiniLM-L6-v2 creates 384-dim vectors
5. **Index**:
   - Vector: Store in ChromaDB
   - BM25: Build in-memory index with rank-bm25
6. **Persist**: ChromaDB persists to `chroma_db/`

### Query Processing

1. **BM25 Search**: Tokenize query, retrieve top 20 by BM25 score
2. **Vector Search**: Embed query, retrieve top 20 by cosine similarity
3. **Normalize Scores**:
   - BM25: Divide by max score → [0, 1]
   - Vector: Distance to similarity → 1/(1 + distance)
4. **Hybrid Fusion**: `hybrid_score = 0.3*BM25 + 0.7*Vector`
5. **Rank**: Sort by hybrid score, return top-k
6. **Format**: Convert to `KnowledgeBaseOutput` for MCP

### Hybrid Fusion Algorithm

Not Reciprocal Rank Fusion (RRF), but **weighted score fusion**:

```python
# BM25 results: [(doc1, 15.2), (doc2, 12.8), ...]
max_bm25 = 15.2
bm25_normalized = {
    doc1: 15.2/15.2 = 1.0,
    doc2: 12.8/15.2 = 0.84,
    ...
}

# Vector results: [(doc1, 0.3), (doc2, 0.5), ...]  # distances
vector_similarity = {
    doc1: 1/(1+0.3) = 0.77,
    doc2: 1/(1+0.5) = 0.67,
    ...
}

# Hybrid scores
hybrid = {
    doc1: 0.3*1.0 + 0.7*0.77 = 0.839,
    doc2: 0.3*0.84 + 0.7*0.67 = 0.721,
    ...
}
```

**Why not RRF?** RRF uses rank positions (1, 2, 3...). Score fusion preserves magnitude differences, better for technical docs where score gaps matter.

## Performance Characteristics

### Latency Breakdown (average)
- BM25 search: 5-10ms
- Vector search: 10-15ms
- Hybrid fusion: 3-5ms
- **Total**: 20-30ms

### Memory Usage
- Embedding model: ~100MB
- BM25 index: ~10-50MB (depends on corpus size)
- ChromaDB: ~200-300MB for vectors
- **Total**: ~300-500MB

### Scalability Tested
- **29 markdown files**, 873 chunks: 100% R@5, 23ms latency
- **Extrapolated**: Should scale to 10,000s of chunks with similar performance

## Testing Strategy

### Unit Tests (107 tests)
- Each component tested in isolation with mocks
- Tests for interfaces, implementations, edge cases
- Run: `pytest tests/unit/ -v`

### Integration Tests
- End-to-end workflows (ingestion → retrieval)
- MCP server interface
- Run: `pytest tests/integration/ -v`

## Extension Points

### Add New Chunking Strategy

Implement `DocumentChunker` interface:

```python
class SemanticChunker(DocumentChunker):
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        # Your semantic chunking logic
        pass
```

Use in factory:
```python
chunker = SemanticChunker()
retriever = create_baseline_retriever(chunker=chunker)
```

### Add New Search Method

Implement `SearchIndex` interface:

```python
class TFIDFIndex(SearchIndex):
    def build_index(self, documents: List[Document]) -> None:
        # Build TF-IDF index
        pass

    def search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        # Search TF-IDF index
        pass
```

### Adjust Fusion Weights

Experiment with different weights in `src/core/config.py`:

```python
BM25_WEIGHT = 0.5   # More keyword emphasis
VECTOR_WEIGHT = 0.5
```

Test with your queries to measure impact on R@k metrics.

## Lessons Learned

1. **Complex ≠ Better**: Our 500-line complex pipeline scored 80% R@5. Simple 200-line hybrid search scored 100%.

2. **Measure Everything**: We only discovered hybrid search superiority by testing. Assumptions would have kept the slow system.

3. **Domain Matters**: Cross-encoder trained on MS MARCO (web) didn't help with personal notes. Know your data.

4. **Keywords Matter**: For technical docs, exact keyword matching (BM25) is crucial. Pure semantic search missed important terms.

5. **Start Simple**: Build baseline first, add complexity only when metrics demand it.
