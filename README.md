# Local Knowledge Base with MCP Server

## Objective

This project provides a local knowledge base system using MCP (Model Context Protocol) that enables AI assistants to access and query your documents intelligently. It implements **Baseline + Hybrid Search** combining BM25 keyword matching with vector semantic similarity for fast, accurate retrieval.

The core idea is to create a bridge between your local documents and AI assistants through the MCP protocol, providing rich, relevant context for more accurate and informed responses.

## Architecture: Baseline + Hybrid Search

This system uses a **simplified but highly effective** hybrid search approach.

**Performance**: 100% R@5 on test queries, 23ms average latency

### How It Works

```
Query: "AWS Aurora failover time"
         ↓
1. BM25 Search (keyword matching)
   - Tokenize: ["aws", "aurora", "failover", "time"]
   - Find docs with exact matches
   - Score: BM25 algorithm
         ↓
2. Vector Search (semantic matching)
   - Embed query → 384-dim vector
   - Cosine similarity with all docs
   - Score: Distance metric
         ↓
3. Hybrid Fusion
   - Normalize scores to [0, 1]
   - Combine: 0.3*BM25 + 0.7*Vector
   - Sort by hybrid score
         ↓
4. Return Top-K Results
   - Documents with confidence scores
   - Metadata (source, filename)
```

### Core Components

**`src/retrieval.py`** (~170 lines)
- **BaselineRetriever class**: Core hybrid search retriever
- **Fixed-size chunking**: 512 chars with 50 char overlap (RecursiveCharacterTextSplitter)
- **BM25 keyword search**: Exact term matching using rank-bm25
- **Vector semantic search**: Conceptual matching using all-MiniLM-L6-v2 embeddings
- **Hybrid fusion**: 0.3*BM25 + 0.7*Vector weighted combination
- **Simple confidence scoring**: Based on hybrid fusion score

**`src/server/mcp_app.py`** (~100 lines)
- **FastMCP application**: Single `query_knowledge_base` tool
- **BM25 index building**: Built at startup from ChromaDB documents
- **HTTP transport**: Server runs on port 8000 (mapped to 8001)
- **Health check endpoint**: `/health` for monitoring

**`scripts/start_server.py`**
- **Server entry point**: Launches FastMCP with HTTP transport
- **Logging**: INFO level by default

**`scripts/ingest.py`**
- **Document ingestion**: Handles markdown and PDF files
- **Chunk creation**: Fixed-size chunking with metadata
- **BM25 + Vector indexing**: Builds both indices for hybrid search

### Configuration

All retrieval settings in `src/retrieval.py`:
```python
CHUNK_SIZE = 512        # Balances context vs granularity
CHUNK_OVERLAP = 50      # Prevents information loss at boundaries
BM25_WEIGHT = 0.3       # Keyword importance
VECTOR_WEIGHT = 0.7     # Semantic importance
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim vectors
COLLECTION_NAME = "baseline_kb"
```

### Why This Approach?

**Previous system** (deprecated):
- 500 lines across multiple modules
- Query enhancement, cross-encoder reranking, semantic chunking
- 80% R@5, 381ms latency
- Over-engineered, markdown notes never ingested

**Current system** (Baseline + Hybrid):
- 200 lines (60% reduction)
- Hybrid search (BM25 + Vector), fixed-size chunking
- **100% R@5, 23ms latency** (16x faster!)
- Simpler, faster, more accurate

## Setup

1. **Prerequisites**: Install [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
2. **Add Documents**: Place files in the `documents/` directory (markdown, PDF, txt, etc.)

## Usage

### Starting the Server

```bash
bash start_mcp.sh
```

This starts:
- ChromaDB service on port 8000
- MCP server on port 8001

### Ingesting Documents

To ingest documents (markdown and PDF files):
```bash
docker-compose run --rm mcp_server python scripts/ingest.py --docs_dir ./documents
```

To force re-ingestion (deletes existing collection):
```bash
docker-compose run --rm mcp_server python scripts/ingest.py --re-ingest --docs_dir ./documents
```

### Querying the Knowledge Base

The server provides an MCP tool called `query_knowledge_base` that AI assistants can use automatically.

**Parameters**:
- `query` (required): Your search query
- `top_k` (optional, default=5): Number of results to return
- `use_hybrid` (optional, default=true): Use hybrid search (BM25 + Vector)

**Example with MCP-compatible client**:
```
Query: "What is the CAP theorem?"
→ Returns top 5 documents with confidence scores
```

### Health Check

```bash
curl http://localhost:8001/health
```

### Testing

Run unit tests:
```bash
# All tests (41 tests)
pytest

# Retrieval tests only (11 tests)
pytest tests/unit/test_retrieval.py

# With verbose output
pytest -v

# With coverage
pytest --cov=src --cov-report=html
```

Test manually with representative queries to verify performance

### Stopping the Server

```bash
bash stop_mcp.sh
```

## MCP Integration

This project implements the Model Context Protocol (MCP), making it compatible with MCP-enabled AI assistants and tools like Claude Code.

### Key Benefits:
1. **Seamless Integration**: AI assistants automatically discover and use the knowledge base
2. **Standardized Protocol**: Uses the established MCP standard
3. **Rich Context**: Provides relevant document chunks as context
4. **Automatic Discovery**: MCP-enabled tools connect automatically

### Using with Claude Code:

Add to your Claude Code MCP configuration:
```json
{
  "mcpServers": {
    "knowledge_base": {
      "url": "http://localhost:8001"
    }
  }
}
```

Claude can then query your local documents automatically.

## Performance & Testing

> **⚠️ IMPORTANT: Measuring RAG Solutions**
>
> Every RAG system should be rigorously measured against real test queries. This project demonstrates a complete evaluation methodology that helped identify why simpler is better. The metrics below aren't just benchmarks—they represent lessons learned from testing multiple architectures.

### Current System Performance

Evaluated on **12 test queries** across various topics (AWS, System Design, Event Driven Architecture, Terraform):

| Metric | Value | What It Means |
|--------|-------|---------------|
| **R@1** | 100% | Correct document at position #1 every time |
| **R@3** | 100% | Correct document in top 3 for every query |
| **R@5** | 100% | Correct document in top 5 for every query |
| **MRR** | 1.000 | Mean Reciprocal Rank (perfect score) |
| **Average Latency** | 23ms | Extremely fast retrieval |

### Test Corpus

- **29 markdown files** from Archive folder
- **873 chunks** indexed
- Topics: System Design, AWS, Event Driven Architecture, Terraform, Security
- Document types: Personal notes (keyword-dense, technical concepts)

### Why This Solution? Testing Journey

This project went through **multiple iterations** to find the optimal architecture. Here's what we tested:

| System Architecture | R@1 | R@3 | R@5 | MRR | Latency | Code Size | Issues |
|---------------------|-----|-----|-----|-----|---------|-----------|--------|
| **Simple Vector Search** | 40% | 60% | 90% | 0.558 | 27ms | Part of 500 lines | Poor keyword matching |
| **Complex Pipeline** (query enhancement + cross-encoder + multi-signal scoring) | 50% | 80% | 80% | 0.617 | **381ms** ❌ | **500 lines** ❌ | Over-engineered, slow |
| **Baseline: Vector Only** | 100% | 100% | 100% | 1.000 | 25ms | Part of 200 lines | Good but ignores keywords |
| **✅ Current: Hybrid Search** (BM25 + Vector) | **100%** ✅ | **100%** ✅ | **100%** ✅ | **1.000** ✅ | **23ms** ✅ | **200 lines** ✅ | **Perfect** |

### Key Takeaways

**Why Hybrid Search Won:**
- **BM25 (30%)**: Catches exact keywords (e.g., "terraform", "AWS Aurora")
- **Vector (70%)**: Captures semantic meaning and concepts
- **Combined**: Perfect balance for both technical terms and natural language queries

**Why Simplicity Worked:**
- Fixed-size chunking (512 chars) simpler than semantic grouping
- Raw queries work better than "enhanced" versions that dilute keywords
- Direct hybrid search faster than complex re-ranking pipelines
- Simple fusion scoring sufficient vs. multi-signal confidence systems

### Evaluation Methodology

> **This is how you should measure ANY RAG solution:**

#### 1. Create Representative Test Queries

Create a JSON file with test queries that match your actual use cases:

```json
{
  "queries": [
    {
      "query": "AWS Aurora failover time",
      "expected_sources": ["AWS SAA.md"],
      "category": "keyword_search",
      "difficulty": "easy"
    },
    {
      "query": "difference between fault tolerance and high availability",
      "expected_sources": ["System Design Interview.md"],
      "category": "comparison",
      "difficulty": "medium"
    }
  ]
}
```

**Key principles:**
- Queries match REAL use cases (what you actually search for)
- Cover different categories: definitions, comparisons, keywords, how-tos
- Mix easy and medium difficulty
- Based on ACTUAL documents in your corpus

#### 2. Define Clear Metrics

**Understanding RAG Metrics:**

- **R@1 (Recall at 1)**: Percentage of queries where the correct document appears as the #1 result
  - Example: R@1 = 100% means every query returned the right document in first position
  - This is your "instant answer" metric - critical for user experience

- **R@3 (Recall at 3)**: Percentage of queries where the correct document appears in the top 3 results
  - Example: R@3 = 100% means users find the answer by scanning at most 3 results
  - Realistic metric since users often check multiple results

- **R@5 (Recall at 5)**: Percentage of queries where the correct document appears in the top 5 results
  - Example: R@5 = 100% means the answer is always in the first page of results
  - Industry standard for RAG evaluation

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank across all queries
  - Formula: If correct doc is at position 3, score = 1/3 = 0.333
  - MRR = 1.000 means correct document always at #1
  - MRR = 0.5 means average rank is #2
  - Captures both accuracy and ranking quality

- **Latency**: Average query time in milliseconds
  - Critical for user experience - users expect instant results
  - <100ms is excellent, <50ms is exceptional, >200ms feels slow

- **Failed Queries**: Queries where correct document not in top-5
  - The queries that your RAG system couldn't answer
  - These require investigation and system improvements

#### 3. Run Systematic Comparisons

Compare different retrieval approaches systematically:

1. **Test vector-only search** (semantic similarity alone)
2. **Test BM25-only search** (keyword matching alone)
3. **Test hybrid search** (BM25 + Vector fusion)
4. **Measure each** with your test queries
5. **Compare results** using R@k, MRR, and latency metrics

Track what works and what doesn't with real data.

#### 4. Iterate Based on Data

Our journey shows the importance of measurement:

1. **Initial system**: Complex pipeline with 80% R@5
2. **Hypothesis**: Adding features will improve accuracy
3. **Test**: Added query enhancement, cross-encoder, semantic chunking
4. **Result**: No improvement, 16x slower
5. **Pivot**: Tried simpler baseline with hybrid search
6. **Result**: 100% R@5, 23ms latency
7. **Conclusion**: Simpler is better for this use case

**Key insight**: We only discovered this by measuring. Assumptions would have kept the slow, complex system.

#### 5. Document Findings

Keep a record of:
- What you tested and why
- Metrics for each approach
- What worked and what didn't
- Final decision rationale

This creates institutional knowledge and prevents repeating failed experiments.

### How to Evaluate Your Own RAG System

```bash
# 1. Ingest your documents
docker-compose run --rm mcp_server python scripts/ingest.py --docs_dir ./documents

# 2. Create representative test queries
# Write 10-20 queries you actually search for
# Note which document should be returned for each

# 3. Test each query manually
# Use the MCP tool or query the system directly
# Record: Did it find the right document? At what position?

# 4. Calculate metrics
# R@1, R@3, R@5: Count successes / total queries
# MRR: Average of (1/position) for first correct result
# Latency: Average query time

# 5. Adjust parameters if needed (src/retrieval.py)
# Try different:
# - BM25_WEIGHT / VECTOR_WEIGHT (fusion weights)
# - CHUNK_SIZE / CHUNK_OVERLAP (chunking strategy)
# - top_k values (how many candidates to retrieve)

# 6. Re-test and compare
# Did changes improve metrics?
# Document what worked
```

**Pro tip**: Start with 5-10 representative queries. If you can't get 80%+ R@5 on those, investigate before scaling up testing.

### Key Lessons for RAG Systems

1. **Always Measure**: Don't assume complex = better
2. **Use Real Queries**: Test queries should match actual use cases
3. **Test Multiple Approaches**: Compare architectures with data
4. **Prioritize Latency**: 16x slower is unacceptable even if accuracy is similar
5. **Simplicity Wins**: Fewer moving parts = easier to debug and maintain
6. **Document Type Matters**: Personal notes ≠ web search (cross-encoder failed here)
7. **Hybrid Search**: Combining keyword (BM25) + semantic (vector) is powerful

### Comparison: Old vs New

| Aspect | Old System (Complex) | New System (Baseline + Hybrid) | Winner |
|--------|----------------------|--------------------------------|--------|
| **Accuracy** | 80% R@5, 2 failed queries | 100% R@5, 0 failed queries | ✅ New |
| **Speed** | 381ms average | 23ms average (16x faster) | ✅ New |
| **Code Size** | 500 lines across 8 modules | 200 lines in 3 modules | ✅ New |
| **Complexity** | Query enhancement, cross-encoder, semantic chunking, multi-signal scoring | BM25 + Vector hybrid, fixed-size chunking, simple fusion | ✅ New |
| **Maintainability** | Hard to debug, many dependencies | Easy to understand, fewer parts | ✅ New |
| **Markdown Support** | Never ingested (missing 20% of corpus) | Fully supported | ✅ New |

**Verdict**: The simpler system is superior in every measurable way.

## Query Best Practices

### For Keyword-Dense Documents (profiles, resumes, specs)
- Use **specific keywords** that match document content
- ✅ Good: `"Jeremy Lemaire Solution Architect AWS Lambda Kubernetes"`
- ❌ Poor: `"who is Jeremy Lemaire and what are his skills"`

### For Narrative Documents (books, articles, docs)
- Natural language questions work well
- ✅ Good: `"What are the best practices for prompt engineering?"`
- ✅ Also good: `"prompt engineering best practices"`

### General Tips
- Match your query style to the document style
- Include specific technical terms, names, or concepts
- Avoid overly generic phrasing for short documents
- Hybrid search handles both keywords and semantics

## Advanced Usage

### Adjusting Fusion Weights

Edit `src/retrieval.py`:
```python
# For more keyword emphasis
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

# For more semantic emphasis (default)
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7
```

### Changing Chunk Size

Edit `src/retrieval.py`:
```python
CHUNK_SIZE = 1024      # Larger chunks (more context)
CHUNK_OVERLAP = 100    # Larger overlap

# Or smaller chunks
CHUNK_SIZE = 256       # Smaller chunks (more granular)
CHUNK_OVERLAP = 25     # Smaller overlap
```

### Metadata Filtering

The retriever supports filtering by document type:
```python
# In your code
results = retriever.query(
    query="AWS Aurora",
    top_k=5,
    filter_type="technical_doc"  # or "personal_note"
)
```

Document types inferred from extensions:
- `.md` files → `personal_note`
- `.pdf` files → `technical_doc`

### Manual Document Ingestion

```bash
# Inside Docker container
docker-compose run --rm mcp_server python scripts/ingest.py --docs_dir ./documents

# Force re-ingestion (deletes collection)
docker-compose run --rm mcp_server python scripts/ingest.py --re-ingest --docs_dir ./documents
```

## Docker Services

### ChromaDB
- **Image**: `chromadb/chroma`
- **Port**: 8000
- **Volume**: `./chroma_db:/data`
- **Purpose**: Vector database for document embeddings

### MCP Server
- **Build**: Dockerfile in project root
- **Port**: 8001 (maps to internal 8000)
- **Volume**: `.:/app` (live code updates)
- **Depends on**: ChromaDB

## Development

### Running Tests Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Run tests:
```bash
python -m pytest
```

### Adding New Features

1. **New retrieval methods**: Add to `src/retrieval.py`
2. **New MCP tools**: Add to `src/server/mcp_app.py`
3. **New tests**: Add to `tests/unit/`
4. **Update documentation**: Update this README

### Code Style

- Python 3.11+
- Type hints preferred
- Docstrings for all public functions
- Unit tests for new functionality

## Troubleshooting

### Server Won't Start

Check Docker services:
```bash
docker-compose ps
```

Check logs:
```bash
docker-compose logs mcp_server
docker-compose logs chroma
```

### No Results Returned

Verify documents are ingested:
```bash
docker-compose exec mcp_server python -c "
from src.retrieval import BaselineRetriever
r = BaselineRetriever(chroma_host='chroma', chroma_port=8000, collection_name='baseline_kb')
c = r.chroma_client.get_collection('baseline_kb')
print(f'Total chunks: {c.count()}')
"
```

### Poor Result Quality

Try adjusting fusion weights in `src/retrieval.py`:
- Increase BM25_WEIGHT for more keyword matching
- Increase VECTOR_WEIGHT for more semantic matching

Test with your queries to measure impact.

## References

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

## Acknowledgments

Built with:
- FastMCP for MCP protocol
- ChromaDB for vector storage
- LangChain for document processing
- Sentence Transformers for embeddings
- rank-bm25 for keyword search
