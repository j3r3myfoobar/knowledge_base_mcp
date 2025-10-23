# System Architecture

## Overview

This is a **local knowledge base system** that provides semantic document retrieval through the Model Context Protocol (MCP). The system enables AI assistants to query your local documents intelligently, using vector embeddings and sophisticated retrieval techniques.

## High-Level Data Flow

```
Documents → Ingestion Pipeline → Vector Store (ChromaDB)
                                        ↓
User Query → MCP Server → Retrieval Pipeline → Ranked Results
```

### 1. Ingestion Flow
```
Filesystem Documents
    ↓
Semantic Parsing (UnstructuredLoader)
    ↓
Semantic Chunking (group by paragraphs, titles, etc.)
    ↓
Vector Embeddings (HuggingFace all-MiniLM-L6-v2)
    ↓
ChromaDB Storage
```

### 2. Retrieval Flow
```
User Query
    ↓
Query Enhancement (convert to question format)
    ↓
Vector Similarity Search (retrieve top_k documents)
    ↓
Cross-Encoder Re-ranking (select most relevant top_n)
    ↓
Confidence Scoring (multi-signal fusion)
    ↓
Ranked Results
```

## System Components

### Core Layer (`src/core/`)
**Purpose**: Foundation components for database connections, configuration, and data models

- **`config.py`**: Centralized configuration
  - Database settings (ChromaDB host/port)
  - Model names (embeddings, cross-encoder)
  - Processing parameters (chunk size, batch size, workers)

- **`database.py`**: Singleton database connections
  - ChromaDB client (lazy initialization)
  - HuggingFace embeddings model
  - Vector store (Chroma with embeddings)
  - CrossEncoder model (with graceful fallback)

- **`models.py`**: Pydantic models for API
  - `Document`: Retrieved chunk with content, metadata, confidence
  - `KnowledgeBaseOutput`: List of documents for MCP response

### Ingestion Layer (`src/ingestion/`)
**Purpose**: Process documents from filesystem into vector store

- **`chunker.py`**: Semantic chunking logic
  - Groups semantic elements (paragraphs, titles, list items)
  - Preserves document structure
  - Maintains chunk overlap for context continuity

- **`processor.py`**: Single file processing
  - Loads file with UnstructuredLoader
  - Applies semantic chunking
  - Enriches metadata (source path, modification time)
  - Filters complex metadata types

- **`sync.py`**: Vector store synchronization
  - Compares filesystem vs vector store state
  - Deletes removed files
  - Ingests new/modified files incrementally
  - Parallel processing with ProcessPoolExecutor

### Retrieval Layer (`src/retrieval/`)
**Purpose**: Query processing and document retrieval

- **`query_enhancer.py`**: Query preprocessing
  - Converts statements to questions ("Python classes" → "what is Python classes?")
  - Optional abbreviation expansion
  - Improves semantic matching

- **`reranker.py`**: Cross-encoder re-ranking
  - Takes initial vector retrieval results
  - Scores query-document relevance more accurately
  - Returns top N most relevant documents

- **`scorer.py`**: Multi-signal confidence scoring
  - Cross-encoder score (40%)
  - Query overlap - keyword matching (30%)
  - Document length penalty (20%)
  - Metadata quality (10%)
  - Returns 0.0-1.0 confidence score

### Server Layer (`src/server/`)
**Purpose**: MCP protocol interface

- **`mcp_app.py`**: FastMCP application
  - Defines `query_knowledge_base` tool
  - Orchestrates retrieval pipeline
  - Handles graceful fallback when cross-encoder unavailable
  - Returns structured results to MCP clients

### Scripts (`scripts/`)
**Purpose**: CLI entry points

- **`ingest.py`**: Document ingestion CLI
  - Argument parsing (--re-ingest, --docs_dir)
  - Collection deletion for full re-ingestion
  - Calls synchronization logic

- **`start_server.py`**: MCP server entry point
  - Configures logging
  - Launches FastMCP server on port 8000

## Key Design Patterns

### 1. Standard RAG (Retrieval-Augmented Generation)
- **Initial retrieval**: Vector similarity search in ChromaDB
- **Enhancement**: Semantic chunking preserves document structure

### 2. Corrective RAG
- **Two-stage retrieval**: Initial retrieval + cross-encoder re-ranking
- **Graceful fallback**: System works even if cross-encoder fails to load

### 3. Fusion RAG
- **Query enhancement**: Preprocessing for better matching
- **Multi-signal scoring**: Combines multiple relevance indicators

### 4. Modular RAG
- **Swappable components**: Each module has single responsibility
- **Singleton pattern**: Expensive resources (models, DB) initialized once
- **Parallel processing**: Multi-core file processing for ingestion

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Vector DB | ChromaDB | Document storage and similarity search |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Convert text to vectors |
| Re-ranking | CrossEncoder ms-marco-MiniLM-L-6-v2 | Accurate relevance scoring |
| Document Parsing | Unstructured | Semantic document structure extraction |
| MCP Server | FastMCP | Model Context Protocol interface |
| Orchestration | Docker Compose | Service management |

## Performance Characteristics

### Ingestion
- **Parallelism**: Uses all CPU cores via ProcessPoolExecutor
- **Incremental**: Only processes new/modified files
- **Batching**: Adds documents in batches (default 5000) to avoid memory issues

### Retrieval
- **Initial retrieval**: ~50-100ms for top 20 documents
- **Re-ranking**: ~50-100ms additional for cross-encoder
- **Total query time**: ~100-200ms end-to-end

### Scalability
- **Documents**: Tested with 1000s of documents
- **Collection size**: ChromaDB handles millions of vectors
- **Memory**: ~500MB-1GB for models + vector store

## Configuration Points

All configurable in `src/core/config.py`:

```python
# Vector Store
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000
COLLECTION_NAME = "knowledge_base"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
INGESTION_BATCH_SIZE = 5000
MAX_WORKERS = os.cpu_count() or 4
```

## Extension Points

### Add New Retrieval Method
Create new module in `src/retrieval/` and update `mcp_app.py` to use it

### Change Chunking Strategy
Modify `src/ingestion/chunker.py` - the interface stays the same

### Add New Scoring Signal
Update `calculate_confidence_score()` in `src/retrieval/scorer.py`

### Switch Embedding Model
Change `EMBEDDING_MODEL` in config - ensure model is compatible with sentence-transformers

## Error Handling

### Ingestion
- **File processing errors**: Logged, but don't stop other files
- **Worker failures**: Caught and logged per-file
- **Empty documents**: Skipped with warning

### Retrieval
- **Cross-encoder failure**: Falls back to simple vector retrieval
- **Empty results**: Returns empty list with no error
- **Database connection**: Fails fast with critical error on startup

## Logging Strategy

- **INFO**: User-facing operations (ingestion progress, query results)
- **WARNING**: Fallback modes, skipped files
- **ERROR**: Processing failures, connection issues
- **DEBUG**: Detailed operation traces (in FastMCP server)

## Directory Structure

```
mcp_server/
├── src/
│   ├── core/           # Database, config, models
│   ├── ingestion/      # Document processing
│   ├── retrieval/      # Query processing
│   └── server/         # MCP interface
├── scripts/            # CLI entry points
├── docs/               # Documentation
├── tests/              # Test suite
├── documents/          # Your documents (ingested)
├── chroma_db/          # Vector store (persisted)
└── docker-compose.yml  # Service orchestration
```

## Future Enhancements

See README.md TODO section for planned features like Fusion RAG (BM25 + vector retrieval).
