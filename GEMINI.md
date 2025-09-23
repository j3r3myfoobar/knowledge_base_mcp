## Project Overview

This project is a local Retrieval-Augmented Generation (RAG) system. It ingests documents from a specified directory, creates vector embeddings using HuggingFace sentence-transformers, and stores them in a ChromaDB. A FastAPI server exposes a `/query` endpoint to search the knowledge base. The goal is to allow an LLM to query this local data to answer questions with specific context.

**Key Technologies:**

*   **Python:** The core language for all scripts.
*   **FastAPI:** Used to create the web server that exposes the `/query` endpoint.
*   **ChromaDB:** The vector database used to store document embeddings.
*   **LangChain:** Used for document loading and text splitting.
*   **HuggingFace Transformers:** Provides the `all-MiniLM-L6-v2` model for creating sentence embeddings.
*   **Docker & Docker Compose:** Used to containerize and manage the application services.

**Architecture:**

The system is composed of two main services managed by `docker-compose.yml`:

1.  **`ingester`:** A service that runs the `ingest.py` script. This script scans the `documents/` directory, processes new or modified files, and updates the ChromaDB with fresh vector embeddings. It is designed to be run on-demand.
2.  **`mcp_server`:** A long-running FastAPI server defined in `main.py`. It provides a `/query` endpoint that takes a natural language query, embeds it, and searches the ChromaDB for the most relevant document chunks.

## Building and Running

The project is managed via shell scripts and Docker Compose.

**Prerequisites:**

*   Docker
*   Docker Compose

**Starting the Server:**

To start the FastAPI server without re-indexing the documents:

```bash
bash start_mcp.sh
```

**Ingesting New or Modified Documents:**

To re-ingest all documents (a full reset) and then start the server:

```bash
bash start_mcp.sh --re-ingest
```

You can also specify a different directory for your documents:

```bash
bash start_mcp.sh --re-ingest ./my_other_docs
```

**Stopping the Server:**

To stop all running Docker containers:

```bash
bash stop_mcp.sh
```

**Querying the Knowledge Base:**

Once the server is running, you can query it via the `/query` endpoint:

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{
      "query": "What are the best practices for AI agent development?"
    }'
```

## Development Conventions

*   **Configuration:** Key settings like the ChromaDB collection name (`knowledge_base`) and the embedding model (`all-MiniLM-L6-v2`) are hardcoded in both `ingest.py` and `main.py`.
*   **Idempotency:** The `ingest.py` script is designed to be idempotent. It tracks file modification times to only process new or changed files, and it can delete records for files that have been removed from the source directory.
*   **Dependencies:** Python dependencies are listed in `requirements.txt`.
*   **Containerization:** The `Dockerfile` and `docker-compose.yml` define the container environment and service orchestration.
