# Local Knowledge Base with MCP Server

## Objective

This project provides a simple yet powerful way to create a local knowledge base from your documents. It allows a Large Language Model (LLM) like Gemini or Claude, when used in a CLI environment, to query this knowledge base and use the retrieved information as context to answer questions.

The core idea is to enable an AI assistant to access and "read" your local files, giving it a memory and knowledge domain specific to your needs.

## How It Works

The project consists of a few key components:

- **`ingest.py`**: A script that scans the `documents` directory for supported files (PDFs, text files, Markdown, etc.). To speed up the process, it uses all available CPU cores to process files in parallel. It splits the documents into manageable chunks and stores their vector embeddings in a ChromaDB database. This script is designed to be idempotent, only updating the database when files are new or have been modified.

### RAG Principles: Chunking and Retrieval Considerations

For effective Retrieval-Augmented Generation (RAG), documents are broken down into smaller, semantically coherent "chunks" during ingestion. This is crucial for several reasons:

*   **LLM Context Window Limits:** Large Language Models (LLMs) have a finite input size (context window). Sending entire large documents would quickly exceed this limit.
*   **Retrieval Precision:** Smaller chunks allow for more precise vector embeddings, improving the accuracy of the semantic search.

The `ingest.py` script currently uses a **chunk size of 1000 characters** with a **chunk overlap of 200 characters**. The `chunk_overlap` helps maintain context across chunk boundaries.

When a query is made via the `/query` endpoint, the `query_knowledge_base` tool retrieves the `top_k` most relevant chunks from the ChromaDB. The current `top_k` value is set to **10**.

It's important to note that the sum of the content of these `top_k` chunks (plus the query and any other prompt instructions) must fit within the context window of the LLM you are using. For reference, this Gemini model has a context window of **1 million tokens**.

Optimizing `chunk_size`, `chunk_overlap`, and `top_k` is an iterative process that depends on your specific documents and the LLM being used, aiming to balance retrieval relevance, LLM comprehension, and cost efficiency.
- **`main.py`**: A FastAPI server that provides a simple API to interact with the knowledge base. It exposes a `/query` endpoint that accepts a natural language query and returns the most relevant document chunks.
- **`docker-compose.yml`**: Manages the two main services:
    -   `ingester`: A short-lived service that runs `ingest.py` to update the knowledge base.
    -   `mcp_server`: The long-running FastAPI server.
- **`chroma_db/`**: The directory where the ChromaDB vector database is persisted.
- **`documents/`**: The directory where you should place your knowledge base files.

## Setup

1.  **Prerequisites**: You must have [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
2.  **Add Documents**: Place the files you want to include in your knowledge base into the `documents/` directory. The ingestion script supports a wide range of file types, including PDFs, Markdown, text files, and more.

## Usage

The project uses simple shell scripts to manage the services.

### Starting the Server

To start the MCP server, run:

```bash
bash start_mcp.sh
```

By default, this will **not** re-ingest the documents. It will just start the server, which is useful for quick restarts.

### Ingesting New or Modified Documents

If you have added, modified, or removed documents in the `documents/` directory, you need to tell the system to re-ingest them. Use the `--re-ingest` flag:

```bash
bash start_mcp.sh --re-ingest
```

This will run the ingestion process before starting the server. This can take some time depending on the number and size of your documents.

You can also specify a different directory for your documents:

```bash
bash start_mcp.sh --re-ingest ./my_other_docs
```

### Querying the Knowledge Base

Once the server is running, you can send queries to it using any HTTP client. Here is an example using `curl`:

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{
      "query": "What are the best practices for AI agent development?"
    }'
```

The server will respond with a JSON object containing the most relevant document chunks related to your query.

### Stopping the Server

To stop all running services, use the provided script:

```bash
bash stop_mcp.sh
```

## Integrating with a CLI Assistant (Gemini/Claude)

The `/query` endpoint is the key to integrating this with a CLI-based AI. A CLI tool can be configured to:

1.  Take a user's prompt.
2.  Send the prompt to the `/query` endpoint of the running `mcp_server`.
3.  Receive the relevant document chunks.
4.  Prepend this context to the user's original prompt before sending it to the LLM (e.g., Gemini API).

This provides the LLM with the necessary local context to answer questions about your documents accurately.

```