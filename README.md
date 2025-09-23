# Local Knowledge Base with MCP Server

## Objective

This project provides a simple yet powerful way to create a local knowledge base from your documents. It allows a Large Language Model (LLM) like Gemini or Claude, when used in a CLI environment, to query this knowledge base and use the retrieved information as context to answer questions.

The core idea is to enable an AI assistant to access and "read" your local files, giving it a memory and knowledge domain specific to your needs.

## How It Works

The project consists of a few key components:

- **`ingest.py`**: A script that scans the `documents` directory for supported files (PDFs, text files, Markdown, etc.). To speed up the process, it uses all available CPU cores to process files in parallel. It splits the documents into manageable chunks and stores their vector embeddings in a ChromaDB database. This script is designed to be idempotent, only updating the database when files are new or have been modified.

### Better Chunking During Ingestion: A Semantic Approach

For effective Retrieval-Augmented Generation (RAG), documents must be broken down into smaller, semantically coherent "chunks." Instead of using a fixed character count, this project now employs a more intelligent, semantic-based approach.

We now use `UnstructuredLoader` in `"elements"` mode, which breaks a document down into its semantic parts (e.g., `Title`, `NarrativeText`, `ListItem`). The new function `group_elements_into_chunks` then intelligently combines these elements into chunks of a desired size. This ensures that paragraphs, list items, and titles stay together, creating far more coherent and meaningful chunks for the LLM.

This leads to:

*   **Better Context:** By keeping related sentences and semantic units together, the context provided to the LLM is much richer and more accurate.
*   **Improved Retrieval:** Semantic chunking leads to more precise vector embeddings, which in turn improves the accuracy of the search results.
*   **Overlap Preservation:** A small overlap between chunks is still maintained to ensure that context is not lost at chunk boundaries.

This updated method of splitting documents by their semantic structure (like paragraphs or sections) rather than by a fixed size results in a significant improvement in the quality of the retrieved context, and therefore, the quality of the LLM's answers.

### Two-Stage Retrieval: Re-ranking for Relevance

To further enhance the quality of the retrieved context, this project implements a two-stage retrieval process that includes a re-ranker.

1.  **Initial Retrieval:** The system first retrieves a larger set of documents from ChromaDB (e.g., the top 20) that are broadly relevant to the user's query. This initial step prioritizes recall, ensuring a wide net is cast.

2.  **Re-ranking:** The retrieved documents are then passed to a `CrossEncoder` model. This model scores each document's relevance to the specific query. Unlike the initial retrieval, which just measures similarity, the re-ranker performs a more sophisticated analysis of the relationship between the query and the document.

3.  **Final Selection:** The documents are then sorted by their new relevance scores, and only the top N (e.g., 5) are selected to be included in the context sent to the LLM. This final step prioritizes precision, ensuring that the context is as relevant and noise-free as possible.

This re-ranking step significantly improves the signal-to-noise ratio of the context, leading to more accurate and focused answers from the LLM.
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