#!/bin/bash

# --- Usage Function ---
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [DOCS_DIR]

Start the MCP knowledge base server with optional document ingestion.

OPTIONS:
    --re-ingest     Force full re-ingestion of all documents (deletes existing data)
    --help, -h      Show this help message

ARGUMENTS:
    DOCS_DIR        Path to documents directory (default: documents)

EXAMPLES:
    # Start server with incremental sync (only new/modified files)
    $(basename "$0")

    # Start server with full re-ingestion
    $(basename "$0") --re-ingest

    # Start server with custom documents directory
    $(basename "$0") ./my_docs

    # Start server with full re-ingestion of custom directory
    $(basename "$0") --re-ingest ./my_docs

EOF
    exit 0
}

# --- Configuration ---
RE_INGEST_FLAG=""
DOCS_DIR="documents"

# --- Argument Handling ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help|-h) usage ;;
        --re-ingest) RE_INGEST_FLAG="--re-ingest"; shift ;;
        *) DOCS_DIR="$1"; shift ;;
    esac
done

# --- Document Synchronization ---
if [[ -n "$RE_INGEST_FLAG" ]]; then
    echo "Running full re-ingestion with arguments: $RE_INGEST_FLAG --docs_dir $DOCS_DIR"
    docker-compose run --rm mcp_server python scripts/ingest.py $RE_INGEST_FLAG --docs_dir "$DOCS_DIR"
else
    echo "Running incremental document synchronization with docs directory: $DOCS_DIR"
    docker-compose run --rm mcp_server python scripts/ingest.py --docs_dir "$DOCS_DIR"
fi

# --- Docker Compose ---
echo "Starting services..."
docker-compose up --build -d

