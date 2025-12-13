#!/bin/bash

# --- Usage Function ---
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [DOCS_DIR]

Start the MCP knowledge base server with optional document ingestion.

OPTIONS:
    --re-ingest     Force full re-ingestion of all documents (deletes existing data)
    --skip-ingest   Skip document ingestion, just start the server (use if data already indexed)
    --help, -h      Show this help message

ARGUMENTS:
    DOCS_DIR        Path to documents directory (default: documents)

EXAMPLES:
    # Start server with existing data (recommended - fastest)
    $(basename "$0") --skip-ingest

    # First time setup OR when documents changed (deletes and rebuilds)
    $(basename "$0") --re-ingest

    # Start server with custom documents directory (first time)
    $(basename "$0") --re-ingest ./my_docs

    # Start server with custom directory (existing data)
    $(basename "$0") --skip-ingest ./my_docs

EOF
    exit 0
}

# --- Configuration ---
RE_INGEST_FLAG=""
SKIP_INGEST=false
DOCS_DIR="documents"

# --- Argument Handling ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help|-h) usage ;;
        --re-ingest) RE_INGEST_FLAG="--re-ingest"; shift ;;
        --skip-ingest) SKIP_INGEST=true; shift ;;
        *) DOCS_DIR="$1"; shift ;;
    esac
done

# --- Document Synchronization ---
if [[ "$SKIP_INGEST" == true ]]; then
    echo "Skipping document ingestion (using existing data)"
elif [[ -n "$RE_INGEST_FLAG" ]]; then
    echo "Running full re-ingestion (deleting existing data): $DOCS_DIR"
    docker-compose run --rm mcp_server python scripts/ingest.py $RE_INGEST_FLAG --docs_dir "$DOCS_DIR"
else
    echo "WARNING: Running full ingestion without deleting existing data (may create duplicates): $DOCS_DIR"
    echo "Use --skip-ingest if data already exists, or --re-ingest to delete and rebuild"
    docker-compose run --rm mcp_server python scripts/ingest.py --docs_dir "$DOCS_DIR"
fi

# --- Docker Compose ---
echo "Starting services..."
docker-compose up --build -d

