#!/bin/bash

# --- Configuration ---
RE_INGEST_ARG=""
DOCS_DIR_ARG="--docs_dir documents"

# --- Argument Handling ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --re-ingest) RE_INGEST_ARG="--re-ingest"; shift ;;
        *) DOCS_DIR_ARG="--docs_dir $1"; shift ;;
    esac
done

# --- Docker Compose ---
echo "Starting services with arguments: $RE_INGEST_ARG $DOCS_DIR_ARG"
docker-compose run --rm --service-ports mcp_server python main.py $RE_INGEST_ARG $DOCS_DIR_ARG
