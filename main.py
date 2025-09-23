import argparse
import uvicorn
import logging

# Import shared components from the ingestion script
from ingest import synchronize_vectorstore, client, CONFIG
from knowledge_base_mcp import app

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main entry point to run ingestion and launch the server."""
    parser = argparse.ArgumentParser(
        description="Document Ingestion and MCP Server Launcher"
    )
    parser.add_argument(
        "--re-ingest",
        action="store_true",
        help="Force a full re-ingestion by deleting the existing collection.",
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=CONFIG["docs_dir"],
        help=f"The directory where the documents are stored (default: {CONFIG['docs_dir']}).",
    )
    args = parser.parse_args()

    if args.re_ingest:
        logging.warning(
            f"--- Performing a full reset of the collection '{CONFIG['collection_name']}' ---"
        )
        try:
            client.delete_collection(name=CONFIG["collection_name"])
            logging.info(
                f"Collection '{CONFIG['collection_name']}' deleted successfully."
            )
        except Exception as e:
            logging.error(f"Could not delete collection (it might not exist): {e}")

    # Always run synchronization to ensure the DB is up-to-date
    synchronize_vectorstore(args.docs_dir)

    logging.info("--- Launching MCP Server on http://0.0.0.0:8001 ---")
    app.run(transport="http", host="0.0.0.0", port=8000, log_level="DEBUG")


if __name__ == "__main__":
    main()
