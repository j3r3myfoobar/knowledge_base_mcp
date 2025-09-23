import argparse
from ingest import synchronize_vectorstore, client as chroma_client, COLLECTION_NAME
from knowledge_base_mcp import app

def main():
    parser = argparse.ArgumentParser(description="Document Ingestion and MCP Server Launcher")
    parser.add_argument(
        "--re-ingest",
        action="store_true",
        help="Force a full re-ingestion by deleting the existing collection.",
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="documents",
        help="The directory where the documents are stored.",
    )
    args = parser.parse_args()

    if args.re_ingest:
        print("--- Performing a full reset of the collection ---")
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted.")
        except Exception as e:
            print(f"Could not delete collection (it might not exist): {e}")

    # Always run synchronization
    synchronize_vectorstore(args.docs_dir)

    print("--- Launching MCP Server ---")
    app.run(transport="http", host="0.0.0.0", port=8000, log_level="DEBUG")

if __name__ == "__main__":
    main()