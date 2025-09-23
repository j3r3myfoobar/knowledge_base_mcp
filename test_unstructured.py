from langchain_unstructured import UnstructuredLoader
import os

# Define the path to the problematic PDF
pdf_path = "/Users/jeremy/Developer/AI/mcp_server/documents/Prompt Engineering for Generative AI.pdf"

print(f"Attempting to load PDF: {pdf_path}")

try:
    loader = UnstructuredLoader(pdf_path, strategy="auto", mode="elements")
    documents = loader.load()

    if documents:
        print(f"Successfully extracted {len(documents)} documents.")
        for i, doc in enumerate(documents):
            print(f"--- Document {i+1} ---")
            print(f"Type: {type(doc)}")
            print(f"Metadata: {doc.metadata}")
            print(f"Content (first 500 chars): {doc.page_content[:500]}...")
    else:
        print("UnstructuredLoader returned no documents.")
except Exception as e:
    print(f"An error occurred: {e}")