from fastmcp import FastMCP
from pydantic import BaseModel
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any

# --- Configuration ---
COLLECTION_NAME = "knowledge_base"

# --- FastMCP App ---
app = FastMCP(name="MCP Knowledge Base Server")

# --- ChromaDB Client and Vector Store (loaded on startup) ---
client = chromadb.HttpClient(host="chroma", port=8000)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


# --- Tool Definition ---
class Document(BaseModel):
    content: str
    metadata: dict


class KnowledgeBaseOutput(BaseModel):
    documents: List[Document]


@app.tool
def query_knowledge_base(query: str, top_k: int = 10) -> KnowledgeBaseOutput:
    """
    Searches a knowledge base for documents relevant to the query.
    This tool is useful for retrieving information from a local knowledge base.
    Use it to answer questions that require specific factual information or context from ingested documents.
    """
    print(f"Received query: {query}")

    # 1. Get query embedding
    query_embedding = embedding_function.embed_query(query)

    # 2. Query ChromaDB
    results = vectorstore._collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    # 3. Process results and return
    documents = []
    if results and results.get("documents"):
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i in range(len(docs)):
            content = docs[i]
            metadata = metadatas[i]
            if content is not None:
                documents.append({"content": content, "metadata": metadata})

    return KnowledgeBaseOutput(documents=documents)
