from fastmcp import FastMCP
from pydantic import BaseModel
from typing import List

# Import shared components from the ingestion script
from ingest import vectorstore, embedding_function, CONFIG


# --- FastMCP App ---
app = FastMCP(
    name="MCP Knowledge Base Server",
)


# --- Tool Definition ---
class Document(BaseModel):
    """A document chunk retrieved from the knowledge base."""

    content: str
    metadata: dict


class KnowledgeBaseOutput(BaseModel):
    """The output of the knowledge base query tool, containing a list of relevant documents."""

    documents: List[Document]


@app.tool
def query_knowledge_base(query: str, top_k: int = 10) -> KnowledgeBaseOutput:
    """
    Searches a knowledge base for documents relevant to the query.
    This tool is useful for retrieving information from a local knowledge base.
    Use it to answer questions that require specific factual information or context from ingested documents.
    """
    print(f"Received query: {query}")

    # The vectorstore is already configured and loaded, so we can use it directly.
    # The retriever will handle embedding the query and finding similar documents.
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Find relevant documents
    docs = retriever.get_relevant_documents(query)

    # Process results into the required output format
    documents = [
        Document(content=doc.page_content, metadata=doc.metadata) for doc in docs
    ]

    return KnowledgeBaseOutput(documents=documents)

