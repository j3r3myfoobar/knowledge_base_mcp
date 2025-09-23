from fastmcp import FastMCP
from pydantic import BaseModel
from typing import List

# Import shared components from the ingestion script
import logging
from ingest import vectorstore, embedding_function, CONFIG

logger = logging.getLogger(__name__)

from sentence_transformers import CrossEncoder

def get_cross_encoder():
    try:
        # Load a pre-trained cross-encoder model for re-ranking
        # This model is typically different from the embedding model used for initial retrieval
        # A good choice for re-ranking is 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        encoder = CrossEncoder(cross_encoder_model_name)
        logger.info(f"Successfully loaded CrossEncoder model: {cross_encoder_model_name}")
        return encoder
    except Exception as e:
        logger.error(f"Failed to load CrossEncoder model: {e}")
        return None


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
def query_knowledge_base(
    query: str, top_k: int = 20, top_n_rerank: int = 5
) -> KnowledgeBaseOutput:
    """
    Searches a knowledge base for documents relevant to the query, then re-ranks them for relevance.
    This tool is useful for retrieving information from a local knowledge base.
    Use it to answer questions that require specific factual information or context from ingested documents.
    """
    logger.info(f"Received query: {query}")

    encoder = get_cross_encoder()

    if not encoder:
        logger.error("CrossEncoder model is not available. Skipping re-ranking.")
        # Fallback to simple retrieval if the model failed to load
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_n_rerank})
        docs = retriever.invoke(query)
        documents = [
            Document(content=doc.page_content, metadata=doc.metadata) for doc in docs
        ]
        return KnowledgeBaseOutput(documents=documents)

    # 1. Initial Retrieval from ChromaDB
    logger.info(f"Retrieving top {top_k} documents from ChromaDB...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(docs)} documents.")

    # 2. Re-ranking with Cross-Encoder
    logger.info("Re-ranking documents with CrossEncoder...")
    doc_contents = [doc.page_content for doc in docs]
    query_doc_pairs = [[query, content] for content in doc_contents]

    scores = encoder.predict(query_doc_pairs)
    logger.info(f"Calculated {len(scores)} scores.")

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 3. Select Top N Re-ranked Documents
    top_n_docs = [doc for score, doc in scored_docs[:top_n_rerank]]
    logger.info(f"Selected top {len(top_n_docs)} re-ranked documents.")

    documents = [
        Document(content=doc.page_content, metadata=doc.metadata) for doc in top_n_docs
    ]

    return KnowledgeBaseOutput(documents=documents)

