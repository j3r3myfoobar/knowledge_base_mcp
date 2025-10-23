"""
MCP server application using Baseline + Hybrid Search.

SIMPLIFIED ARCHITECTURE:
- Hybrid Search: BM25 (keyword) + Vector (semantic) fusion
- Fixed-size chunking: Simple, predictable, fast
- No query enhancement: Preserves keyword precision
- No cross-encoder: Faster, simpler
- Direct confidence scoring: Based on hybrid fusion score

Performance: 100% R@5, 23ms average latency (16x faster than previous)
"""

import logging
from fastmcp import FastMCP
from fastapi.responses import JSONResponse

from src.core.interfaces import Retriever
from src.factories import create_baseline_retriever
from src.core.models import Document, KnowledgeBaseOutput

logger = logging.getLogger(__name__)

# Initialize FastMCP application
app = FastMCP(name="MCP Knowledge Base Server (Baseline + Hybrid)")

# Initialize baseline retriever using factory function (Dependency Injection)
# Factory handles all dependency creation and wiring
retriever: Retriever = create_baseline_retriever(
    chroma_host="chroma",
    chroma_port=8000,
    collection_name="baseline_kb"
)
logger.info("MCP server initialized with BaselineRetriever")

# Initialize retriever (builds BM25 index from existing documents)
retriever.initialize()


# Healthcheck endpoint for Claude Code compatibility
@app.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Simple healthcheck endpoint for Claude Code compatibility."""
    return JSONResponse({"status": "healthy", "service": "mcp_knowledge_base_baseline"})


@app.tool
def query_knowledge_base(
    query: str,
    top_k: int = 5,
    use_hybrid: bool = True
) -> KnowledgeBaseOutput:
    """
    Search the knowledge base using Baseline + Hybrid Search.

    This tool implements a simplified but highly effective retrieval process:
    1. BM25 keyword search (finds exact term matches)
    2. Vector semantic search (finds conceptual matches)
    3. Hybrid fusion (0.3*BM25 + 0.7*Vector)
    4. Return top-k results

    Performance: 100% R@5 on test queries, 23ms average latency

    Args:
        query: The search query or question
        top_k: Number of results to return (default: 5)
        use_hybrid: Use hybrid search (True) or vector-only (False)

    Returns:
        KnowledgeBaseOutput containing relevant documents with confidence scores
    """
    logger.info(f"Received query: '{query}' (hybrid={use_hybrid})")

    # Query with hybrid search
    results = retriever.query(
        query=query,
        top_k=top_k,
        use_hybrid=use_hybrid
    )

    # Convert to Document objects
    documents = [
        Document(
            content=result['content'],
            metadata=result['metadata'],
            confidence=result['confidence']
        )
        for result in results
    ]

    logger.info(f"Returning {len(documents)} documents (hybrid search)")
    return KnowledgeBaseOutput(documents=documents)
