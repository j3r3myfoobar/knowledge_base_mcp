"""
Pydantic models for MCP server API.

Defines the data structures used by the MCP server for input/output validation.
"""

from typing import List
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    A document chunk retrieved from the knowledge base.

    Attributes:
        content: The text content of the document chunk
        metadata: Dictionary containing source file, modification time, etc.
        confidence: Confidence score (0.0-1.0) indicating relevance to query
    """

    content: str = Field(
        ...,
        description="The text content of the retrieved document chunk"
    )
    metadata: dict = Field(
        ...,
        description="Metadata including source file, last_modified timestamp, etc."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (low) to 1.0 (high) indicating relevance"
    )


class KnowledgeBaseOutput(BaseModel):
    """
    The output of the knowledge base query tool.

    Contains a list of relevant documents retrieved and ranked by the system.

    Attributes:
        documents: List of Document objects sorted by relevance (highest first)
    """

    documents: List[Document] = Field(
        ...,
        description="List of relevant document chunks, sorted by relevance"
    )
