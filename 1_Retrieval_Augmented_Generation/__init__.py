"""
Level 1: Simple Retrieval Augmented Generation (RAG)

This module implements a basic RAG pipeline for CKD management:
- Document embedding with EmbeddingGemma
- Vector storage with ChromaDB
- Retrieval and generation with MedGemma
"""

from .embeddings import EmbeddingGemmaWrapper
from .vectorstore import CKDVectorStore
from .retriever import CKDRetriever
from .chain import SimpleRAGChain

__all__ = [
    "EmbeddingGemmaWrapper",
    "CKDVectorStore",
    "CKDRetriever",
    "SimpleRAGChain",
]
