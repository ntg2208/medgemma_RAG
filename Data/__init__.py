"""
Data processing module for the CKD RAG System.

Handles:
- PDF document loading and extraction
- Text preprocessing and cleaning
- Semantic chunking
- Metadata extraction
"""

from .preprocessing import DocumentPreprocessor

__all__ = [
    "DocumentPreprocessor",
]
