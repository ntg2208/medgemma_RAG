"""
Document preprocessing pipeline for CKD RAG System.

Handles PDF extraction, text cleaning, and semantic chunking
for NICE guidelines and KidneyCareUK documents.
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    MIN_CHUNK_SIZE,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""

    source: str
    title: str
    page_number: int
    section: Optional[str] = None
    # ckd_stages: list[int] = field(default_factory=list)
    document_type: str = "guideline"  # guideline, dietary, clinical


class DocumentPreprocessor:
    """
    Preprocessor for CKD-related PDF documents.

    Handles:
    - PDF text extraction with PyMuPDF
    - Text cleaning and normalization
    - Semantic chunking with overlap
    - Metadata extraction
    """

    # Patterns for identifying CKD stages in text
    CKD_STAGE_PATTERNS = [
        r"stage\s*([1-5])\s*(?:ckd|chronic kidney disease)?",
        r"ckd\s*(?:stage\s*)?([1-5])",
        r"g([1-5])\s*(?:a[1-3])?",  # G1-G5 nomenclature
        r"egfr\s*(?:<|>|<=|>=)?\s*(\d+)",  # eGFR values
    ]

    # Common headers/footers to remove
    NOISE_PATTERNS = [
        r"page\s+\d+\s+of\s+\d+",
        r"©\s*\d{4}.*?(?:nice|kidney)",
        r"www\.\S+",
        r"isbn\s*[\d\-]+",
        r"\[\d+\]",  # Reference numbers alone
    ]

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        """
        Initialize the preprocessor.

        Args:
            chunk_size: Target size for text chunks (in characters, roughly tokens*4)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size * 4  # Convert tokens to approximate chars
        self.chunk_overlap = chunk_overlap * 4
        self.min_chunk_size = min_chunk_size * 4

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Major section breaks
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence breaks
                ", ",  # Clause breaks
                " ",  # Word breaks
            ],
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> list[dict]:
        """
        Extract text from a PDF file with page-level granularity.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dicts with 'text', 'page_number', and 'metadata'
        """
        pages = []

        try:
            doc = fitz.open(pdf_path)
            title = doc.metadata.get("title", pdf_path.stem)

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")

                if text.strip():
                    pages.append(
                        {
                            "text": text,
                            "page_number": page_num,
                            "metadata": {
                                "source": pdf_path.name,
                                "title": title,
                                "page_number": page_num,
                                "total_pages": len(doc),
                            },
                        }
                    )

            doc.close()
            logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

        return pages

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
        # Remove common noise patterns
        for pattern in self.NOISE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix common OCR issues
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        text = text.replace(
            """, "'")
        text = text.replace(""",
            "'",
        )

        # Remove excessive punctuation
        text = re.sub(r"\.{3,}", "...", text)
        text = re.sub(r"-{3,}", "---", text)

        # Normalize line breaks for section detection
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def extract_ckd_stages(self, text: str) -> list[int]:
        """
        Extract CKD stages mentioned in text.

        Args:
            text: Text to analyze

        Returns:
            List of CKD stages (1-5) mentioned
        """
        stages = set()
        text_lower = text.lower()

        for pattern in self.CKD_STAGE_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    stage = int(match)
                    if 1 <= stage <= 5:
                        stages.add(stage)
                except ValueError:
                    # Handle eGFR values
                    try:
                        egfr = int(match)
                        if egfr >= 90:
                            stages.add(1)
                        elif egfr >= 60:
                            stages.add(2)
                        elif egfr >= 30:
                            stages.add(3)
                        elif egfr >= 15:
                            stages.add(4)
                        else:
                            stages.add(5)
                    except ValueError:
                        continue

        return sorted(stages)

    def detect_section(self, text: str) -> Optional[str]:
        """
        Detect the section/chapter from text content.

        Args:
            text: Text chunk to analyze

        Returns:
            Section name if detected, None otherwise
        """
        # Common section patterns in medical guidelines
        section_patterns = [
            r"^(\d+\.?\d*\.?\s+[A-Z][^.]+)",  # Numbered sections
            r"^(Chapter\s+\d+[:\s]+[^.]+)",
            r"^(Section\s+\d+[:\s]+[^.]+)",
            r"^(Appendix\s+[A-Z\d]+[:\s]+[^.]+)",
        ]

        lines = text.split("\n")[:5]  # Check first few lines

        for line in lines:
            line = line.strip()
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    return match.group(1)[:100]  # Limit length

        return None

    def classify_document_type(self, filename: str, text: str) -> str:
        """
        Classify the document type based on filename and content.

        Args:
            filename: Name of the source file
            text: Sample text from document

        Returns:
            Document type classification
        """
        filename_lower = filename.lower()
        text_lower = text.lower()

        if "diet" in filename_lower or "nutrition" in filename_lower:
            return "dietary"
        elif "nice" in filename_lower or "guideline" in filename_lower:
            return "guideline"
        elif any(
            word in text_lower
            for word in ["potassium", "phosphorus", "sodium", "protein intake"]
        ):
            return "dietary"
        elif any(
            word in text_lower
            for word in ["recommendation", "should offer", "should consider"]
        ):
            return "guideline"
        else:
            return "clinical"

    def process_pdf(self, pdf_path: Path) -> list[Document]:
        """
        Process a single PDF file into LangChain Documents.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of LangChain Document objects with metadata
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing {pdf_path.name}...")

        # Extract raw pages
        pages = self.extract_text_from_pdf(pdf_path)

        if not pages:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return []

        # Combine and clean text
        full_text = "\n\n".join(p["text"] for p in pages)
        cleaned_text = self.clean_text(full_text)

        # Determine document type
        doc_type = self.classify_document_type(pdf_path.name, cleaned_text[:2000])

        # Create chunks
        chunks = self.text_splitter.split_text(cleaned_text)

        # Filter out small chunks
        chunks = [c for c in chunks if len(c) >= self.min_chunk_size]

        # Create LangChain Documents with metadata
        documents = []

        for i, chunk in enumerate(chunks):
            # Extract metadata for this chunk
            ckd_stages = self.extract_ckd_stages(chunk)
            section = self.detect_section(chunk)

            # Estimate page number based on position
            char_position = cleaned_text.find(chunk[:100])
            estimated_page = 1
            if char_position >= 0 and pages:
                chars_per_page = len(cleaned_text) / len(pages)
                estimated_page = min(
                    int(char_position / chars_per_page) + 1, len(pages)
                )

            metadata = {
                "source": pdf_path.name,
                "title": pages[0]["metadata"]["title"],
                "chunk_id": i,
                "total_chunks": len(chunks),
                "page_number": estimated_page,
                "section": section,
                "ckd_stages": ckd_stages,
                "document_type": doc_type,
            }

            documents.append(Document(page_content=chunk, metadata=metadata))

        logger.info(f"Created {len(documents)} chunks from {pdf_path.name}")
        return documents

    def process_directory(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> list[Document]:
        """
        Process all PDF files in a directory.

        Args:
            input_dir: Directory containing PDFs (default: DOCUMENTS_DIR)
            output_dir: Directory for processed output (default: PROCESSED_DIR)

        Returns:
            Combined list of all Document objects
        """
        input_dir = Path(input_dir) if input_dir else DOCUMENTS_DIR
        output_dir = Path(output_dir) if output_dir else PROCESSED_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        all_documents = []
        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            try:
                documents = self.process_pdf(pdf_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue

        logger.info(f"Total documents created: {len(all_documents)}")
        return all_documents

    def get_document_stats(self, documents: list[Document]) -> dict:
        """
        Get statistics about processed documents.

        Args:
            documents: List of processed Document objects

        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_documents": 0}

        sources = set(d.metadata.get("source", "unknown") for d in documents)
        doc_types = {}
        ckd_coverage = {i: 0 for i in range(1, 6)}

        for doc in documents:
            doc_type = doc.metadata.get("document_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            for stage in doc.metadata.get("ckd_stages", []):
                ckd_coverage[stage] += 1

        total_chars = sum(len(d.page_content) for d in documents)

        return {
            "total_documents": len(documents),
            "unique_sources": len(sources),
            "sources": list(sources),
            "document_types": doc_types,
            "ckd_stage_coverage": ckd_coverage,
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(documents) if documents else 0,
        }


# Convenience function for quick processing
def preprocess_documents(
    input_dir: Optional[Path] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Convenience function to preprocess all documents in a directory.

    Args:
        input_dir: Directory containing PDF files
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of processed Document objects
    """
    preprocessor = DocumentPreprocessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return preprocessor.process_directory(input_dir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    preprocessor = DocumentPreprocessor()
    documents = preprocessor.process_directory()

    if documents:
        stats = preprocessor.get_document_stats(documents)
        print("\nDocument Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
