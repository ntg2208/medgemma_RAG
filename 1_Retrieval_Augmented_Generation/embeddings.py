"""
EmbeddingGemma wrapper for the CKD RAG System.

Provides a LangChain-compatible embedding interface using
Google's EmbeddingGemma model with Matryoshka Representation Learning (MRL).
"""

import logging
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIMENSION,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)


class EmbeddingGemmaWrapper(Embeddings):
    """
    LangChain-compatible wrapper for EmbeddingGemma.

    Supports Matryoshka Representation Learning (MRL) for flexible
    embedding dimensions: 128, 256, 512, 768.

    Example:
        >>> embeddings = EmbeddingGemmaWrapper(dimension=768)
        >>> vector = embeddings.embed_query("What is CKD stage 3?")
        >>> vectors = embeddings.embed_documents(["Doc 1", "Doc 2"])
    """

    SUPPORTED_DIMENSIONS = [128, 256, 512, 768]

    def __init__(
        self,
        model_id: str = EMBEDDING_MODEL_ID,
        dimension: int = EMBEDDING_DIMENSION,
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize the EmbeddingGemma wrapper.

        Args:
            model_id: HuggingFace model identifier
            dimension: Embedding dimension (128, 256, 512, or 768)
            device: Device to run model on (auto-detected if None)
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for embedding multiple documents
        """
        if dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"Dimension {dimension} not supported. "
                f"Choose from: {self.SUPPORTED_DIMENSIONS}"
            )

        self.model_id = model_id
        self.dimension = dimension
        self.normalize = normalize
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading EmbeddingGemma on {self.device}...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN if HF_TOKEN else None,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_id,
            token=HF_TOKEN if HF_TOKEN else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)

        self.model.eval()
        logger.info(f"EmbeddingGemma loaded successfully (dim={dimension})")

    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mean pooling to get sentence embeddings.

        Args:
            model_output: Model's hidden states
            attention_mask: Attention mask from tokenizer

        Returns:
            Pooled embeddings tensor
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def _truncate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Truncate embeddings to the desired dimension (MRL).

        Args:
            embeddings: Full-dimension embeddings

        Returns:
            Truncated embeddings
        """
        return embeddings[:, :self.dimension]

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize embeddings.

        Args:
            embeddings: Input embeddings

        Returns:
            Normalized embeddings
        """
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            output = self.model(**encoded)

        # Pool and process
        embeddings = self._mean_pooling(output, encoded["attention_mask"])
        embeddings = self._truncate_embeddings(embeddings)

        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)

        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self._embed_batch([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            if len(texts) > self.batch_size:
                logger.debug(
                    f"Embedded {min(i + self.batch_size, len(texts))}/{len(texts)} documents"
                )

        return all_embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1 if normalized)
        """
        emb1 = torch.tensor(self.embed_query(text1))
        emb2 = torch.tensor(self.embed_query(text2))

        return torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()


class CachedEmbeddingGemma(EmbeddingGemmaWrapper):
    """
    EmbeddingGemma wrapper with simple in-memory caching.

    Useful for development and testing to avoid re-embedding
    the same texts repeatedly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: dict[str, list[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def embed_query(self, text: str) -> list[float]:
        """Embed query with caching."""
        if text in self._cache:
            self._cache_hits += 1
            return self._cache[text]

        self._cache_misses += 1
        embedding = super().embed_query(text)
        self._cache[text] = embedding
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with caching."""
        # Separate cached and uncached texts
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text not in self._cache:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = super().embed_documents(uncached_texts)
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._cache[text] = embedding

        # Build result in original order
        return [self._cache[text] for text in texts]

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


def get_embeddings(
    dimension: int = EMBEDDING_DIMENSION,
    use_cache: bool = False,
) -> Embeddings:
    """
    Factory function to get an embedding instance.

    Args:
        dimension: Embedding dimension to use
        use_cache: Whether to use cached embeddings

    Returns:
        EmbeddingGemma instance
    """
    cls = CachedEmbeddingGemma if use_cache else EmbeddingGemmaWrapper
    return cls(dimension=dimension)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("Testing EmbeddingGemma wrapper...")

    embeddings = EmbeddingGemmaWrapper(dimension=768)

    # Test single query
    query = "What are the dietary restrictions for CKD stage 3?"
    query_embedding = embeddings.embed_query(query)
    print(f"\nQuery embedding shape: {len(query_embedding)}")

    # Test multiple documents
    docs = [
        "Patients with CKD stage 3 should limit potassium intake.",
        "NICE guidelines recommend ACE inhibitors for CKD with proteinuria.",
        "Regular exercise is beneficial for kidney health.",
    ]
    doc_embeddings = embeddings.embed_documents(docs)
    print(f"Document embeddings: {len(doc_embeddings)} x {len(doc_embeddings[0])}")

    # Test similarity
    similarity = embeddings.similarity(
        "CKD dietary guidelines",
        "Kidney disease nutrition recommendations"
    )
    print(f"\nSimilarity score: {similarity:.4f}")
