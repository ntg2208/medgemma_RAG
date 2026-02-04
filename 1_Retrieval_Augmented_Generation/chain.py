"""
Simple RAG chain for the CKD RAG System.

Implements a basic retrieval-augmented generation pipeline
using MedGemma for response generation.
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MEDGEMMA_MODEL_ID,
    HF_TOKEN,
    GENERATION_CONFIG,
    RAG_SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG chain."""
    answer: str
    source_documents: list[Document]
    query: str


class MedGemmaLLM:
    """
    Wrapper for MedGemma language model.

    Handles model loading with optional quantization and
    provides a simple generation interface.
    """

    def __init__(
        self,
        model_id: str = MEDGEMMA_MODEL_ID,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = GENERATION_CONFIG["max_new_tokens"],
        temperature: float = GENERATION_CONFIG["temperature"],
    ):
        """
        Initialize MedGemma.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run on (auto-detected if None)
            load_in_4bit: Whether to use 4-bit quantization
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                load_in_4bit = False  # bitsandbytes not supported on MPS
            else:
                self.device = "cpu"
                load_in_4bit = False
        else:
            self.device = device

        logger.info(f"Loading MedGemma on {self.device} (4-bit: {load_in_4bit})...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN if HF_TOKEN else None,
            trust_remote_code=True,
        )

        # Load model with optional quantization
        model_kwargs = {
            "token": HF_TOKEN if HF_TOKEN else None,
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("MedGemma loaded successfully")

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=GENERATION_CONFIG.get("top_p", 0.9),
                repetition_penalty=GENERATION_CONFIG.get("repetition_penalty", 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def as_langchain_llm(self) -> HuggingFacePipeline:
        """
        Get a LangChain-compatible LLM interface.

        Returns:
            HuggingFacePipeline instance
        """
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            top_p=GENERATION_CONFIG.get("top_p", 0.9),
            repetition_penalty=GENERATION_CONFIG.get("repetition_penalty", 1.1),
            return_full_text=False,
        )

        return HuggingFacePipeline(pipeline=pipe)


class SimpleRAGChain:
    """
    Simple RAG chain for CKD question answering.

    Implements the basic pattern:
    Query -> Retrieve -> Augment -> Generate

    Example:
        >>> chain = SimpleRAGChain(retriever, llm)
        >>> response = chain.invoke("What are the dietary restrictions for CKD stage 3?")
        >>> print(response.answer)
    """

    def __init__(
        self,
        retriever: Any,
        llm: Optional[MedGemmaLLM] = None,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: Document retriever
            llm: MedGemma LLM instance (created if None)
            system_prompt: System prompt for the LLM
            prompt_template: Template for combining context and question
        """
        self.retriever = retriever
        self.llm = llm or MedGemmaLLM()
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

        # Build the LangChain pipeline
        self._chain = self._build_chain()

    def _format_documents(self, docs: list[Document]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted context string with source citations
        """
        formatted_parts = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_number", "?")
            section = doc.metadata.get("section", "")

            citation = f"[{i}] Source: {source}, Page {page}"
            if section:
                citation += f", Section: {section}"

            formatted_parts.append(f"{citation}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_parts)

    def _build_chain(self):
        """Build the LangChain LCEL chain."""
        # Create the prompt template
        prompt = PromptTemplate(
            template=f"{self.system_prompt}\n\n{self.prompt_template}",
            input_variables=["context", "question"],
        )

        # Get LangChain LLM
        langchain_llm = self.llm.as_langchain_llm()

        # Build the chain using LCEL
        chain = (
            {
                "context": self.retriever | RunnableLambda(self._format_documents),
                "question": RunnablePassthrough(),
            }
            | prompt
            | langchain_llm
            | StrOutputParser()
        )

        return chain

    def invoke(self, query: str) -> RAGResponse:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User question

        Returns:
            RAGResponse with answer and source documents
        """
        # Retrieve documents separately to include in response
        docs = self.retriever.invoke(query)

        # Generate answer
        answer = self._chain.invoke(query)

        return RAGResponse(
            answer=answer,
            source_documents=docs,
            query=query,
        )

    def stream(self, query: str):
        """
        Stream the response generation.

        Args:
            query: User question

        Yields:
            Response chunks
        """
        for chunk in self._chain.stream(query):
            yield chunk

    def batch(self, queries: list[str]) -> list[RAGResponse]:
        """
        Process multiple queries.

        Args:
            queries: List of user questions

        Returns:
            List of RAGResponse objects
        """
        return [self.invoke(q) for q in queries]


def create_rag_chain(
    vectorstore: Any,
    ckd_stage: Optional[int] = None,
    llm: Optional[MedGemmaLLM] = None,
) -> SimpleRAGChain:
    """
    Factory function to create a complete RAG chain.

    Args:
        vectorstore: CKDVectorStore instance
        ckd_stage: Optional CKD stage for filtering
        llm: Optional pre-loaded LLM

    Returns:
        Configured SimpleRAGChain
    """
    from .retriever import CKDRetriever

    retriever = CKDRetriever(
        vectorstore=vectorstore,
        ckd_stage=ckd_stage,
    )

    return SimpleRAGChain(retriever=retriever, llm=llm)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Testing SimpleRAGChain...")
    print("Note: This requires a GPU and HuggingFace access to MedGemma")

    # This would be the full test with actual components:
    # from embeddings import EmbeddingGemmaWrapper
    # from vectorstore import CKDVectorStore
    #
    # embeddings = EmbeddingGemmaWrapper()
    # store = CKDVectorStore(embeddings)
    # chain = create_rag_chain(store)
    # response = chain.invoke("What are the dietary restrictions for CKD stage 3?")
    # print(response.answer)
