"""
Gradio UI for the CKD Management RAG System.

Multi-tab interface providing access to:
- Simple RAG (Level 1)
- Agentic RAG (Level 2)
- Multi-Agent System (Level 3)
- About/Documentation
"""

import logging
import sys
import importlib.util
from pathlib import Path
from typing import Tuple
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (handles numeric prefixes)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Global state for loaded components
_components = {
    "initialized": False,
    "embeddings": None,
    "vectorstore": None,
    "llm": None,
    "simple_rag": None,
    "agentic_rag": None,
    "orchestrator": None,
    "pii_handler": None,
}


def initialize_components():
    """Initialize all RAG components. Called on first query."""
    if _components["initialized"]:
        return True

    try:
        logger.info("Initializing components...")

        # Import config and factory functions
        from config import EMBEDDING_DIMENSION, get_llm, get_embeddings

        # Import modules with numeric prefixes using importlib
        embeddings_mod = import_from_path(
            "rag_embeddings",
            PROJECT_ROOT / "1_Retrieval_Augmented_Generation" / "embeddings.py"
        )
        vectorstore_mod = import_from_path(
            "rag_vectorstore",
            PROJECT_ROOT / "1_Retrieval_Augmented_Generation" / "vectorstore.py"
        )
        chain_mod = import_from_path(
            "rag_chain",
            PROJECT_ROOT / "1_Retrieval_Augmented_Generation" / "chain.py"
        )
        retriever_mod = import_from_path(
            "rag_retriever",
            PROJECT_ROOT / "1_Retrieval_Augmented_Generation" / "retriever.py"
        )
        pii_mod = import_from_path(
            "agentic_pii",
            PROJECT_ROOT / "2_Agentic_RAG" / "pii_handler.py"
        )
        graph_mod = import_from_path(
            "agentic_graph",
            PROJECT_ROOT / "2_Agentic_RAG" / "graph.py"
        )
        orchestrator_mod = import_from_path(
            "multi_orchestrator",
            PROJECT_ROOT / "3_MultiAgent_RAG" / "orchestrator.py"
        )

        # Extract classes
        EmbeddingGemmaWrapper = embeddings_mod.EmbeddingGemmaWrapper
        CKDVectorStore = vectorstore_mod.CKDVectorStore
        MedGemmaLLM = chain_mod.MedGemmaLLM
        SimpleRAGChain = chain_mod.SimpleRAGChain
        CKDRetriever = retriever_mod.CKDRetriever
        PIIHandler = pii_mod.PIIHandler
        AgenticRAGGraph = graph_mod.AgenticRAGGraph
        MultiAgentOrchestrator = orchestrator_mod.MultiAgentOrchestrator

        # Initialize embeddings (local or remote based on config)
        logger.info("Loading embeddings model...")
        _components["embeddings"] = get_embeddings()

        # Initialize vector store
        logger.info("Initializing vector store...")
        _components["vectorstore"] = CKDVectorStore(_components["embeddings"])

        # Initialize LLM (local or remote based on config)
        logger.info("Loading MedGemma LLM...")
        _components["llm"] = get_llm()

        # Initialize PII handler
        logger.info("Initializing PII handler...")
        _components["pii_handler"] = PIIHandler()

        # Initialize retriever
        retriever = CKDRetriever(vectorstore=_components["vectorstore"])

        # Initialize Simple RAG
        _components["simple_rag"] = SimpleRAGChain(
            retriever=retriever,
            llm=_components["llm"]
        )

        # Initialize Agentic RAG
        _components["agentic_rag"] = AgenticRAGGraph(
            pii_handler=_components["pii_handler"],
            retriever=retriever,
            llm=_components["llm"]
        )

        # Initialize Multi-Agent Orchestrator
        _components["orchestrator"] = MultiAgentOrchestrator(
            retriever=retriever,
            llm=_components["llm"],
            pii_handler=_components["pii_handler"]
        )

        _components["initialized"] = True
        logger.info("All components initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False


def check_initialization() -> str:
    """Check if components are initialized and return status message."""
    if not _components["initialized"]:
        return (
            "**System Status: Not Initialized**\n\n"
            "Components will be loaded on first query. This may take a few minutes.\n\n"
            "Required:\n"
            "- HuggingFace token with MedGemma access\n"
            "- GPU recommended for faster inference\n"
            "- PDF documents in Data/documents/"
        )
    return "**System Status: Ready**"


# =============================================================================
# Tab 1: Simple RAG
# =============================================================================

def simple_rag_query(
    query: str,
    ckd_stage: int,
    show_sources: bool,
    history: list,
) -> Tuple[list, str]:
    """Process a query through the Simple RAG system."""
    if not query.strip():
        return history, ""

    # Initialize if needed
    if not _components["initialized"]:
        history.append((query, "Initializing system... Please wait."))
        if not initialize_components():
            history[-1] = (query, "Failed to initialize. Check logs for details.")
            return history, ""
        history[-1] = (query, "System initialized! Processing your query...")

    try:
        # Process query
        response = _components["simple_rag"].invoke(query)

        # Format response
        answer = response.answer

        if show_sources and response.source_documents:
            answer += "\n\n---\n**Sources:**\n"
            for i, doc in enumerate(response.source_documents, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page_number", "?")
                answer += f"{i}. {source}, Page {page}\n"

        history.append((query, answer))

    except Exception as e:
        logger.error(f"Simple RAG error: {e}")
        history.append((query, f"Error: {str(e)}"))

    return history, ""


# =============================================================================
# Tab 2: Agentic RAG
# =============================================================================

def agentic_rag_query(
    query: str,
    ckd_stage: int,
    history: list,
) -> Tuple[list, str, str, str]:
    """Process a query through the Agentic RAG system."""
    if not query.strip():
        return history, "", "No query", "N/A"

    # Initialize if needed
    if not _components["initialized"]:
        history.append((query, "Initializing system... Please wait."))
        if not initialize_components():
            history[-1] = (query, "Failed to initialize.")
            return history, "", "Error", "N/A"

    try:
        # Process query
        result = _components["agentic_rag"].invoke(
            query=query,
            ckd_stage=ckd_stage if ckd_stage > 0 else None
        )

        answer = result.get("final_response", "No response generated.")

        # PII info
        pii_info = "No PII detected"
        if result.get("pii_detected"):
            pii_info = f"PII detected and redacted: {len(result.get('pii_map', {}))} items"

        # Evaluation scores
        scores = result.get("evaluation_scores", {})
        if scores:
            eval_info = "\n".join(f"- {k}: {v:.2f}" for k, v in scores.items())
        else:
            eval_info = "Evaluation not available"

        # Processing steps
        steps = result.get("processing_steps", [])
        steps_info = " → ".join(steps) if steps else "N/A"

        history.append((query, answer))
        return history, "", pii_info, eval_info

    except Exception as e:
        logger.error(f"Agentic RAG error: {e}")
        history.append((query, f"Error: {str(e)}"))
        return history, "", "Error", "N/A"


# =============================================================================
# Tab 3: Multi-Agent
# =============================================================================

def multi_agent_query(
    query: str,
    ckd_stage: int,
    weight_kg: float,
    history: list,
) -> Tuple[list, str, str, str]:
    """Process a query through the Multi-Agent system."""
    if not query.strip():
        return history, "", "No query", ""

    # Initialize if needed
    if not _components["initialized"]:
        history.append((query, "Initializing system... Please wait."))
        if not initialize_components():
            history[-1] = (query, "Failed to initialize.")
            return history, "", "Error", ""

    try:
        # Process query
        response = _components["orchestrator"].process(
            query=query,
            ckd_stage=ckd_stage if ckd_stage > 0 else None,
            weight_kg=weight_kg if weight_kg > 0 else None
        )

        # Format agent info
        agents_info = f"**Agents Used:** {', '.join(response.agents_used)}"

        # Routing info
        routing = response.routing_decision
        routing_info = (
            f"**Primary:** {routing.primary_agent.value}\n"
            f"**Confidence:** {routing.confidence:.0%}\n"
            f"**Reasoning:** {routing.reasoning}"
        )

        history.append((query, response.answer))
        return history, "", agents_info, routing_info

    except Exception as e:
        logger.error(f"Multi-Agent error: {e}")
        history.append((query, f"Error: {str(e)}"))
        return history, "", "Error", ""


# =============================================================================
# Build Gradio Interface
# =============================================================================

def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(
        title="CKD Management Assistant",
        theme=gr.themes.Soft(),
        css="""
        .disclaimer {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        """
    ) as app:

        gr.Markdown(
            """
            # CKD Management Assistant

            An AI-powered assistant for Chronic Kidney Disease management based on
            **NICE guidelines** and **KidneyCareUK** resources.

            <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> This tool provides educational information only
            and should not replace professional medical advice. Always consult your healthcare
            provider for personalized recommendations.
            </div>
            """
        )

        with gr.Tabs():
            # =================================================================
            # Tab 1: Simple RAG
            # =================================================================
            with gr.TabItem("Simple RAG", id="simple"):
                gr.Markdown(
                    """
                    ### Level 1: Simple RAG
                    Basic question answering with document retrieval and source citations.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        simple_chatbot = gr.Chatbot(
                            label="Chat",
                            height=400,
                            show_copy_button=True,
                        )
                        simple_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the dietary restrictions for CKD stage 3?",
                            lines=2,
                        )
                        simple_submit = gr.Button("Ask", variant="primary")

                    with gr.Column(scale=1):
                        simple_ckd_stage = gr.Slider(
                            minimum=0,
                            maximum=5,
                            step=1,
                            value=0,
                            label="CKD Stage (0 = not specified)",
                        )
                        simple_show_sources = gr.Checkbox(
                            label="Show Sources",
                            value=True,
                        )
                        simple_clear = gr.Button("Clear Chat")

                simple_submit.click(
                    simple_rag_query,
                    inputs=[simple_input, simple_ckd_stage, simple_show_sources, simple_chatbot],
                    outputs=[simple_chatbot, simple_input],
                )
                simple_input.submit(
                    simple_rag_query,
                    inputs=[simple_input, simple_ckd_stage, simple_show_sources, simple_chatbot],
                    outputs=[simple_chatbot, simple_input],
                )
                simple_clear.click(lambda: [], outputs=[simple_chatbot])

            # =================================================================
            # Tab 2: Agentic RAG
            # =================================================================
            with gr.TabItem("Agentic RAG", id="agentic"):
                gr.Markdown(
                    """
                    ### Level 2: Agentic RAG
                    Advanced workflow with PII detection, query routing, and response evaluation.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        agentic_chatbot = gr.Chatbot(
                            label="Chat",
                            height=400,
                            show_copy_button=True,
                        )
                        agentic_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., My name is John, what medications should I avoid?",
                            lines=2,
                        )
                        agentic_submit = gr.Button("Ask", variant="primary")

                    with gr.Column(scale=1):
                        agentic_ckd_stage = gr.Slider(
                            minimum=0,
                            maximum=5,
                            step=1,
                            value=0,
                            label="CKD Stage",
                        )
                        agentic_pii_info = gr.Textbox(
                            label="PII Detection",
                            interactive=False,
                        )
                        agentic_eval_info = gr.Textbox(
                            label="Evaluation Scores",
                            interactive=False,
                            lines=4,
                        )
                        agentic_clear = gr.Button("Clear Chat")

                agentic_submit.click(
                    agentic_rag_query,
                    inputs=[agentic_input, agentic_ckd_stage, agentic_chatbot],
                    outputs=[agentic_chatbot, agentic_input, agentic_pii_info, agentic_eval_info],
                )
                agentic_input.submit(
                    agentic_rag_query,
                    inputs=[agentic_input, agentic_ckd_stage, agentic_chatbot],
                    outputs=[agentic_chatbot, agentic_input, agentic_pii_info, agentic_eval_info],
                )
                agentic_clear.click(lambda: ([], "", ""), outputs=[agentic_chatbot, agentic_pii_info, agentic_eval_info])

            # =================================================================
            # Tab 3: Multi-Agent
            # =================================================================
            with gr.TabItem("Multi-Agent", id="multi"):
                gr.Markdown(
                    """
                    ### Level 3: Multi-Agent System
                    Specialized agents for different domains: Diet, Medication, Lifestyle, and General Knowledge.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        multi_chatbot = gr.Chatbot(
                            label="Chat",
                            height=400,
                            show_copy_button=True,
                        )
                        multi_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What foods should I avoid and is ibuprofen safe?",
                            lines=2,
                        )
                        multi_submit = gr.Button("Ask", variant="primary")

                    with gr.Column(scale=1):
                        multi_ckd_stage = gr.Slider(
                            minimum=0,
                            maximum=5,
                            step=1,
                            value=3,
                            label="CKD Stage",
                        )
                        multi_weight = gr.Number(
                            label="Weight (kg)",
                            value=70,
                            minimum=30,
                            maximum=200,
                        )
                        multi_agents_info = gr.Textbox(
                            label="Agents Used",
                            interactive=False,
                        )
                        multi_routing_info = gr.Textbox(
                            label="Routing Decision",
                            interactive=False,
                            lines=4,
                        )
                        multi_clear = gr.Button("Clear Chat")

                multi_submit.click(
                    multi_agent_query,
                    inputs=[multi_input, multi_ckd_stage, multi_weight, multi_chatbot],
                    outputs=[multi_chatbot, multi_input, multi_agents_info, multi_routing_info],
                )
                multi_input.submit(
                    multi_agent_query,
                    inputs=[multi_input, multi_ckd_stage, multi_weight, multi_chatbot],
                    outputs=[multi_chatbot, multi_input, multi_agents_info, multi_routing_info],
                )
                multi_clear.click(lambda: ([], "", ""), outputs=[multi_chatbot, multi_agents_info, multi_routing_info])

            # =================================================================
            # Tab 4: About
            # =================================================================
            with gr.TabItem("About", id="about"):
                gr.Markdown(
                    """
                    ## About CKD Management Assistant

                    ### Project Overview
                    This is a 3-tier Retrieval-Augmented Generation (RAG) system for Chronic Kidney
                    Disease management, built for the **Kaggle MedGemma Impact Challenge**.

                    ### Technology Stack
                    | Component | Technology |
                    |-----------|------------|
                    | LLM | MedGemma 1.5 4B |
                    | Embeddings | EmbeddingGemma 300M |
                    | Vector Store | ChromaDB |
                    | Framework | LangChain + LangGraph |
                    | PII Detection | Microsoft Presidio |
                    | Evaluation | RAGAS |
                    | UI | Gradio |

                    ### Three Levels of RAG

                    **Level 1: Simple RAG**
                    - Basic retrieval and generation
                    - Source citations
                    - CKD stage filtering

                    **Level 2: Agentic RAG**
                    - PII detection and redaction
                    - Query intent classification
                    - LangGraph workflow orchestration
                    - Response evaluation with RAGAS

                    **Level 3: Multi-Agent RAG**
                    - Specialized agents for different domains
                    - Intelligent query routing
                    - Response synthesis from multiple sources

                    ### Data Sources
                    - NICE NG203: Chronic Kidney Disease Guidelines
                    - KidneyCareUK: Dietary and Patient Resources
                    - UK Kidney Association: Clinical Guidance

                    ### Disclaimer
                    This tool is for **educational purposes only** and should not be used as a
                    substitute for professional medical advice, diagnosis, or treatment. Always
                    seek the advice of your physician or other qualified health provider with
                    any questions you may have regarding a medical condition.

                    ### Competition
                    Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

                    ---
                    *Powered by MedGemma from Google Health AI*
                    """
                )

                with gr.Accordion("System Status", open=True):
                    status_text = gr.Markdown(check_initialization())
                    refresh_btn = gr.Button("Refresh Status")
                    refresh_btn.click(check_initialization, outputs=[status_text])

    return app


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
