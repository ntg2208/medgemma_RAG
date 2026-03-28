"""
Terminal chat interface for the CKD RAG System.

Provides streaming chat for all 3 RAG levels with full tool/step logging.

Usage:
    uv run python main.py simple      # Level 1: Simple RAG
    uv run python main.py agentic     # Level 2: Agentic RAG
    uv run python main.py multi       # Level 3: Multi-Agent
"""

import argparse
import importlib
import importlib.util
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Colored output helpers
# ---------------------------------------------------------------------------
class C:
    GREY = "\033[90m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log_step(icon: str, label: str, detail: str = ""):
    ts = time.strftime("%H:%M:%S")
    print(f"  {C.GREY}{ts}{C.RESET} {icon} {C.CYAN}{label}{C.RESET} {detail}")


def log_tool(name: str, detail: str = ""):
    log_step("\U0001f527", name, detail)


def log_info(msg: str):
    log_step("\u2139\ufe0f ", "info", msg)


def log_warn(msg: str):
    log_step("\u26a0\ufe0f ", "warn", f"{C.YELLOW}{msg}{C.RESET}")


def log_error(msg: str):
    log_step("\u274c", "error", f"{C.RED}{msg}{C.RESET}")


# ---------------------------------------------------------------------------
# Package importer (handles numeric directory prefixes)
# ---------------------------------------------------------------------------
def import_package(package_name: str, package_dir: Path):
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Component initialization
# ---------------------------------------------------------------------------
def load_processed_chunks(processed_dir: Path) -> list:
    """Load all chunk JSON files from Data/processed/ into LangChain Documents."""
    import json
    from langchain_core.documents import Document

    docs = []
    chunk_files = sorted(processed_dir.glob("*_chunks.json"))
    for f in chunk_files:
        data = json.loads(f.read_text())
        meta = data.get("export_metadata", {})
        title = meta.get("document_title", f.stem)
        source = meta.get("source_file", f.name)
        for i, chunk in enumerate(data.get("chunks", [])):
            chunk_meta = chunk.get("metadata", {})
            chunk_meta.update({
                "source": source,
                "document_title": title,
                "chunk_id": i,
            })
            docs.append(Document(
                page_content=chunk["content"],
                metadata=chunk_meta,
            ))
    return docs, len(chunk_files)


def init_components():
    from config import get_llm, get_embeddings

    log_tool("embeddings", "loading embedding model...")
    embeddings = get_embeddings()

    log_tool("vectorstore", "loading ChromaDB...")
    rag_pkg = import_package("rag_pkg", PROJECT_ROOT / "1_Retrieval_Augmented_Generation")
    vectorstore = rag_pkg.CKDVectorStore(embeddings)

    # If vectorstore is empty, load processed chunks
    stats = vectorstore.get_collection_stats()
    if stats["document_count"] == 0:
        processed_dir = PROJECT_ROOT / "Data" / "processed"
        log_tool("vectorstore", f"empty collection — loading chunks from {processed_dir}...")
        docs, n_files = load_processed_chunks(processed_dir)
        if docs:
            log_info(f"loaded {len(docs)} chunks from {n_files} files")
            vectorstore.add_documents(docs)
            log_info(f"indexed {len(docs)} chunks into ChromaDB")
        else:
            log_warn(f"no chunk files found in {processed_dir}")
    else:
        log_info(f"vectorstore has {stats['document_count']} documents")

    log_tool("retriever", "initializing CKD retriever...")
    retriever = rag_pkg.CKDRetriever(vectorstore=vectorstore)

    log_tool("llm", "loading LLM...")
    llm = get_llm()

    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "llm": llm,
        "rag_pkg": rag_pkg,
    }


# ---------------------------------------------------------------------------
# Level 1: Simple RAG
# ---------------------------------------------------------------------------
def chat_simple(comps: dict):
    rag_pkg = comps["rag_pkg"]
    log_tool("simple_rag", "building RAG chain...")
    chain = rag_pkg.SimpleRAGChain(retriever=comps["retriever"], llm=comps["llm"])

    print(f"\n{C.BOLD}=== Simple RAG (Level 1) ==={C.RESET}")
    print(f"{C.GREY}Type 'quit' to exit.{C.RESET}\n")

    while True:
        try:
            query = input(f"{C.GREEN}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        log_tool("retriever", f"searching for top-k documents...")
        docs = comps["retriever"].invoke(query)
        log_info(f"retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "?")
            section = doc.metadata.get("section", "")
            log_step("  \U0001f4c4", f"doc {i}", f"{src}" + (f" > {section}" if section else ""))

        log_tool("llm", "generating response (streaming)...")
        # vLLM + MedGemma 1.5 renders <think> as <unused94>thought\n and </think> as <unused95>
        # Reset state for every query.
        OPEN_TAG, CLOSE_TAG = "<unused94>", "<unused95>"
        LOOK = max(len(OPEN_TAG), len(CLOSE_TAG)) - 1
        state = "pre"   # pre | think | answer
        buf = ""
        header_printed = False

        try:
            for chunk in chain.stream(query):
                buf += chunk
                while True:
                    if state == "pre":
                        idx = buf.find(OPEN_TAG)
                        if idx == -1:
                            safe = buf[: max(0, len(buf) - LOOK)]
                            if safe:
                                if not header_printed:
                                    print(f"\n{C.BOLD}Assistant:{C.RESET} ", end="", flush=True)
                                    header_printed = True
                                print(safe, end="", flush=True)
                            buf = buf[len(safe):]
                            break
                        buf = buf[idx + len(OPEN_TAG):]
                        # Strip the "thought\n" header text that follows the token
                        if buf.startswith("thought\n"):
                            buf = buf[len("thought\n"):]
                        state = "think"
                        print(f"\n{C.GREY}{'─' * 40}")
                        print(f"{C.BOLD}Thinking:{C.RESET}")
                    elif state == "think":
                        idx = buf.find(CLOSE_TAG)
                        if idx == -1:
                            safe = buf[: max(0, len(buf) - LOOK)]
                            buf = buf[len(safe):]
                            print(f"{C.GREY}{safe}{C.RESET}", end="", flush=True)
                            break
                        print(f"{C.GREY}{buf[:idx]}{C.RESET}", end="", flush=True)
                        buf = buf[idx + len(CLOSE_TAG):].lstrip("\n")
                        state = "answer"
                        print(f"\n{C.GREY}{'─' * 40}{C.RESET}")
                        print(f"\n{C.BOLD}Assistant:{C.RESET} ", end="", flush=True)
                    elif state == "answer":
                        print(buf, end="", flush=True)
                        buf = ""
                        break

            # Flush remainder
            if state == "answer":
                print(buf, end="", flush=True)
            elif state == "think":
                print(f"{C.GREY}{buf}{C.RESET}", end="", flush=True)
            elif buf:
                # Remaining chars held back for tag detection
                if not header_printed:
                    print(f"\n{C.BOLD}Assistant:{C.RESET} ", end="", flush=True)
                print(buf, end="", flush=True)
        except Exception as e:
            log_warn(f"streaming failed ({e}), falling back to invoke")
            resp = chain.invoke(query)
            print(f"\n{C.BOLD}Assistant:{C.RESET} {resp.answer}", end="")
        print()


# ---------------------------------------------------------------------------
# Level 2: Agentic RAG
# ---------------------------------------------------------------------------
def chat_agentic(comps: dict):
    log_tool("pii_handler", "initializing Presidio PII handler...")
    agentic_pkg = import_package("agentic_pkg", PROJECT_ROOT / "2_Agentic_RAG")
    pii_handler = agentic_pkg.PIIHandler()

    log_tool("agentic_graph", "building LangGraph workflow...")
    graph = agentic_pkg.AgenticRAGGraph(
        pii_handler=pii_handler,
        retriever=comps["retriever"],
        llm=comps["llm"],
    )

    print(f"\n{C.BOLD}=== Agentic RAG (Level 2) ==={C.RESET}")
    print(f"{C.GREY}Type 'quit' to exit. Prefix with 'stage:N ' to set CKD stage.{C.RESET}\n")

    while True:
        try:
            raw = input(f"{C.GREEN}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        # Parse optional CKD stage prefix
        ckd_stage = None
        query = raw
        if raw.lower().startswith("stage:"):
            parts = raw.split(" ", 1)
            try:
                ckd_stage = int(parts[0].split(":")[1])
                query = parts[1] if len(parts) > 1 else ""
            except (ValueError, IndexError):
                pass

        if not query:
            continue

        # Stream through the graph (yields node-by-node updates)
        log_info(f"ckd_stage={ckd_stage}")
        print()
        final_state = {}
        try:
            for state_update in graph.stream(query, ckd_stage=ckd_stage):
                for node_name, node_output in state_update.items():
                    # Log each node execution
                    steps = node_output.get("processing_steps", [])
                    step_label = steps[-1] if steps else node_name
                    log_tool(node_name, f"{C.GREY}{step_label}{C.RESET}")

                    # Show PII detection
                    if node_output.get("pii_detected"):
                        pii_map = node_output.get("pii_map", {})
                        log_warn(f"PII detected: {len(pii_map)} item(s) redacted")

                    # Show intent routing
                    if "query_intent" in node_output:
                        intent = node_output["query_intent"]
                        intent_str = intent.value if hasattr(intent, "value") else str(intent)
                        log_info(f"intent → {C.YELLOW}{intent_str}{C.RESET}")

                    # Show retrieved docs
                    if "retrieved_documents" in node_output:
                        docs = node_output["retrieved_documents"]
                        log_info(f"retrieved {len(docs)} documents")
                        for i, doc in enumerate(docs[:5], 1):
                            src = doc.metadata.get("source", "?")
                            log_step("  \U0001f4c4", f"doc {i}", src)

                    # Show evaluation
                    if "evaluation_scores" in node_output:
                        scores = node_output["evaluation_scores"]
                        if scores:
                            parts = [f"{k}={v:.2f}" for k, v in scores.items()]
                            log_info(f"eval: {', '.join(parts)}")

                    final_state.update(node_output)

        except Exception as e:
            log_error(str(e))
            continue

        answer = final_state.get("final_response", final_state.get("raw_response", "No response."))
        print(f"\n{C.BOLD}Assistant:{C.RESET} {answer}\n")


# ---------------------------------------------------------------------------
# Level 3: Multi-Agent
# ---------------------------------------------------------------------------
def chat_multi(comps: dict):
    log_tool("pii_handler", "initializing Presidio PII handler...")
    agentic_pkg = import_package("agentic_pkg", PROJECT_ROOT / "2_Agentic_RAG")
    pii_handler = agentic_pkg.PIIHandler()

    log_tool("orchestrator", "initializing multi-agent system...")
    import_package("multi_pkg", PROJECT_ROOT / "3_MultiAgent_RAG")
    import_package("multi_pkg.agents", PROJECT_ROOT / "3_MultiAgent_RAG" / "agents")
    multi_pkg = sys.modules["multi_pkg"]
    orchestrator = multi_pkg.MultiAgentOrchestrator(
        retriever=comps["retriever"],
        llm=comps["llm"],
        pii_handler=pii_handler,
    )

    print(f"\n{C.BOLD}=== Multi-Agent RAG (Level 3) ==={C.RESET}")
    print(f"{C.GREY}Type 'quit' to exit. Prefix with 'stage:N ' to set CKD stage.{C.RESET}\n")

    while True:
        try:
            raw = input(f"{C.GREEN}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        # Parse optional CKD stage prefix
        ckd_stage = None
        query = raw
        if raw.lower().startswith("stage:"):
            parts = raw.split(" ", 1)
            try:
                ckd_stage = int(parts[0].split(":")[1])
                query = parts[1] if len(parts) > 1 else ""
            except (ValueError, IndexError):
                pass

        if not query:
            continue

        log_info(f"ckd_stage={ckd_stage}")

        # Route
        log_tool("router", "classifying query...")
        routing = orchestrator.route(query)
        log_info(
            f"route → {C.YELLOW}{routing.primary_agent.value}{C.RESET} "
            f"(confidence={routing.confidence:.0%})"
        )
        if routing.secondary_agents:
            agents_str = ", ".join(a.value for a in routing.secondary_agents)
            log_info(f"secondary agents: {agents_str}")
        log_info(f"reasoning: {routing.reasoning}")

        # Process
        log_tool("agents", "processing query...")
        t0 = time.time()
        try:
            response = orchestrator.process(
                query=query,
                ckd_stage=ckd_stage,
            )
        except Exception as e:
            log_error(str(e))
            continue
        elapsed = time.time() - t0

        log_info(f"agents used: {', '.join(response.agents_used)}")
        log_info(f"completed in {elapsed:.1f}s")

        print(f"\n{C.BOLD}Assistant:{C.RESET} {response.answer}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CKD RAG Terminal Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  uv run python main.py simple      # Level 1: Simple RAG
  uv run python main.py agentic     # Level 2: Agentic RAG (LangGraph)
  uv run python main.py multi       # Level 3: Multi-Agent
  uv run python main.py simple -v   # With debug logging
        """,
    )
    parser.add_argument(
        "level",
        choices=["simple", "agentic", "multi"],
        help="RAG level to use",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging — show all component logs to terminal
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=f"{C.GREY}%(asctime)s{C.RESET} %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{C.BOLD}CKD Management RAG — Terminal Chat{C.RESET}")
    print(f"{C.GREY}{'─' * 40}{C.RESET}\n")

    log_info("initializing components...")
    try:
        comps = init_components()
    except Exception as e:
        log_error(f"initialization failed: {e}")
        sys.exit(1)

    log_info("ready!\n")

    handlers = {
        "simple": chat_simple,
        "agentic": chat_agentic,
        "multi": chat_multi,
    }

    try:
        handlers[args.level](comps)
    except KeyboardInterrupt:
        pass

    print(f"\n{C.GREY}Goodbye.{C.RESET}")


if __name__ == "__main__":
    main()
