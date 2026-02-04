# Level 2: Agentic RAG with LangGraph

Advanced RAG workflow with PII detection, query routing, and response evaluation.

## Architecture

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ PII Check   │ ──── Redact if needed
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Query       │
                    │ Analysis    │ ──── Classify intent
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌───▼────┐ ┌────▼─────┐
       │ Retrieval   │ │ Direct │ │ Clarify  │
       │ + Generate  │ │ Answer │ │ Question │
       └──────┬──────┘ └───┬────┘ └────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │ Evaluation  │ ──── RAGAS metrics
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    END      │
                    └─────────────┘
```

## Components

### pii_handler.py - PII Detection

Microsoft Presidio-based PII detection and anonymization.

**Supported Entities:**
- Names, emails, phone numbers
- UK NHS numbers (custom recognizer)
- Medical IDs and patient numbers
- Dates and locations

```python
from pii_handler import PIIHandler

handler = PIIHandler()
result = handler.anonymize("My name is John Smith, NHS: 123-456-7890")

print(result.anonymized_text)
# "My name is <PERSON_1>, NHS: <NHS_1>"

print(result.placeholder_map)
# {"<PERSON_1>": "John Smith", "<NHS_1>": "123-456-7890"}

# Restore original values
restored = handler.deanonymize(result)
```

### nodes.py - Workflow Nodes

Individual node functions for the LangGraph workflow.

**Nodes:**
- `pii_check`: Detect and redact PII
- `analyze_query`: Classify query intent
- `retrieve_documents`: Get relevant context
- `generate_response`: Create answer with LLM
- `evaluate_response`: Score with RAGAS

**Query Intents:**
- `RETRIEVAL`: Needs document lookup
- `DIRECT`: Can answer without retrieval
- `CLARIFICATION`: Needs more info
- `OUT_OF_SCOPE`: Outside CKD domain

### graph.py - LangGraph Workflow

Stateful workflow orchestration with conditional routing.

```python
from graph import AgenticRAGGraph

graph = AgenticRAGGraph(
    pii_handler=pii_handler,
    retriever=retriever,
    llm=llm,
    evaluator=evaluator
)

result = graph.invoke(
    query="My name is John, what potassium foods should I avoid?",
    ckd_stage=3
)

print(result["final_response"])
print(result["pii_detected"])  # True
print(result["evaluation_scores"])
```

## Evaluation Framework

### evaluation/ragas_eval.py - RAGAS Metrics

Standard RAG evaluation metrics:

| Metric | Description |
|--------|-------------|
| Faithfulness | Is answer grounded in context? |
| Answer Relevancy | Does it address the question? |
| Context Precision | Are retrieved docs relevant? |
| Context Recall | Are all relevant docs found? |

```python
from evaluation.ragas_eval import RAGASEvaluator

evaluator = RAGASEvaluator()
scores = evaluator.evaluate(
    query="What are potassium restrictions?",
    response="Limit potassium to 2000-3000mg daily.",
    contexts=["CKD stage 3 patients should limit potassium..."]
)

print(scores.faithfulness)
print(scores.answer_relevancy)
```

### evaluation/custom_metrics.py - CKD-Specific Metrics

Domain-specific quality measures:

| Metric | Description |
|--------|-------------|
| Citation Score | Are sources properly cited? |
| CKD Stage Appropriateness | Is advice correct for stage? |
| Disclaimer Present | Medical disclaimer included? |
| Actionability Score | Is advice actionable? |

```python
from evaluation.custom_metrics import CKDMetrics

metrics = CKDMetrics()
scores = metrics.evaluate(
    query="Potassium limits for stage 3?",
    response="Limit to 2000-3000mg daily [Source: NICE NG203]",
    ckd_stage=3
)

print(scores.citation_score)
print(scores.ckd_stage_appropriateness)
```

### evaluation/langsmith_setup.py - Tracing

LangSmith integration for debugging and analytics.

```python
from evaluation.langsmith_setup import setup_langsmith, LangSmithTracer

# Enable tracing
setup_langsmith(api_key="...", project_name="ckd-rag")

# Custom tracing
tracer = LangSmithTracer()

@tracer.trace("custom_operation")
def my_function():
    ...
```

## Usage

```python
from pii_handler import PIIHandler
from graph import AgenticRAGGraph

# Initialize components
pii_handler = PIIHandler()
graph = AgenticRAGGraph(
    pii_handler=pii_handler,
    retriever=retriever,
    llm=llm
)

# Process query
result = graph.invoke("What medications should I avoid?", ckd_stage=3)

# Stream results
for update in graph.stream("Dietary advice for stage 4"):
    print(update)
```

## Configuration

Environment variables:
- `LANGSMITH_API_KEY`: For tracing
- `LANGSMITH_PROJECT`: Project name
- `LANGSMITH_TRACING`: Enable/disable

See `config.py` for PII entity configuration.
