# Level 3: Multi-Agent RAG System

Specialized agents for different CKD management domains with intelligent query routing.

## Architecture

```
                         ┌─────────────────┐
                         │   User Query    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  ORCHESTRATOR   │
                         │  (Router Agent) │
                         └────────┬────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
    ┌──────▼──────┐       ┌──────▼──────┐       ┌──────▼──────┐
    │ RAG Agent   │       │ Diet Agent  │       │ Medication  │
    │ (Knowledge) │       │ (Calculator)│       │   Agent     │
    └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
           │                      │                      │
           │              ┌──────▼──────┐               │
           │              │ Lifestyle   │               │
           │              │   Agent     │               │
           │              └──────┬──────┘               │
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  Synthesize &   │
                         │    Respond      │
                         └─────────────────┘
```

## Specialized Agents

### agents/rag_agent.py - Knowledge Retrieval

General CKD knowledge from NICE guidelines and KidneyCareUK.

**Capabilities:**
- Evidence-based information retrieval
- Source citation
- CKD stage filtering

```python
from agents.rag_agent import RAGAgent

agent = RAGAgent(retriever=retriever, llm=llm)
response = agent.answer("What is the target blood pressure for CKD?")
print(response.answer)
print(response.sources)
```

### agents/diet_agent.py - Dietary Calculator

Personalized dietary recommendations based on CKD stage and weight.

**Daily Limits by Stage:**

| Nutrient | Stage 1-2 | Stage 3 | Stage 4-5 |
|----------|-----------|---------|-----------|
| Potassium | 2500-4700mg | 2000-3000mg | 1500-2500mg |
| Phosphorus | 800-1200mg | 800-1000mg | 800-1000mg |
| Sodium | <2300mg | <2000mg | <2000mg |
| Protein | 0.8-1.0 g/kg | 0.6-0.8 g/kg | 0.6-0.8 g/kg |

```python
from agents.diet_agent import DietAgent

agent = DietAgent()
response = agent.calculate(ckd_stage=3, weight_kg=70)

for rec in response.recommendations:
    print(f"{rec.nutrient}: {rec.daily_limit} {rec.unit}")
    print(f"Foods to limit: {rec.foods_to_limit}")
```

### agents/medication_agent.py - Medication Checker

Kidney-safe medication guidance with drug interaction warnings.

**Risk Levels:**
- `SAFE`: Generally safe for CKD
- `CAUTION`: Use with monitoring
- `AVOID`: Generally avoid in CKD
- `CONTRAINDICATED`: Do not use

**Covered Medications:**
- NSAIDs (ibuprofen, naproxen) - AVOID
- Paracetamol - SAFE
- ACE inhibitors - SAFE (kidney protective)
- Metformin - CAUTION (dose adjust)
- Aminoglycosides - AVOID

```python
from agents.medication_agent import MedicationAgent

agent = MedicationAgent()
response = agent.check("ibuprofen", ckd_stage=3)

print(response.medications_analyzed[0].risk_level)  # AVOID
print(response.medications_analyzed[0].alternatives)
```

### agents/lifestyle_agent.py - Lifestyle Coach

Guidance on exercise, blood pressure, smoking, stress, and sleep.

**Topics:**
- Physical activity recommendations
- Hydration guidance
- Blood pressure monitoring
- Smoking cessation
- Stress management
- Sleep hygiene
- Weight management

```python
from agents.lifestyle_agent import LifestyleAgent

agent = LifestyleAgent()
response = agent.answer("What exercise is safe for CKD?", ckd_stage=3)

for rec in response.recommendations:
    print(f"{rec.title}")
    print(f"{rec.guidance}")
    print(f"Tips: {rec.tips}")
```

## Orchestrator

### orchestrator.py - Query Routing

Intelligent routing to appropriate agent(s) based on query analysis.

**Routing Keywords:**
- Diet: "food", "eat", "potassium", "phosphorus", "nutrition"
- Medication: "medicine", "drug", "safe to take", "painkiller"
- Lifestyle: "exercise", "blood pressure", "smoking", "stress"
- RAG: "guideline", "recommendation", "what is", "stage"

```python
from orchestrator import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator(
    retriever=retriever,
    llm=llm,
    pii_handler=pii_handler
)

# Single-agent query
response = orchestrator.process(
    query="What foods should I avoid?",
    ckd_stage=3
)
print(response.agents_used)  # ["diet"]

# Multi-agent query
response = orchestrator.process(
    query="What foods should I avoid and is ibuprofen safe?",
    ckd_stage=3
)
print(response.agents_used)  # ["diet", "medication"]
```

## Response Synthesis

For multi-agent queries, responses are synthesized into a coherent answer:

```python
response = orchestrator.process(
    query="Diet and exercise advice for stage 3 CKD",
    ckd_stage=3,
    weight_kg=70
)

print(response.answer)
# ## Dietary Recommendations
# Key daily limits:
# - Potassium: 2000-3000 mg
# - Protein: 42-56g (based on weight)
# ...
#
# ## Lifestyle Recommendations
# Exercise is beneficial for CKD patients...
```

## Usage Example

```python
from orchestrator import MultiAgentOrchestrator

# Initialize
orchestrator = MultiAgentOrchestrator(
    retriever=retriever,
    llm=llm
)

# Process query
response = orchestrator.process(
    query="I weigh 70kg, what protein should I eat and can I take ibuprofen?",
    ckd_stage=3,
    weight_kg=70
)

print(f"Answer: {response.answer}")
print(f"Agents: {response.agents_used}")
print(f"Routing: {response.routing_decision.reasoning}")
print(f"Confidence: {response.confidence:.0%}")
```

## Agent Capabilities Summary

| Agent | Input | Output |
|-------|-------|--------|
| RAG | Query, CKD stage | Answer with citations |
| Diet | Query, stage, weight | Nutrient limits, food lists |
| Medication | Drug name, stage | Risk level, alternatives |
| Lifestyle | Topic, stage | Recommendations, tips |
