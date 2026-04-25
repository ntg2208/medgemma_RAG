# MedGemma RAG — Architecture Documentation

> CKD Clinical Knowledge Assistant | Kaggle MedGemma Impact Challenge

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Level 1 — Simple RAG](#2-level-1--simple-rag)
3. [Level 2 — Agentic RAG (LangGraph)](#3-level-2--agentic-rag-langgraph)
4. [Level 3 — Multi-Agent Orchestration](#4-level-3--multi-agent-orchestration)
5. [Component Class Diagram](#5-component-class-diagram)
6. [Data Structures](#6-data-structures)
7. [Infrastructure & Deployment](#7-infrastructure--deployment)
8. [Developer Workflow](#8-developer-workflow)

---

## 1. System Overview

The system is a 3-tier RAG architecture. Each tier builds on the previous, adding intelligence and specialization.

```mermaid
graph TB
    User([User Query]) --> UI[Gradio UI\napp.py]

    UI --> L1[Level 1\nSimple RAG]
    UI --> L2[Level 2\nAgentic RAG]
    UI --> L3[Level 3\nMulti-Agent]

    subgraph L1_box["Level 1 — Simple RAG"]
        L1 --> E1[EmbeddingGemma\n300M]
        E1 --> C1[ChromaDB\nVector Search]
        C1 --> G1[MedGemma 1.5 4B\nGenerate]
        G1 --> R1[RAGResponse]
    end

    subgraph L2_box["Level 2 — Agentic RAG (LangGraph)"]
        L2 --> PII[PII Handler\nPresidio]
        PII --> ANA[Query Analyzer\nIntent Classification]
        ANA --> ROT{Router}
        ROT --> RET[Retrieval Path]
        ROT --> DIR[Direct Answer]
        ROT --> CLR[Clarification]
        ROT --> OOS[Out of Scope]
        RET --> EVL[RAGAS Evaluator]
        EVL --> R2[Evaluated Response]
    end

    subgraph L3_box["Level 3 — Multi-Agent"]
        L3 --> ORC[Orchestrator\nKeyword Router]
        ORC --> DA[Diet Agent]
        ORC --> MA[Medication Agent]
        ORC --> LA[Lifestyle Agent]
        ORC --> RA[RAG Agent]
        DA & MA & LA & RA --> SYN[Synthesizer]
        SYN --> R3[OrchestratorResponse]
    end

    subgraph SharedInfra["Shared Components"]
        VDB[(ChromaDB\nVectorstore)]
        LLM[MedGemma LLM\nor vLLM Remote]
        EMB[EmbeddingGemma\nor TEI Remote]
    end

    C1 --- VDB
    G1 --- LLM
    E1 --- EMB
    RET --- VDB
    RA --- VDB
```

---

## 2. Level 1 — Simple RAG

**File:** `simple_rag/`

Straightforward retrieve-then-generate pipeline. Best for direct clinical questions.

```mermaid
flowchart LR
    Q[/"User Query"/] --> EW

    subgraph Embed["Embedding"]
        EW[EmbeddingGemmaWrapper\n.embed_query]
        EW -->|768-dim vector| QV[(Query Vector)]
    end

    subgraph Retrieve["Retrieval"]
        QV --> EX[CKDRetriever\n._expand_query\nmedical term expansion]
        EX --> CS[ChromaDB\ncosine similarity]
        CS -->|top-k docs\nscore ≥ 0.3| RD[Retrieved\nDocuments]
    end

    subgraph Generate["Generation"]
        RD --> FC[SimpleRAGChain\n._format_context]
        FC -->|context + prompt| LLM[MedGemmaLLM\n.generate]
        LLM --> ANS[/"RAGResponse\nanswer + sources"/]
    end
```

**Key classes:**

| Class | File | Role |
|-------|------|------|
| `EmbeddingGemmaWrapper` | `embeddings.py` | Embed queries and documents (MRL: 128–768 dims) |
| `CachedEmbeddingGemma` | `embeddings.py` | In-memory cache for dev/testing |
| `CKDVectorStore` | `vectorstore.py` | ChromaDB wrapper with metadata filtering |
| `CKDRetriever` | `retriever.py` | LangChain retriever + medical term expansion |
| `HybridRetriever` | `retriever.py` | Semantic + keyword search via Reciprocal Rank Fusion |
| `MedGemmaLLM` | `chain.py` | 4-bit quantized MedGemma wrapper |
| `SimpleRAGChain` | `chain.py` | Orchestrates retrieve → augment → generate |

---

## 3. Level 2 — Agentic RAG (LangGraph)

**File:** `agentic_rag/`

Stateful LangGraph workflow with PII protection, intent routing, and RAGAS evaluation.

### 3.1 Graph Structure

```mermaid
stateDiagram-v2
    [*] --> pii_check
    pii_check --> analyze_query

    analyze_query --> router

    state router <<choice>>
    router --> retrieve_documents : RETRIEVAL
    router --> generate_direct_response : DIRECT
    router --> generate_clarification : CLARIFICATION
    router --> generate_out_of_scope : OUT_OF_SCOPE

    retrieve_documents --> generate_response
    generate_response --> evaluate_response
    evaluate_response --> [*]

    generate_direct_response --> [*]
    generate_clarification --> [*]
    generate_out_of_scope --> [*]
```

### 3.2 Detailed Node Flow

```mermaid
flowchart TD
    IN[/"Query + CKD Stage"/]

    IN --> N1

    subgraph N1["pii_check (RetryPolicy: 3×, 2–60s)"]
        P1[Presidio Analyzer] --> P2[NHSNumberRecognizer]
        P1 --> P3[MedicalIDRecognizer]
        P2 & P3 --> P4[Anonymize → pii_map]
    end

    N1 -->|anonymized_query\npii_map| N2

    subgraph N2["analyze_query"]
        A1[Keyword Extraction] --> A2{Intent\nClassification}
        A2 -->|ckd + retrieve terms| RINT[RETRIEVAL]
        A2 -->|direct knowledge| DINT[DIRECT]
        A2 -->|incomplete info| CINT[CLARIFICATION]
        A2 -->|non-CKD| OINT[OUT_OF_SCOPE]
    end

    RINT --> N3

    subgraph N3["retrieve_documents (RetryPolicy: 3×, 2–60s)"]
        R1[CKDRetriever.invoke] --> R2[ChromaDB Search]
        R2 --> R3[Top-K Documents\nscore ≥ threshold]
    end

    N3 --> N4

    subgraph N4["generate_response (RetryPolicy: 3×, 1–30s)"]
        G1[Format Context\nwith Citations] --> G2[MedGemmaLLM.generate]
        G2 --> G3[PIIHandler.restore\nif pii_detected]
    end

    N4 --> N5

    subgraph N5["evaluate_response (RetryPolicy: 2×, 1s)"]
        E1[RAGAS Evaluator]
        E1 -->|faithfulness| SC[Scores]
        E1 -->|answer_relevance| SC
        E1 -->|context_precision| SC
    end

    N5 --> OUT[/"Final Response\n+ Evaluation Scores"/]

    DINT --> ND[generate_direct_response] --> OUT
    CINT --> NC[generate_clarification] --> OUT
    OINT --> NO[generate_out_of_scope] --> OUT
```

### 3.3 State Schema

```mermaid
classDiagram
    class AgenticGraphState {
        +str original_query
        +int ckd_stage
        +str anonymized_query
        +bool pii_detected
        +dict pii_map
        +str query_intent
        +list query_keywords
        +list retrieved_documents
        +str context
        +str raw_response
        +str final_response
        +dict evaluation_scores
        +str error
        +list processing_steps
    }
```

---

## 4. Level 3 — Multi-Agent Orchestration

**File:** `multi_agent_rag/`

Keyword-scored routing dispatches to one or more specialized agents. Responses are synthesized into a unified answer.

### 4.1 Orchestration Flow

```mermaid
flowchart TD
    Q[/"Query + CKD Stage\n+ Weight kg"/]

    Q --> PII_L3[PIIHandler\nAnonymize]

    PII_L3 --> SCORE

    subgraph SCORE["Keyword Scoring (orchestrator.py)"]
        direction LR
        KD[diet keywords\n→ score]
        KM[medication keywords\n→ score]
        KL[lifestyle keywords\n→ score]
        KR[rag keywords\n→ score]
    end

    SCORE --> ROUTE{Route\nDecision}

    ROUTE -->|single winner| SINGLE[Single Agent]
    ROUTE -->|score ≥ 30% of primary| MULTI[Multi-Agent\nDispatch]

    SINGLE --> AGENTS

    subgraph AGENTS["Specialized Agents"]
        direction TB
        DA[DietAgent\nK+/P/Na/protein limits\nby stage + weight]
        MA[MedicationAgent\nNSAID warnings\nnephrotoxic flags]
        LA[LifestyleAgent\nexercise, BP, hydration\nsmoking, sleep, stress]
        RA[RAGAgent\nNICE/KDIGO guidelines\nvectorstore retrieval]
    end

    MULTI --> AGENTS

    AGENTS --> SYN{Synthesize\nif multi}
    SYN --> RESP[/"OrchestratorResponse\nanswer + agents_used\n+ individual_responses\n+ confidence + disclaimer"/]
```

### 4.2 Agent Class Hierarchy

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        +LLM llm
        +can_handle(query) bool*
        +answer(query, **kwargs) AgentResponse*
    }

    class AgentResponse {
        +str answer
        +float confidence
        +str disclaimer
    }

    class RAGAgent {
        +CKDRetriever retriever
        +can_handle(query) bool
        +answer(query, ckd_stage) RAGAgentResponse
    }

    class DietAgent {
        +can_handle(query) bool
        +answer(query, ckd_stage, weight_kg) DietAgentResponse
        -_calculate_limits(stage, weight_kg) dict
        -_food_database dict
    }

    class MedicationAgent {
        +can_handle(query) bool
        +answer(query, ckd_stage) MedicationAgentResponse
        -_nephrotoxic_drugs list
    }

    class LifestyleAgent {
        +can_handle(query) bool
        +answer(query, ckd_stage) LifestyleAgentResponse
    }

    class MultiAgentOrchestrator {
        +RAGAgent rag_agent
        +DietAgent diet_agent
        +MedicationAgent medication_agent
        +LifestyleAgent lifestyle_agent
        +route(query) RoutingDecision
        +process(query, ckd_stage, weight_kg) OrchestratorResponse
        -_call_agent(agent, query, kwargs) AgentResponse
        -_synthesize_responses(responses) str
    }

    BaseAgent <|-- RAGAgent
    BaseAgent <|-- DietAgent
    BaseAgent <|-- MedicationAgent
    BaseAgent <|-- LifestyleAgent
    MultiAgentOrchestrator --> RAGAgent
    MultiAgentOrchestrator --> DietAgent
    MultiAgentOrchestrator --> MedicationAgent
    MultiAgentOrchestrator --> LifestyleAgent
    BaseAgent ..> AgentResponse
```

### 4.3 Routing Logic

```mermaid
flowchart LR
    Q[Query] --> KS[Score each\nagent type\nby keyword hits]

    KS --> PRIM[Primary =\nhighest score]
    KS --> THRESH["Secondary threshold =\nprimary × 0.30"]

    THRESH --> CHECK{Any others\nabove threshold?}
    CHECK -->|no| SINGLE_D[Single agent]
    CHECK -->|yes| MULTI_D[Multi-agent\nall above threshold]
```

---

## 5. Component Class Diagram

Full cross-tier dependency map.

```mermaid
classDiagram
    class config {
        +MEDGEMMA_MODEL_ID
        +EMBEDDING_MODEL_ID
        +CHUNK_SIZE = 2000
        +TOP_K_RESULTS = 5
        +SIMILARITY_THRESHOLD = 0.3
        +CKD_STAGES dict
        +DIETARY_LIMITS dict
        +get_llm() LLM
        +get_embeddings() Embeddings
    }

    class MedGemmaLLM {
        +str model_id
        +str device
        +generate(prompt) str
        +as_langchain_llm() BaseLLM
    }

    class EmbeddingGemmaWrapper {
        +int dimension
        +embed_query(text) list
        +embed_documents(texts) list
        +similarity(v1, v2) float
    }

    class CKDVectorStore {
        +ChromaClient client
        +add_documents(docs) list
        +search(query, k) list
        +search_with_scores(query) list
    }

    class CKDRetriever {
        +CKDVectorStore vectorstore
        +_expand_query(q) str
        +_get_relevant_documents(q) list
    }

    class SimpleRAGChain {
        +CKDRetriever retriever
        +MedGemmaLLM llm
        +invoke(query) RAGResponse
        +stream(query) iterator
    }

    class AgenticRAGGraph {
        +StateGraph graph
        +invoke(query, ckd_stage) dict
        +stream(query) iterator
    }

    class RAGNodes {
        +PIIHandler pii_handler
        +CKDRetriever retriever
        +MedGemmaLLM llm
        +pii_check(state) state
        +analyze_query(state) state
        +retrieve_documents(state) state
        +generate_response(state) state
        +evaluate_response(state) state
    }

    class PIIHandler {
        +AnalyzerEngine analyzer
        +anonymize(text) PIIDetectionResult
        +restore_in_response(text, map) str
    }

    class MultiAgentOrchestrator {
        +process(query, stage, weight) OrchestratorResponse
        +route(query) RoutingDecision
    }

    config --> MedGemmaLLM : get_llm()
    config --> EmbeddingGemmaWrapper : get_embeddings()
    EmbeddingGemmaWrapper --> CKDVectorStore
    CKDVectorStore --> CKDRetriever
    CKDRetriever --> SimpleRAGChain
    MedGemmaLLM --> SimpleRAGChain
    CKDRetriever --> RAGNodes
    MedGemmaLLM --> RAGNodes
    PIIHandler --> RAGNodes
    RAGNodes --> AgenticRAGGraph
    CKDRetriever --> MultiAgentOrchestrator
    MedGemmaLLM --> MultiAgentOrchestrator
    PIIHandler --> MultiAgentOrchestrator
```

---

## 6. Data Structures

### Response Types by Level

```mermaid
classDiagram
    class RAGResponse {
        +str answer
        +list~Document~ source_documents
        +str query
    }

    class AgenticGraphState {
        +str original_query
        +str anonymized_query
        +bool pii_detected
        +dict pii_map
        +str query_intent
        +list retrieved_documents
        +str final_response
        +dict evaluation_scores
        +list processing_steps
    }

    class RoutingDecision {
        +AgentType primary_agent
        +list secondary_agents
        +float confidence
        +str reasoning
    }

    class OrchestratorResponse {
        +str answer
        +list~str~ agents_used
        +RoutingDecision routing_decision
        +dict individual_responses
        +int ckd_stage
        +float confidence
        +str disclaimer
    }

    class DietAgentResponse {
        +list~DietaryRecommendation~ recommendations
        +str summary
        +int ckd_stage
        +float weight_kg
        +float confidence
    }

    class DietaryRecommendation {
        +str nutrient
        +str daily_limit
        +list foods_to_limit
        +list foods_to_prefer
    }

    OrchestratorResponse --> RoutingDecision
    OrchestratorResponse --> DietAgentResponse
    DietAgentResponse --> DietaryRecommendation
```

---

## 7. Infrastructure & Deployment

### 7.1 Two Deployment Modes

```mermaid
flowchart TB
    subgraph LOCAL["Local Mode (default)"]
        direction TB
        LU([User]) --> LGUI[Gradio UI\nlocalhost:7860]
        LGUI --> LLLM[MedGemma 1.5 4B\nMPS/CPU]
        LGUI --> LEMB[EmbeddingGemma 300M\nMPS/CPU]
        LGUI --> LDB[(ChromaDB\nlocal disk)]
    end

    subgraph REMOTE["Remote Mode (USE_REMOTE_MODELS=true)"]
        direction TB
        RU([User]) --> RGUI[Gradio UI\nMacBook]
        RGUI --> RDB[(ChromaDB\nlocal disk)]
        RGUI <-->|HTTP| RVLLM["vLLM :8000\nMedGemma 1.5 4B\nA10G / L4 GPU"]
        RGUI <-->|HTTP| RTEI["TEI :8001\nEmbeddingGemma 300M\nGPU"]

        subgraph EC2["EC2 Spot Instance (g5/g6.xlarge)"]
            RVLLM
            RTEI
            RS3[(S3 Model Cache\n~9.3 GB)]
            RVLLM -.->|sync on boot\n30–60 sec| RS3
            RTEI -.->|sync on boot| RS3
        end
    end
```

### 7.2 AWS Infrastructure (Terraform — one-time apply)

```mermaid
graph TB
    subgraph TF["terraform apply (one-time)"]
        S3[S3 Bucket\nmedgemma-models-{account_id}\nprevent_destroy=true]
        IAM[IAM Role\nmedgemma-model-server-role]
        POL[IAM Policy\ns3:Get/Put/List]
        PRF[IAM Instance Profile\nmedgemma-model-server-profile]
        SG[Security Group\nmedgemma-model-server-sg\nSSH:22 vLLM:8000 TEI:8001]

        IAM --> POL
        POL -->|allows access to| S3
        IAM --> PRF
    end

    subgraph SCRIPTS["scripts/ — per-session"]
        START[start.sh\naws ec2 run-instances\nspot one-time\n100GB EBS gp3\ndelete_on_termination=true]
        STOP[stop.sh\naws ec2 terminate-instances\nby tag Name]
        STATUS[status.sh\nid, ip, type, uptime\nby tag Name]
        SYNC[sync.sh\nrsync local → EC2\nor pull EC2 → local]
    end

    PRF -->|attached to| START
    SG -->|applied to| START
    START -->|tags| EC2I[EC2 Instance\nmedgemma-model-server]
    EC2I -->|terminated by| STOP
    EC2I -->|queried by| STATUS
    EC2I -->|synced by| SYNC
```

### 7.3 EC2 Session Workflow

```mermaid
sequenceDiagram
    participant Dev as Developer (local)
    participant Script as scripts/
    participant AWS as AWS EC2
    participant S3 as S3 Model Cache

    Note over Dev,AWS: Session Start
    Dev->>Script: ./scripts/start.sh
    Script->>AWS: aws ec2 run-instances (spot, g6.xlarge)
    AWS-->>Script: instance-id
    Script->>AWS: wait instance-running
    AWS-->>Script: public IP
    Script->>Dev: Updates ~/.ssh/config → ssh medgemma-gpu

    Dev->>Script: ./scripts/sync.sh
    Script->>AWS: rsync local → /home/ubuntu/medgemma_RAG/

    Dev->>AWS: ssh medgemma-gpu
    AWS->>S3: aws s3 sync models (30–60 sec)
    AWS->>AWS: startup.sh --start (vLLM + TEI)

    Note over Dev,AWS: Working Session
    Dev->>Script: ./scripts/sync.sh (push changes)
    Dev->>AWS: test endpoints

    Note over Dev,AWS: Session End
    Dev->>Script: ./scripts/stop.sh
    Script->>AWS: aws ec2 terminate-instances (by tag)
    AWS-->>AWS: EBS volume deleted automatically
```

---

## 8. Developer Workflow

```mermaid
flowchart TD
    FIRST{First time\nthis project?}

    FIRST -->|yes| TF[cd infrastructure/terraform\nterraform apply]
    FIRST -->|no| START

    TF -->|creates S3, IAM, SG| START[./scripts/start.sh]

    START --> SYNC[./scripts/sync.sh\nrsync local → EC2]
    SYNC --> SSH[ssh medgemma-gpu]
    SSH --> STARTUP[bash startup.sh --start\nvLLM + TEI servers]
    STARTUP --> DEV[Develop + Test]

    DEV --> PUSHSYNC[./scripts/sync.sh\npush code changes]
    PUSHSYNC --> DEV

    DEV --> CHECK[./scripts/status.sh\ncheck id, ip, uptime]
    CHECK --> DEV

    DEV --> DONE{Done for\nthe day?}
    DONE -->|no| DEV
    DONE -->|yes| STOP[./scripts/stop.sh\nterminate instance\nEBS auto-deleted]
```

---

## Quick Reference

| Script | Command | Description |
|--------|---------|-------------|
| One-time setup | `cd infrastructure/terraform && terraform apply` | Create S3, IAM, SG |
| Launch instance | `./scripts/start.sh` | Spot g6.xlarge + 100GB EBS |
| Check status | `./scripts/status.sh` | ID, IP, type, uptime |
| Push code | `./scripts/sync.sh` | Local → EC2 rsync |
| Pull code | `./scripts/sync.sh --pull` | EC2 → local rsync |
| Terminate | `./scripts/stop.sh` | Terminate by tag name |

| Env Var | Default | Description |
|---------|---------|-------------|
| `USE_REMOTE_MODELS` | `false` | Use EC2 vLLM/TEI instead of local |
| `MODEL_SERVER_URL` | `http://localhost:8000` | vLLM endpoint |
| `EMBEDDING_SERVER_URL` | `http://localhost:8001` | TEI endpoint |
| `INSTANCE_TYPE` | `g6.xlarge` | Override in start.sh |
| `AMI_ID` | Deep Learning AMI | Override in start.sh |
