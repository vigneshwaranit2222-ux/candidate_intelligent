# AI-Powered Candidate Intelligence and Job Readiness Platform (Phase 1)

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.6+-red.svg)](https://docs.pydantic.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-purple.svg)](https://python.langchain.com/docs/langgraph)
[![PostgreSQL pgvector](https://img.shields.io/badge/pgvector-VectorSearch-orange.svg)](https://github.com/pgvector/pgvector)

An end-to-end asynchronous FastAPI candidate evaluation microservice combining **PyMuPDF / pdfplumber resume parsing**, **LangGraph state machines**, **PostgreSQL pgvector RAG similarity search**, **hybrid role matching**, and **adaptive difficulty technical assessments**.

---

## 🏗️ Architecture & 5 End-to-End Functional Modules

```text
ai-candidate-platform/
├── app/
│   ├── main.py                     # FastAPI application startup, lifespan & CORS middleware
│   ├── core/
│   │   └── config.py               # Pydantic v2 BaseSettings (.env loader)
│   ├── db/
│   │   ├── session.py              # Async SQLAlchemy 2.0 engine & session maker
│   │   └── models.py               # Candidate, ResumeChunk (pgvector), Assessment & Report ORMs
│   ├── models/
│   │   └── schemas.py              # Pydantic v2 domain models (CandidateProfile, Turn, Report)
│   ├── services/
│   │   ├── parser.py               # PDF/Text extraction + structured CandidateProfile parser
│   │   ├── rag_service.py          # Semantic chunker, 1536-dim vector embedding & RAG search
│   │   ├── gap_filler.py           # Weighted completeness engine & LangGraph gap filling agent
│   │   ├── recommender.py          # Hybrid role matching (60% Vector Similarity + 40% Skills)
│   │   ├── assessment.py           # Adaptive state machine scaling difficulty (Level 1 to 5)
│   │   └── scoring.py              # 4D benchmark calibration engine & Phase 2 upskilling roadmap
│   └── api/
│       └── routes.py               # Complete REST API endpoints (/upload -> /report)
├── tests/
│   ├── conftest.py                 # Async test client & SQLite in-memory DB fixtures
│   └── test_pipeline.py            # Automated end-to-end integration test
├── sample_resumes/
│   └── mock_resume.txt             # Clean mock resume text for instant zero-dependency testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md                       # Comprehensive docs + Big-O breakdown explained simply
```

### Module 1: Resume Extraction & Normalization (`parser.py` & `rag_service.py`)
- Extracts raw text from binary PDFs or unformatted text strings.
- Parses unstructured text into a strictly validated `CandidateProfile` Pydantic model (`identity`, `education`, `experience`, `projects`, `skills`, `certifications`).
- Slices candidate experience into semantic chunks, generates 1536-dimensional vector embeddings, and indexes them in `pgvector`.

### Module 2: Deterministic Completeness & Gap-Filling Agent (`gap_filler.py`)
- Calculates initial candidate completeness percentage using weighted category math:
  - **Projects**: 25% | **Experience**: 25% | **Skills & Evidence**: 20% | **Education**: 15% | **Identity**: 15%
- Executes a **LangGraph StateGraph** workflow:
  1. `detect_gaps`: Identifies missing fields (e.g. project metrics or contact emails).
  2. `query_rag_first`: Searches candidate vector chunks to check if context was already provided.
  3. `generate_followups`: Formulates a maximum of 3 single-turn follow-up questions for the candidate.
  4. `ingest_answers_and_patch`: Ingests candidate answers, updates the database profile, and recalculates completeness.

### Module 3: Intelligent Role Recommender (`recommender.py`)
- Matches candidate profile against target tech role taxonomies (*Backend Engineer*, *Data/AI Engineer*, *Full Stack Engineer*, *DevOps Engineer*, *Frontend Engineer*).
- Applies a transparent **Hybrid Score Formula**:
  $$\text{Hybrid Score} = (0.60 \times \text{Vector Semantic Similarity}) + (0.40 \times \text{Required Skill Match Percentage})$$
- Returns the top 5 job recommendations with plain-English rationales, matched skills, and missing skill gaps.

### Module 4: Adaptive Assessment Engine (`assessment.py`)
- Manages live technical interviews using an adaptive state machine:
  - **Correct / Deep Answer**: Increases difficulty (+1 Level, up to Level 5) and asks advanced architectural system design questions.
  - **Partial Answer**: Retains difficulty (Hold level) and probes missing sub-concepts.
  - **Weak / Incorrect Answer**: Decreases difficulty (-1 Level, down to Level 1) and tests core fundamentals.
- Dynamically grounds questions on candidate resume projects and tech stacks.

### Module 5: Benchmark & Diagnostic Report (`scoring.py`)
- Evaluates candidate readiness across 4 core dimensions:
  - *Technical Fundamentals* (30%) | *Role Depth* (30%) | *Problem Solving* (25%) | *Communication Clarity* (15%)
- Assigns calibrated placement tiers: `Foundation`, `Entry-Level Ready`, `Strong Entry-Level`, `Intermediate Potential`.
- Constructs an actionable Phase 2 upskilling roadmap.

---

## 🎓 Big-O Complexity Explained Simply ("Explain Like I'm 13")

In Computer Science, **Big-O Notation** measures how much slower an algorithm gets (Time Complexity $T$) or how much extra memory it needs (Space Complexity $S$) as the input size grows.

### 1. Resume Ingestion & Semantic Chunking
- **Time Complexity**: $O(N \cdot D)$ where $N$ is the total character count of the resume text and $D = 1536$ is the vector embedding dimension.
- **Space Complexity**: $O(K \cdot D)$ where $K$ is the number of text chunks created.
- **Simple Analogy**: *Think of reading a resume like scanning a 500-word book page by page. If the book gets 2x longer, it takes 2x as long to read ($O(N)$ linear time). Turning each page into an index vector takes a fixed 1536-coordinate space ($O(D)$).*

### 2. Profile Completeness Evaluation
- **Time Complexity**: $O(F)$ where $F$ is the fixed number of fields in the Pydantic profile schema graph ($F \approx 15$).
- **Space Complexity**: $O(1)$ constant space.
- **Simple Analogy**: *Think of completeness evaluation like checking off items on a grocery shopping list. Because your list always has the exact same 15 check-boxes (Name, Email, Degree, Projects, Skills), checking off the boxes takes the exact same split-second every single time, no matter who the candidate is ($O(1)$ constant time).*

### 3. Candidate RAG Vector Semantic Search
- **Time Complexity**: $O(\log M)$ using PostgreSQL pgvector HNSW (Hierarchical Navigable Small World) index graphs, where $M$ is total stored candidate chunks.
- **Space Complexity**: $O(M \cdot D)$ to store vectors in index nodes.
- **Simple Analogy**: *Think of pgvector HNSW indexing like an express library index card system! Instead of picking up and reading every single book in a 10-story library one by one ($O(M)$ linear search), you follow express highway signs on index cards that jump straight to the exact bookshelf holding your topic in just a few hops ($O(\log M)$ logarithmic search).*

### 4. Adaptive Assessment State Machine Transition
- **Time Complexity**: $O(1)$ constant state table lookup + $O(L)$ context window token pass to LLM, where $L$ is conversation history length.
- **Space Complexity**: $O(L)$ space storing turn history in database JSON.
- **Simple Analogy**: *Think of updating adaptive assessment levels like keeping score in a video game. Updating your current difficulty checkpoint (Level 1 to Level 5) takes 1 millisecond ($O(1)$). Passing previous interview questions to the LLM requires reading the recent notes saved in your notebook ($O(L)$).*

---

## ⚡ Quickstart Guide

### Option 1: Run Integration Tests Out-of-the-Box
Execute the complete test suite using SQLite in-memory async database (no external PostgreSQL required!):

```bash
cd ai-candidate-platform
pip install -r requirements.txt
pytest tests/test_pipeline.py -v
```

### Option 2: Run Local FastAPI Dev Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open your browser to:
- **Interactive OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/](http://localhost:8000/)

### Option 3: Run with Docker Compose (PostgreSQL + pgvector)
```bash
docker-compose up --build
```
This launches:
1. **PostgreSQL container** with pre-installed `pgvector/pgvector:pg16` extension on port `5432`.
2. **FastAPI Web Application** container listening on port `8000`.
#   c a n d i d a t e _ i n t e l l i g e n t  
 