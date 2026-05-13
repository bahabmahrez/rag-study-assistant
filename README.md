# RAG Study Assistant 📚

An agentic Retrieval-Augmented Generation (RAG) system that answers questions grounded in personal course material for **Théorie des Langages & Automates (TLA)** and **Mathématiques de l'Ingénieur**.

Built as a final project for the AI course at Tek-Up University.

**Group members:** Baha Eddine Ben Mahrez · Mahmoud Ben Moussa

---

## What is this project?

Most AI assistants answer questions from general knowledge. This system is different — it answers questions **strictly from your own course PDFs**. Every answer is grounded in retrieved chunks from the uploaded documents, with source and page number cited.

The system handles two specific technical challenges:
- **Math-heavy content** — formal notation, set theory, transition functions, linear algebra formulas
- **Structured diagrams** — finite automata (DFA/NFA) embedded as images in the TLA course

---

## Architecture

The system went through two versions during development, both accessible via a toggle in the UI.

### v1 — Pipeline (baseline)

```
User Question
     ↓
  Vector Search (fixed, single call)
     ↓
  Groq LLM (Llama 3.3 70B)
     ↓
  Answer with sources
```

A fixed pipeline: one question triggers one retrieval call, the result is passed to the LLM, and the answer is returned. Simple and functional but rigid — the system cannot adapt if the first retrieval is insufficient.

### v2 — ReAct Agent (current)

```
User Question
     ↓
  LLM reasons: which tool do I need?
     ↓
  ┌─────────────────────────────────────┐
  │ Tool 1: vector_search               │  → retrieves relevant chunks from ChromaDB
  │ Tool 2: math_explainer              │  → targets formal notation and formulas
  └─────────────────────────────────────┘
     ↓
  LLM reads result, decides: enough context?
     ↓  (if not → calls another tool)
  Final answer with sources
```

The LLM drives the tool selection autonomously using the **ReAct pattern** (Reason → Act → Observe → repeat). It reads each tool's description, decides which one fits the question, reads the result, and decides whether another search is needed before answering. The pipeline logic is gone — replaced by genuine agentic reasoning.

### Ingestion pipeline (run once, offline)

```
PDFs → PyMuPDF extraction → quality check (math symbol detection)
     → Semantic chunking (512 tokens, 50 token overlap)
     → Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
     → ChromaDB persistent vector store
```

---

## What changed between v1 and v2

| | v1 Pipeline | v2 ReAct Agent |
|---|---|---|
| Tool selection | Manual keyword if/else | LLM decides from tool descriptions |
| Number of retrieval calls | Always exactly 1 | 1 to 4 depending on the question |
| Adaptability | Fixed, brittle | Flexible, semantic |
| Transparency | Silent | Verbose reasoning visible in terminal |
| LangChain pattern | Direct chain | `create_tool_calling_agent` + `AgentExecutor` |

---

## Project structure

```
rag-study-assistant/
├── data/
│   └── pdfs/                  # Course PDFs (not tracked by git)
├── ingestion/
│   ├── __init__.py
│   ├── extractor.py           # PDF → text extraction with math quality check
│   ├── chunker.py             # Text → overlapping chunks with metadata
│   └── embedder.py            # Chunks → vectors stored in ChromaDB
├── agent/
│   ├── __init__.py
│   ├── tools.py               # @tool decorated vector_search and math_explainer
│   └── agent.py               # ReAct agent with AgentExecutor
├── app.py                     # Streamlit UI with v1/v2 toggle
├── ingest.py                  # Run once to build the ChromaDB index
├── test_retrieval.py          # Retrieval quality testing
├── .env                       # API keys (not tracked by git)
├── .gitignore
└── README.md
```

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| PDF extraction | PyMuPDF | Fast, handles French text and formal notation cleanly |
| Embeddings | sentence-transformers (multilingual MiniLM) | Free, local, supports French |
| Vector store | ChromaDB | Local, persistent, no infrastructure needed |
| Agent framework | LangChain 0.2.16 | Tool-calling agent, ReAct loop |
| LLM backbone | Groq API — Llama 3.3 70B | Free tier, fast inference |
| Interface | Streamlit | Clean chat UI with v1/v2 mode toggle |

**100% free stack** — no paid APIs except the free Groq tier.

---

## Prerequisites

- Python **3.11** (required — 3.12+ causes wheel build failures for several dependencies on Windows)
- A free [Groq API key](https://console.groq.com)
- Git

---

## Setup & installation

### 1 — Clone the repository

```bash
git clone https://github.com/bahabmahrez/rag-study-assistant.git
cd rag-study-assistant
```

### 2 — Create a virtual environment with Python 3.11

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

Use these exact pinned versions to avoid compatibility issues:

```bash
pip install pymupdf sentence-transformers chromadb groq streamlit python-dotenv \
    langchain==0.2.16 \
    langchain-core==0.2.38 \
    langchain-groq==0.1.9 \
    langchain-community==0.2.16 \
    langchain-text-splitters==0.2.4 \
    langgraph==0.2.16
```

> ⚠️ LangChain's tool-calling API changed significantly between versions. Do not upgrade these packages without testing — later versions may break the agent.

### 4 — Add your API key

Create a `.env` file at the root of the project:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free key at [console.groq.com](https://console.groq.com). Never commit this file.

### 5 — Add your course PDFs

Place your PDF files in `data/pdfs/`. Use a clear naming convention:

```
data/pdfs/
├── TLA_chap1_mots_et_langages.pdf
├── TLA_chap2_automates_finis.pdf
├── MATH_chap1_groupes.pdf
├── MATH_chap2_espaces_vectoriels.pdf
└── ...
```

The `TLA_` and `MATH_` prefixes are stored as metadata on every chunk and appear in source citations.

### 6 — Build the index

Run this once to extract, chunk, embed and store all your PDFs:

```bash
python ingest.py
```

Expected output:
```
=== RAG Study Assistant — Ingestion Pipeline ===

Extracting: data\pdfs\TLA_chap2_Automates_finis_.pdf
  → 72 pages, 0 flagged
...
Total: 231 pages from 22 PDFs
Created 377 chunks from 231 pages
Embedding 377 chunks...
Stored 377 chunks in ChromaDB

Ingestion complete.
```

The first run downloads the embedding model (~470MB). Subsequent runs reuse the cached model and are fast. The `chroma_db/` folder created here is your persistent knowledge base.

### 7 — Test retrieval (optional but recommended)

```bash
python test_retrieval.py
```

Runs two sample queries and prints the top retrieved chunks with source and page number. Use this to verify the index is working before launching the app.

### 8 — Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Usage

The app has a mode toggle at the top:

- **🔗 Pipeline (v1)** — original fixed pipeline, one retrieval call per question
- **🤖 Agent ReAct (v2)** — LLM-driven agent, multiple tool calls when needed

Each mode maintains its own independent chat history so you can compare answers to the same question side by side.

Example questions:

- `Qu'est-ce qu'un automate fini déterministe ?`
- `Comment rendre déterministe un AFN ambigu ?`
- `Explique la différence entre un AFN et un AFD`
- `Définition d'un espace vectoriel`
- `Expliquer le lemme d'Arden`
- `Soit (A, +, ×) un anneau, montrer que le centre C est un sous-anneau`

Each answer cites the source PDF and page number it was retrieved from.

---

## How the ReAct agent works internally

When you ask a question in v2 mode, the terminal shows the agent's reasoning in real time:

```
> Entering new AgentExecutor chain...

Invoking: vector_search with {'question': 'AFN vs AFD méthode déterminisation'}
→ [retrieves chunks from TLA_chap2...]

Invoking: math_explainer with {'expression': 'AFN et AFD'}
→ [retrieves formal definition chunks...]

La différence entre un AFN et un AFD est...

> Finished chain.
```

The agent called two tools because one retrieval was not sufficient. This behavior is decided by the LLM at runtime — not hardcoded.

---

## Known limitations

- **Automata diagrams** — DFA/NFA diagrams embedded as images in the TLA PDFs are not extracted. The system relies on the surrounding text. Questions requiring visual diagram interpretation may return incomplete answers. A vision-based tool using Groq's multimodal model is planned as a bonus extension.
- **Chunk boundary splits** — if a definition spans a page split, retrieval may return properties instead of the base definition. Documented as a known failure case in the technical report.
- **No conversation memory** — each question is independent. The agent does not remember previous questions in the same session.
- **Diagram-only pages** — pages that contain only an image with no surrounding text produce empty chunks and are silently skipped.

---

## How to add new documents

1. Place the new PDF in `data/pdfs/`
2. Re-run `python ingest.py` — uses `upsert` so existing chunks are not duplicated
3. Restart the Streamlit app

---

## Evaluation

Retrieval was tested on course-specific questions across both subjects. A baseline comparison (same questions to the LLM without RAG) showed that questions referencing the professor's exact notation, definitions, and exercise corrections were answered significantly better with retrieval than without — particularly for TLA formal proofs and linear algebra demonstrations.

---

## Repository

> ⚠️ Course PDFs are not included in this repository. Add your own PDFs to `data/pdfs/` following the naming convention above.
>
> ⚠️ The `.env` file is excluded from git. Never commit your API key.
