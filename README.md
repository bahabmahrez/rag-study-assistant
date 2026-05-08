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

```
User Question
     ↓
  Agent (rule-based tool selector)
     ↓
  ┌─────────────────────────────┐
  │ Tool 1: Vector Search       │  → retrieves relevant chunks from ChromaDB
  │ Tool 2: Math Explainer      │  → targets formula/definition queries
  └─────────────────────────────┘
     ↓
  Groq LLM (Llama 3.3 70B)
     ↓
  Answer with sources
     ↓
  Streamlit chat interface
```

### Ingestion pipeline (run once)

```
PDFs → PyMuPDF extraction → Semantic chunking (512 tokens, 50 overlap)
     → Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
     → ChromaDB vector store
```

---

## Project structure

```
rag-study-assistant/
├── data/
│   └── pdfs/                  # Course PDFs (not tracked by git)
├── ingestion/
│   ├── __init__.py
│   ├── extractor.py           # PDF → text extraction with quality check
│   ├── chunker.py             # Text → overlapping chunks with metadata
│   └── embedder.py            # Chunks → vectors stored in ChromaDB
├── agent/
│   ├── __init__.py
│   ├── tools.py               # Vector search and math explainer tools
│   └── agent.py               # Agent logic + Groq LLM integration
├── app.py                     # Streamlit chat interface
├── ingest.py                  # Run this once to build the index
├── test_retrieval.py          # Test retrieval quality
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
| LLM backbone | Groq API — Llama 3.3 70B | Free tier, fast inference |
| Interface | Streamlit | Clean chat UI, full Python |

**100% free stack** — no paid APIs except the free Groq tier.

---

## Prerequisites

- Python **3.11** (required — 3.12+ causes dependency issues with some packages)
- A free [Groq API key](https://console.groq.com)
- Git

---

## Setup & installation

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-study-assistant.git
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

```bash
pip install pymupdf sentence-transformers chromadb langchain langchain-community langchain-text-splitters groq streamlit python-dotenv
```

### 4 — Add your API key

Create a `.env` file at the root of the project:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free key at [console.groq.com](https://console.groq.com).

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

The `TLA_` and `MATH_` prefixes are used for source identification in citations.

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

The first run downloads the embedding model (~470MB). Subsequent runs are fast.

### 7 — Test retrieval (optional)

Before launching the app, verify that retrieval is working:

```bash
python test_retrieval.py
```

This runs two sample queries and shows the top retrieved chunks with their sources.

### 8 — Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Usage

Type any question in French or English related to your course material:

- `Qu'est-ce qu'un automate fini déterministe ?`
- `Comment rendre déterministe un AFN ambigu ?`
- `Définition d'un espace vectoriel`
- `Expliquer le lemme d'Arden`
- `Qu'est-ce qu'une application linéaire ?`

Each answer includes a **Sources** section citing the PDF file and page number the answer was retrieved from.

---

## Known limitations

- **Automata diagrams** — DFA/NFA diagrams embedded as images in the TLA PDFs are not extracted. The system relies on the surrounding text description. Questions that require interpreting a specific diagram visually may return incomplete answers.
- **Definition coverage** — if a definition appears only once on a page that was split across chunk boundaries, retrieval may return properties instead of the base definition. This is documented as a known failure case.
- **No conversation memory** — each question is treated independently. The agent does not remember previous questions in the same session.

---

## How to add new documents

1. Place the new PDF in `data/pdfs/`
2. Re-run `python ingest.py` — it uses `upsert` so existing chunks are not duplicated
3. Restart the Streamlit app

---

## Evaluation

Retrieval was evaluated on 10 course-specific questions across both subjects. Results are documented in the technical report.

A baseline comparison (same questions asked to the LLM without RAG) showed that course-specific questions — particularly those referencing the professor's exact notation and definitions — were answered significantly better with retrieval than without.

---

## Repository

> ⚠️ Course PDFs are not included in this repository for copyright reasons. Add your own PDFs following the naming convention described above.
> 
> ⚠️ The `.env` file containing your API key is excluded from git. Never commit it.
