# 20 RAG Coding Interview Questions for 2026

Coding answers with **explanations** and **Python examples** for common RAG (Retrieval-Augmented Generation) interview questions—from ingestion and retrieval to generation and advanced agentic flows.

---

## What’s in this repo
- **Part-wise folders** (Part 1–4), each with:
  - **README.md** — Index of that part’s questions with links to answer files.
  - **One `.md` per question** — Filename: `{number}-{slug}.md`. Each file has the question, explanation, and runnable Python code.

Use the tables below to jump to any question.

---

## Part 1: Data Ingestion & Chunking

| # | Question | Link |
|---|----------|------|
| 1 | Semantic chunking: split by similarity, not fixed length | [Part 1](./Part-1-Data-Ingestion-Chunking/) → `01-semantic-chunking-split-by-similarity-not-fixed-length.md` |
| 2 | Extract and represent tables from PDF (searchable) | [Part 1](./Part-1-Data-Ingestion-Chunking/) → `02-extract-and-represent-tables-from-pdf-searchable.md` |
| 3 | Recursive splitter: implement and why better than fixed-size? | [Part 1](./Part-1-Data-Ingestion-Chunking/) → `03-recursive-splitter-implement-and-why-better-than-fixed-size.md` |
| 4 | Append parent doc IDs and summary tags to chunks | [Part 1](./Part-1-Data-Ingestion-Chunking/) → `04-append-parent-doc-ids-and-summary-tags-to-chunks.md` |

---

## Part 2: Retrieval & Vector Databases

| # | Question | Link |
|---|----------|------|
| 5 | Combine BM25 and vector search with RRF | [Part 2](./Part-2-Retrieval-Vector-Databases/) → `05-combine-bm25-and-vector-search-with-rrf.md` |
| 6 | Rerank with Cross-Encoder ("Lost in the Middle") | [Part 2](./Part-2-Retrieval-Vector-Databases/) → `06-rerank-with-cross-encoder-lost-in-the-middle.md` |
| 7 | Query expansion: LLM generates 3 variations | [Part 2](./Part-2-Retrieval-Vector-Databases/) → `07-query-expansion-llm-generates-3-variations.md` |
| 8 | Metadata filtering for "read" permissions | [Part 2](./Part-2-Retrieval-Vector-Databases/) → `08-metadata-filtering-for-read-permissions.md` |
| 9 | Benchmark two embedding models | [Part 2](./Part-2-Retrieval-Vector-Databases/) → `09-benchmark-two-embedding-models.md` |

---

## Part 3: Generation & Prompting

| # | Question | Link |
|---|----------|------|
| 10 | System prompt: no internal knowledge without context | [Part 3](./Part-3-Generation-Prompting/) → `10-system-prompt-no-internal-knowledge-without-context.md` |
| 11 | Return answer + source IDs and snippets (citations) | [Part 3](./Part-3-Generation-Prompting/) → `11-return-answer-plus-source-ids-and-snippets.md` |
| 12 | Summarize retrieved docs before the generator | [Part 3](./Part-3-Generation-Prompting/) → `12-summarize-retrieved-docs-before-generator.md` |
| 13 | Fallback when retriever returns low similarity | [Part 3](./Part-3-Generation-Prompting/) → `13-fallback-when-retriever-returns-low-similarity.md` |

---

## Part 4: Advanced & Agentic RAG

| # | Question | Link |
|---|----------|------|
| 14 | Agent critiques retrieval and re-searches if insufficient | [Part 4](./Part-4-Advanced-Agentic-RAG/) → `14-agent-critiques-retrieval-and-re-searches-if-insufficient.md` |
| 15 | Multi-hop retrieval: "CEO of company that acquired Figma?" | [Part 4](./Part-4-Advanced-Agentic-RAG/) → `15-multi-hop-retrieval-ceo-of-company-that-acquired-figma.md` |
| 16 | Router: summarization vs factual Q&A indexes | [Part 4](./Part-4-Advanced-Agentic-RAG/) → `16-router-summarization-vs-factual-qa-indexes.md` |

---

## Requirements (Python)

Use a virtual environment, then install as needed for the questions you run:

```bash
# Core (most answers)
pip install sentence-transformers rank-bm25 numpy

# PDF (Q2)
pip install pypdf2 pdfplumber

# LangChain, agents, summarization (Q7, Q12, Q14)
pip install langchain langchain-openai

# Vector DB (Q8)
pip install chromadb
```

Add any other packages mentioned in the individual answer files.
