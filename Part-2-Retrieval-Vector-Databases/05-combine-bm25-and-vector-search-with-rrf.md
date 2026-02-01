# Q5: Combine BM25 + Vector Search with Reciprocal Rank Fusion (RRF)

## The Question

> Write a **Python snippet** that combines **BM25 (keyword)** and **Vector (semantic)** search results using **Reciprocal Rank Fusion (RRF)**.

---

## Concepts

- **Keyword search (BM25)**: Ranks documents by how often query words appear (and how rare they are). Good for exact terms (e.g. "Python", "API").
- **Vector (semantic) search**: Ranks by meaning (embeddings). Good for paraphrases (e.g. "how to install" vs "installation steps").
- **Hybrid**: Run both, then **merge** the two ranked lists so we get the best of both.
- **RRF (Reciprocal Rank Fusion)**: A simple way to merge. For each document, add a score = `1 / (k + rank)` where `rank` is its position in each list (1st, 2nd, …). Sum over both lists. Higher total → better. Typical `k = 60`.

---

## Approach

1. **Index your documents**  
   - For BM25: build a vocabulary and term frequencies (e.g. with `rank_bm25`).  
   - For vector: embed each doc and store in a list (or vector DB).

2. **At query time**  
   - Run **BM25**: get a list of doc IDs (or indices) sorted by BM25 score, e.g. `[doc_3, doc_1, doc_5]`.  
   - Run **vector search**: embed query, find top‑k nearest docs, e.g. `[doc_1, doc_3, doc_7]`.

3. **RRF score**  
   - For each document, from BM25 list: `score += 1 / (60 + rank_bm25)`.  
   - From vector list: `score += 1 / (60 + rank_vector)`.  
   - Sum these. Sort by total score descending.

4. **Return** the merged list (e.g. top 10 by RRF score).

---

## Python Implementation

```python
# Install: pip install rank-bm25 sentence-transformers numpy

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

def build_bm25(docs: list[str]):
    """Build BM25 index from list of document strings."""
    tokenized = [d.lower().split() for d in docs]
    return BM25Okapi(tokenized)

def bm25_top_k(bm25, docs: list[str], query: str, k: int = 10) -> list[tuple[int, float]]:
    """Return top-k (index, score) by BM25."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_indices]

def vector_top_k(embeddings: np.ndarray, query_embedding: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
    """Return top-k (index, similarity) by cosine similarity."""
    sims = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_indices]

def rrf_merge(
    bm25_list: list[tuple[int, float]],  # (doc_id, score)
    vector_list: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.
    Score for doc = sum over both lists: 1 / (k + rank).
    Returns list of (doc_id, rrf_score) sorted by rrf_score descending.
    """
    scores = {}
    for rank, (doc_id, _) in enumerate(bm25_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, (doc_id, _) in enumerate(vector_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    sorted_ids = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_ids

def hybrid_search_rrf(
    query: str,
    docs: list[str],
    bm25,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """Run BM25 + vector search, merge with RRF, return top_k (doc_index, rrf_score)."""
    bm25_top = bm25_top_k(bm25, docs, query, k=top_k * 2)  # get a bit more
    query_emb = model.encode(query, convert_to_numpy=True)
    vector_top = vector_top_k(doc_embeddings, query_emb, k=top_k * 2)
    merged = rrf_merge(bm25_top, vector_top, k=rrf_k)
    return merged[:top_k]


# ---------- Example ----------
if __name__ == "__main__":
    docs = [
        "Python is a programming language used for data science.",
        "The API endpoint for users is GET /users.",
        "How to install Python on Windows and Mac.",
        "REST API design best practices and status codes.",
    ]
    query = "Python installation guide"

    bm25 = build_bm25(docs)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(docs, convert_to_numpy=True)

    results = hybrid_search_rrf(query, docs, bm25, model, doc_embeddings, top_k=3)
    print("Hybrid RRF top 3:")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"  {rank}. [doc {idx}] {score:.4f} — {docs[idx][:50]}...")
```

---

## Summary

- **BM25** = keyword ranking; **vector** = meaning ranking.  
- **RRF**: For each document, add `1/(k+rank)` from each list; sort by total.  
- **Python**: Use `rank_bm25` for BM25, `sentence-transformers` for embeddings, then merge with the RRF formula above.
