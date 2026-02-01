# Q9: Benchmark Two Embedding Models

## The Question

> If your **retrieval accuracy is low**, how do you **programmatically benchmark two different embedding models**?

---

## Concepts

- **Retrieval accuracy**: Do the chunks we retrieve actually contain the answer? We measure this with a **eval set**: (question, list of doc IDs that *should* be in the top‑k). For each question we run retrieval and check: how many of the "relevant" docs are in our top‑k? (e.g. **Recall@k** or **MRR**.)
- **Embedding model**: The model that turns text into vectors. Different models (e.g. `all-MiniLM-L6-v2` vs `all-mpnet-base-v2`) give different rankings, so retrieval accuracy can change.
- **Benchmark**: Run the **same** eval set with **model A** and **model B** (same chunks, same queries). Compare **Recall@k** or **MRR**. Pick the model with higher scores.

---

## Approach

1. **Build an eval set**  
   - List of (query, relevant_doc_ids).  
   - Example: 50 questions; for each, you have the 1–3 doc IDs that should be retrieved (from human judgment or a gold set).

2. **Index your corpus once** (or per model):  
   - Chunk documents, embed with **model A**, store in a simple in-memory list (or vector DB).  
   - Same chunks, embed with **model B**, store separately.

3. **For each query in the eval set**  
   - Get top‑k doc IDs with **model A** and with **model B**.  
   - Compute **Recall@k** = (number of relevant docs in top‑k) / (total relevant docs).  
   - Or **MRR** = 1 / rank of first relevant doc (higher is better).

4. **Average** over all queries: e.g. **Recall@10 (model A)** vs **Recall@10 (model B)**.  
   - The model with higher average is better for your data (for that k).

5. **Repeat for different k** (e.g. 5, 10, 20) if you want a fuller picture.

---

## Python Implementation

```python
# Install: pip install sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer

def get_top_k_indices(query_emb: np.ndarray, doc_embeddings: np.ndarray, k: int = 10) -> list[int]:
    """Return indices of top-k docs by cosine similarity."""
    sims = np.dot(doc_embeddings, query_emb) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    return np.argsort(sims)[::-1][:k].tolist()

def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Recall@k: how many relevant docs are in the top-k?"""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)

def evaluate_model(
    model: SentenceTransformer,
    queries: list[str],
    documents: list[str],
    eval_set: list[tuple[int, set[int]]],  # (query_idx, relevant_doc_indices)
    k: int = 10,
) -> float:
    """Embed docs and queries, then compute average Recall@k."""
    doc_embs = model.encode(documents, convert_to_numpy=True)
    recall_sum = 0.0
    for query_idx, relevant in eval_set:
        q_emb = model.encode(queries[query_idx], convert_to_numpy=True)
        top = get_top_k_indices(q_emb, doc_embs, k=k)
        recall_sum += recall_at_k(top, relevant, k)
    return recall_sum / len(eval_set) if eval_set else 0.0


# ---------- Example ----------
if __name__ == "__main__":
    documents = [
        "Python is a programming language.",
        "Java is used for enterprise applications.",
        "Python installation on Windows and Mac.",
        "How to install Python using pip.",
        "JavaScript runs in the browser.",
    ]
    queries = [
        "How do I install Python?",
        "What is Python?",
    ]
    # eval_set: (query index, set of relevant doc indices)
    eval_set = [
        (0, {2, 3}),   # query 0 relevant to doc 2, 3
        (1, {0}),      # query 1 relevant to doc 0
    ]

    model_a = SentenceTransformer("all-MiniLM-L6-v2")
    model_b = SentenceTransformer("all-mpnet-base-v2")

    rec_a = evaluate_model(model_a, queries, documents, eval_set, k=3)
    rec_b = evaluate_model(model_b, queries, documents, eval_set, k=3)

    print(f"Recall@3 Model A (MiniLM): {rec_a:.3f}")
    print(f"Recall@3 Model B (mpnet): {rec_b:.3f}")
    print("Better:", "Model A" if rec_a >= rec_b else "Model B")
```

---

## Summary

- **Eval set**: (query, set of relevant doc IDs) for many queries.  
- **Benchmark**: Embed corpus + queries with model A and B; for each query get top‑k and compute **Recall@k** (or MRR).  
- **Compare** average Recall@k; choose the model with higher score for your use case.
