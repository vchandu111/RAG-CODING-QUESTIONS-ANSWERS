# Q6: Rerank with a Cross-Encoder ("Lost in the Middle")

## The Question

> How do you **code a reranking step (using a Cross-Encoder)** to ensure the most relevant context isn't buried in a long prompt? (The "Lost in the Middle" problem)

---

## Concepts

- **Lost in the Middle**: When you put many retrieved chunks into one long prompt, the model often pays more attention to the **start and end** and "loses" the middle. So the **best** chunk might be in the middle and get ignored.
- **Reranking**: After you retrieve e.g. 20 chunks with a vector search, you **score each chunk with the query** using a **Cross-Encoder** (a model that takes query + chunk together and outputs one relevance score). Then you **sort by that score** and keep only the **top 5–10**. So the chunks you put in the prompt are the **most relevant**, not just the first ones.
- **Cross-Encoder vs Bi-Encoder**: Bi-Encoder embeds query and chunk separately (fast, good for retrieval). Cross-Encoder sees query+chunk together (slower, but more accurate). So we use Cross-Encoder only for **reranking** a small set (e.g. 20 → 5).

---

## Approach

1. **Retrieve** a larger set (e.g. top 20 or 30 chunks) with your normal retriever (vector or hybrid).

2. **Load a Cross-Encoder** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face). It takes two strings: `(query, chunk)` and returns a score.

3. **Score each (query, chunk) pair** and collect `(chunk, score)`.

4. **Sort by score descending** and take the top k (e.g. 5 or 10). These are your **reranked** chunks.

5. **Pass only these top chunks** to the LLM. Now the "middle" of your context is still highly relevant, so the model is less likely to miss the best part.

---

## Python Implementation

```python
# Install: pip install sentence-transformers

from sentence_transformers import CrossEncoder

def rerank(query: str, chunks: list[str], top_k: int = 5, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[tuple[str, float]]:
    """
    Rerank chunks by relevance to the query using a Cross-Encoder.
    Returns top_k (chunk, score) sorted by score descending.
    """
    model = CrossEncoder(model_name)
    pairs = [(query, c) for c in chunks]
    scores = model.predict(pairs)
    # Sort by score descending
    indexed = list(zip(chunks, scores))
    indexed.sort(key=lambda x: -x[1])
    return indexed[:top_k]


# ---------- Example: simulate "lost in the middle" then fix with rerank ----------
if __name__ == "__main__":
    query = "What is the refund policy?"
    # Simulate 10 chunks where the BEST one is in the middle (position 4)
    chunks = [
        "Our company was founded in 2010.",
        "We have offices in three countries.",
        "Contact us at support@example.com.",
        "Refunds are available within 30 days. Contact support with your order ID.",  # BEST
        "Our team has 50 employees.",
        "We use Python and JavaScript.",
        "Shipping takes 3-5 business days.",
        "We offer a free trial.",
        "Our mission is to simplify workflows.",
        "Thank you for choosing us.",
    ]

    top = rerank(query, chunks, top_k=3)
    print("After reranking (top 3):")
    for i, (chunk, score) in enumerate(top, 1):
        print(f"  {i}. [{score:.3f}] {chunk[:60]}...")
```

---

## Summary

- **Problem**: Long context → model underuses the middle (Lost in the Middle).  
- **Fix**: Retrieve more, then **rerank** with a Cross-Encoder and keep only top‑k.  
- **Code**: Use `sentence_transformers.CrossEncoder` on `(query, chunk)` pairs, sort by score, take top‑k.
