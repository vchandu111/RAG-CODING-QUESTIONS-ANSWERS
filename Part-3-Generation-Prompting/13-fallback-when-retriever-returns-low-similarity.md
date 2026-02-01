# Q13: Fallback When Retriever Returns Low Similarity

## The Question

> **Implement a fallback logic** for when the **retriever returns low-similarity scores**. How does the model communicate this **gracefully**?

---

## Concepts

- **Low similarity**: The retriever returns chunks, each with a **score** (e.g. 0.0–1.0). If the **best** score is low (e.g. &lt; 0.5), it often means "no good match" — the docs might not contain the answer.
- **Fallback**: Instead of still sending those weak chunks to the LLM (and risking a wrong or generic answer), we **don’t use them** and instead **tell the user** clearly: "I couldn’t find relevant information in the documents."
- **Graceful**: The model (or your app) responds in a **friendly, clear** way: e.g. "I don’t have enough information in the provided documents to answer that. Try rephrasing or asking about a different topic." No made-up answer, no blame.

---

## Approach

1. **After retrieval**, check the **scores** of the top chunks (e.g. the best score, or the average of top‑k).

2. **Define a threshold** (e.g. `min_score = 0.5`). If **best_score < min_score**, treat it as "no good results."

3. **If below threshold**  
   - **Option A**: Don’t call the LLM with context. Return a **fixed message**: "I couldn’t find relevant information in the documents for this question. Try rephrasing or asking something else."  
   - **Option B**: Call the LLM with a **special prompt**: "The retriever found no highly relevant documents. Reply to the user saying you don’t have enough information in the documents, and suggest they rephrase or ask another question." So the model can phrase it naturally.

4. **If above threshold**  
   Proceed as usual: pass chunks to the LLM and return its answer.

5. **Optional**: Log low-score queries for improving docs or retrieval later.

---

## Python Implementation

```python
# No extra deps for core logic

MIN_SIMILARITY_THRESHOLD = 0.5

FALLBACK_MESSAGE = (
    "I couldn't find relevant information in the provided documents for this question. "
    "You can try rephrasing your question or asking about a different topic."
)

def should_use_retrieved_chunks(scores: list[float], threshold: float = MIN_SIMILARITY_THRESHOLD) -> bool:
    """If the best score is below threshold, we don't use the chunks (fallback)."""
    if not scores:
        return False
    return max(scores) >= threshold

def rag_with_fallback(
    question: str,
    chunks: list[str],
    scores: list[float],
    threshold: float = MIN_SIMILARITY_THRESHOLD,
    generate_fn=None,
) -> dict:
    """
    If best score >= threshold: call generate_fn(question, chunks) and return answer.
    Else: return fallback message and no sources.
    """
    if not should_use_retrieved_chunks(scores, threshold):
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "fallback_used": True,
            "best_score": max(scores) if scores else 0.0,
        }
    # Normal path: call your generator
    if generate_fn is None:
        answer = "(In production, call your LLM here with question + chunks.)"
    else:
        answer = generate_fn(question, chunks)
    return {
        "answer": answer,
        "sources": chunks,
        "fallback_used": False,
        "best_score": max(scores),
    }


# ---------- Example ----------
if __name__ == "__main__":
    # Simulate low scores
    low_scores = [0.3, 0.25, 0.2]
    chunks_low = ["Some unrelated text.", "More unrelated text."]
    out_low = rag_with_fallback("What is the refund policy?", chunks_low, low_scores)
    print("When scores are low:")
    print("  fallback_used:", out_low["fallback_used"])
    print("  answer:", out_low["answer"][:80], "...")

    # Simulate good scores
    good_scores = [0.85, 0.7, 0.6]
    chunks_good = ["Refunds allowed within 30 days. Contact support."]
    out_good = rag_with_fallback("What is the refund policy?", chunks_good, good_scores)
    print("\nWhen scores are good:")
    print("  fallback_used:", out_good["fallback_used"])
    print("  answer:", out_good["answer"][:80], "...")
```

---

## Summary

- **Check** the retriever’s best (or top‑k) score; if **below a threshold**, treat as "no good results."
- **Fallback**: Don’t send weak chunks to the LLM; return a **clear, friendly** message: "I couldn’t find relevant information in the documents. Try rephrasing or another question."
- **Optional**: Use the LLM once with a "no results" prompt to phrase the message naturally; otherwise a fixed string is fine and robust.
