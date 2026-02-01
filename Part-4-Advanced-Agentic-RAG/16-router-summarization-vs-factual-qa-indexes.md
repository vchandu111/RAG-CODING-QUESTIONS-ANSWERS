# Q16: Router — Summarization vs Factual Q&A Indexes

## The Question

> **Implement a router** that sends **"summarization"** queries to **one index** and **"factual Q&A"** queries to **another**.

---

## Concepts

- **Two use cases**: (1) **Summarization**: "Summarize this doc" or "Give me a summary of X" — you might use an index of **full documents** or long chunks. (2) **Factual Q&A**: "What is the refund policy?" — you want **short, factual chunks** (e.g. small chunks from a FAQ or policy).
- **Router**: A component that **classifies** the user query into a **type** (e.g. "summarization" vs "factual_qa"). Based on the type, we **choose which index** to query (and maybe which retriever or chunk size).
- **Implementation**: (1) **Classifier**: LLM or a small model: "Is this query asking for a summary/overview or a specific fact?" → "summarization" or "factual_qa". (2) **Dispatch**: If summarization → query index A (e.g. doc-level); if factual_qa → query index B (e.g. chunk-level). (3) Return results from the chosen index.

---

## Approach

1. **User query**: e.g. "Summarize the product roadmap" or "What is the refund period?"

2. **Route**: Call a **router** (LLM or rule-based):  
   - "Summarize the product roadmap" → **summarization**  
   - "What is the refund period?" → **factual_qa**

3. **Dispatch**:  
   - **summarization** → use **index A** (e.g. full-doc or large-chunk index); retrieve 1–3 docs/chunks.  
   - **factual_qa** → use **index B** (e.g. small-chunk FAQ index); retrieve 5–10 chunks.

4. **Retrieve** from the chosen index with the same query (or a rewritten query per type if you want).

5. **Generate**: Pass the retrieved context to the LLM. Optionally use a different prompt per type (e.g. "Summarize the following" vs "Answer the question using the following context").

---

## Python Implementation

```python
# Router: classify query -> "summarization" | "factual_qa"; then query the right index.

from typing import Literal
import re

RouteType = Literal["summarization", "factual_qa"]

def route_query_rule_based(query: str) -> RouteType:
    """
    Simple rule-based router: keywords for summarization vs factual.
    In production, use an LLM: "Classify as summarization or factual_qa."
    """
    q = query.lower().strip()
    summarization_keywords = ["summarize", "summary", "overview", "summarise", "recap", "brief"]
    if any(kw in q for kw in summarization_keywords):
        return "summarization"
    return "factual_qa"

def route_query_llm(query: str, llm_fn=None) -> RouteType:
    """Use LLM to classify. llm_fn(query) -> 'summarization' or 'factual_qa'."""
    if llm_fn is None:
        return route_query_rule_based(query)
    response = llm_fn(f"Classify this user query as exactly one of: summarization, factual_qa. Query: {query}")
    return "summarization" if "summarization" in response.lower() else "factual_qa"

def retrieve_from_index(query: str, index_name: str, retriever_fns: dict, top_k: int = 5) -> list[str]:
    """Dispatch to the right retriever by index_name."""
    retriever = retriever_fns.get(index_name)
    if retriever is None:
        return []
    chunks, _ = retriever(query, top_k=top_k)
    return chunks

def rag_with_router(
    query: str,
    retriever_fns: dict,  # {"summarization": fn1, "factual_qa": fn2}
    generate_fn,
    route_fn=None,
    top_k_summary: int = 3,
    top_k_factual: int = 8,
) -> dict:
    """
    Route query -> summarization or factual_qa; retrieve from corresponding index; generate.
    """
    route_fn = route_fn or route_query_rule_based
    route = route_fn(query)
    top_k = top_k_summary if route == "summarization" else top_k_factual
    chunks = retrieve_from_index(query, route, retriever_fns, top_k=top_k)
    answer = generate_fn(query, chunks, route_type=route)
    return {"answer": answer, "route": route, "chunks_used": len(chunks)}


# ---------- Example ----------
if __name__ == "__main__":
    def mock_retriever_summary(q, top_k=5):
        return ["Full document text: Product roadmap 2024 includes A, B, C. Timeline Q1-Q4."], [0.9]

    def mock_retriever_factual(q, top_k=5):
        return [
            "Refund period is 30 days from purchase.",
            "Contact support@example.com for refunds.",
        ], [0.9, 0.8]

    def mock_generate(q, chunks, route_type="factual_qa"):
        return "Answer: " + (chunks[0][:80] + "..." if chunks else "No info.")

    retriever_fns = {"summarization": mock_retriever_summary, "factual_qa": mock_retriever_factual}

    out1 = rag_with_router("Summarize the product roadmap", retriever_fns, mock_generate)
    print("Query: Summarize the product roadmap")
    print("  Route:", out1["route"], "| Chunks:", out1["chunks_used"])

    out2 = rag_with_router("What is the refund period?", retriever_fns, mock_generate)
    print("Query: What is the refund period?")
    print("  Route:", out2["route"], "| Chunks:", out2["chunks_used"])
```

---

## Summary

- **Router** = classify query into **summarization** vs **factual_qa** (LLM or rules).
- **Dispatch**: Summarization → index A (e.g. full docs); factual_qa → index B (e.g. small chunks).
- **Retrieve** from the chosen index; **generate** with the same or a type-specific prompt.
