# Q7: Query Expansion — LLM Generates 3 Variations

## The Question

> Implement a **"query expansion" function** that uses an **LLM to generate 3 variations** of a user query to improve retrieval recall.

---

## Concepts

- **One query** might not match how the answer is written in the docs (e.g. user asks "how to install" but the doc says "installation steps"). So we might miss good chunks.
- **Query expansion**: Create **several versions** of the same question (rephrased, more formal, with synonyms, etc.). Run **retrieval for each** and then **merge** the results (e.g. by RRF or union). So we "cast a wider net" and improve **recall** (find more of the right chunks).
- **Using an LLM**: Ask the LLM: "Given this user question, write 3 different ways to ask the same question." Then use those 3 (+ original) as queries for your retriever.

---

## Approach

1. **User query**  
   e.g. "How do I cancel my subscription?"

2. **Call the LLM** with a short prompt:  
   "Generate exactly 3 alternative phrasings of this question. One per line. No numbering."  
   + user query.  
   Parse the response into 3 strings (and optionally keep the original as a 4th).

3. **Retrieve for each query**  
   For each of the 4 queries, run your retriever (vector or hybrid) and get top‑k doc IDs.

4. **Merge results**  
   - **Union**: Collect all unique doc IDs from the 4 lists.  
   - **RRF**: Treat each list as a ranking and compute RRF scores across the 4 lists, then sort and take top‑k.  
   RRF usually gives better order than simple union.

5. **Return** the merged list (e.g. top 10 by RRF) to pass to the generator.

---

## Python Implementation

```python
# Install: pip install langchain langchain-openai (or use OpenAI client directly)

import os
from typing import Optional

# Option A: using OpenAI API directly
def expand_query_openai(user_query: str, num_variations: int = 3, api_key: Optional[str] = None) -> list[str]:
    """Use OpenAI to generate num_variations alternative phrasings. Returns [original, var1, var2, var3]."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    prompt = f"""Generate exactly {num_variations} alternative phrasings of the following question. Each phrasing should ask the same thing in different words. Write one per line, no numbering.

Question: {user_query}

Alternative phrasings:"""
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    text = resp.choices[0].message.content.strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:num_variations]
    return [user_query] + lines


# Option B: no API — rule-based "expansion" for demo (synonyms, formal/informal)
def expand_query_simple(user_query: str) -> list[str]:
    """Simple expansion without LLM: add one variation (lowercase, no question mark). Good for testing."""
    variations = [user_query]
    low = user_query.lower().strip().rstrip("?")
    if low != user_query:
        variations.append(low + "?")
    return variations


def rrf_merge_rankings(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge multiple ranked lists of doc_ids using RRF. Returns (doc_id, score) sorted by score."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------- Example ----------
if __name__ == "__main__":
    query = "How do I cancel my subscription?"
    # With API (set OPENAI_API_KEY):
    # expanded = expand_query_openai(query, num_variations=3)
    # Without API:
    expanded = expand_query_simple(query)
    expanded.append("What are the steps to cancel my subscription?")
    expanded.append("How can I unsubscribe?")
    print("Queries to run retrieval for:")
    for i, q in enumerate(expanded, 1):
        print(f"  {i}. {q}")

    # Simulate 3 rankings (e.g. from 3 different queries)
    rank1 = ["doc_a", "doc_b", "doc_c"]
    rank2 = ["doc_b", "doc_d", "doc_a"]
    rank3 = ["doc_a", "doc_d", "doc_e"]
    merged = rrf_merge_rankings([rank1, rank2, rank3])
    print("\nMerged RRF top 5:", merged[:5])
```

---

## Summary

- **Query expansion** = generate 3 (or more) phrasings of the user question using an LLM.  
- **Retrieve** for each phrasing; **merge** with RRF (or union) to get one ranked list.  
- **Result**: Better recall because different wordings can hit different chunks.
