# Q11: Return Answer + Source IDs and Snippets (Citations)

## The Question

> **How do you modify your generator** to return not just an answer, but also the **specific source IDs and snippets** it used?

---

## Concepts

- **Citations**: The user (and the system) need to know **which chunks** the model used to produce the answer. So we return: **answer** + **list of (source_id, snippet)**.
- **Source ID**: A unique ID for each retrieved chunk (e.g. `doc_3_chunk_1` or a path). We pass these to the LLM with the chunk text so it can "name" them.
- **Snippet**: The exact text (or a short slice) of the chunk. Shown to the user as "Source 1: …".
- **Two ways**: (1) Ask the LLM to output structured text (e.g. "Answer: … Sources: [1], [2]") and parse it. (2) Return the **same** chunks we sent as context, labeled by ID, and show them as sources (no need for the model to list them if we always attach the chunks we used).

---

## Approach

1. **Retrieve** chunks and keep **ids** and **text**: e.g. `[(id1, text1), (id2, text2)]`.

2. **Build the prompt** so the model sees chunks **with labels**:  
   "Context: [1] text1 ... [2] text2 ... Question: ..."

3. **Option A — Model cites**: Ask the model to end with "Sources: [1], [2]". Parse the response to extract answer and source IDs. Map IDs back to (id, snippet) and return `{ "answer": "...", "sources": [(id1, snippet1), ...] }`.

4. **Option B — We attach sources**: We don’t ask the model to list sources. We always return the **same** chunks we sent as context as "sources", with their IDs and snippets. Simpler and robust.

5. **Return** a structured object: `answer` (string) and `sources` (list of `{ "id": "...", "snippet": "..." }`).

---

## Python Implementation

```python
# Option B: we pass chunks with IDs; we return those same chunks as sources (no parsing needed)

def build_context_with_ids(chunks: list[tuple[str, str]]) -> str:
    """chunks = [(source_id, text), ...]. Build context string with [1], [2] labels."""
    lines = []
    for i, (sid, text) in enumerate(chunks, 1):
        lines.append(f"[{i}] (id={sid})\n{text}")
    return "\n\n".join(lines)

CITATION_SYSTEM_PROMPT = """You answer using only the provided context. Each context block is labeled with [1], [2], etc.
At the end of your answer, list which blocks you used, e.g. "Sources: [1], [3]."
If the answer is not in the context, say "I cannot find this in the provided documents." """

def build_user_message_with_sources(question: str, chunks: list[tuple[str, str]]) -> str:
    context = build_context_with_ids(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (and list Sources: [N], ... at the end):"

def parse_answer_and_sources(response: str, chunks: list[tuple[str, str]]) -> dict:
    """
    Parse "Answer ... Sources: [1], [2]" from response.
    Returns {"answer": "...", "sources": [(id, snippet), ...]}
    """
    response = response.strip()
    sources = []
    if "Sources:" in response:
        parts = response.split("Sources:", 1)
        answer_part = parts[0].strip()
        ref_part = parts[1].strip()
        # Parse [1], [2] or [1, 2]
        import re
        indices = re.findall(r"\[(\d+)\]", ref_part)
        for idx_str in indices:
            i = int(idx_str)
            if 1 <= i <= len(chunks):
                sid, snippet = chunks[i - 1]
                sources.append({"id": sid, "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet})
        return {"answer": answer_part, "sources": sources}
    return {"answer": response, "sources": []}


# ---------- Simpler: always return the chunks we sent as sources (no parsing) ----------
def generate_with_sources_simple(question: str, chunks: list[tuple[str, str]]) -> dict:
    """
    Assume you call an LLM with build_user_message_with_sources(question, chunks).
    Regardless of what the LLM returns, we attach the chunks we used as sources.
    So we always return: answer (from LLM) + sources = the chunks we passed.
    """
    # In real code: answer = llm.generate(...)
    answer = "(In production, this would be the LLM response.)"
    sources = [{"id": sid, "snippet": text[:200] + "..." if len(text) > 200 else text} for sid, text in chunks]
    return {"answer": answer, "sources": sources}


# ---------- Example ----------
if __name__ == "__main__":
    chunks = [
        ("doc1_chunk0", "Refunds are allowed within 30 days. Contact support."),
        ("doc2_chunk1", "Shipping takes 3-5 business days."),
    ]
    user_msg = build_user_message_with_sources("What is the refund policy?", chunks)
    print(user_msg[:400], "...")
    out = generate_with_sources_simple("What is the refund policy?", chunks)
    print("\nResult:", out)
```

---

## Summary

- **Pass** chunks to the LLM with **labels** (e.g. [1], [2]) and keep **(id, text)** for each.
- **Either** ask the model to output "Sources: [1], [2]" and parse, **or** (simpler) always return the **same** chunks you sent as the "sources" list with their IDs and snippets.
- **Return** `{ "answer": "...", "sources": [{"id": "...", "snippet": "..."}] }` so the UI can show citations.
