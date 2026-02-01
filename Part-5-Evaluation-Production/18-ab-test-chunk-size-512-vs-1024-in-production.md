# Q18: A/B Test Chunk Size (512 vs 1024) in Production

## The Question

> **How would you architect a system** to test a change in **chunk size** (e.g., **512 vs 1024**) in a **live production** environment?

---

## Concepts

- **A/B test**: Some users get **variant A** (e.g. chunk size 512), others get **variant B** (e.g. 1024). We compare metrics (e.g. user satisfaction, answer quality, latency) to see which is better.
- **Chunk size**: When we split documents, we can use 512 tokens per chunk or 1024. Different sizes change retrieval quality and cost. We want to **test** this in production without breaking the app.
- **Architecture**: (1) **Two indexes** (or one index with metadata "chunk_size": 512 or 1024). (2) **Router**: For each request, assign user/session to A or B (e.g. 50/50). (3) **Query** the right index per variant. (4) **Log** variant + outcome (e.g. thumbs up/down, latency, RAGAS score if available). (5) **Analyze** after enough traffic: compare metrics for A vs B.

---

## Approach

1. **Build two pipelines (or one parameterized)**  
   - **Variant A**: Chunk with size 512; build index A (or tag chunks with `chunk_size=512`).  
   - **Variant B**: Chunk with size 1024; build index B (or tag with `chunk_size=1024`).

2. **At request time**:  
   - **Assign** the user/session/request to A or B (e.g. hash(user_id) % 2, or use an A/B platform).  
   - **Retrieve** from the index (or with filter) for that variant.  
   - **Generate** answer as usual.  
   - **Log**: variant (A/B), request_id, latency, and optionally feedback (thumbs up/down) or RAGAS score.

3. **Metrics**:  
   - **Primary**: e.g. user satisfaction (thumbs up rate), or RAGAS faithfulness/relevance if you compute it.  
   - **Secondary**: latency, token usage.

4. **Analysis**: After N requests per variant, compare mean satisfaction (or mean RAGAS) for A vs B; check statistical significance (e.g. t-test or A/B platform).  
5. **Ship** the winning variant (e.g. make 1024 the default) and remove the other.

---

## Python Implementation (Conceptual)

```python
# No extra deps for core logic; A/B assignment and logging are conceptual.

import hashlib
from typing import Literal
from enum import Enum

Variant = Literal["A", "B"]
CHUNK_SIZE_A = 512
CHUNK_SIZE_B = 1024

def assign_variant(user_id: str, seed: str = "rag_chunk_ab") -> Variant:
    """Deterministically assign user to A or B (50/50)."""
    h = hashlib.sha256(f"{seed}_{user_id}".encode()).hexdigest()
    return "A" if int(h, 16) % 2 == 0 else "B"

def get_chunk_size_for_variant(variant: Variant) -> int:
    return CHUNK_SIZE_A if variant == "A" else CHUNK_SIZE_B

def rag_request_with_ab(
    user_id: str,
    question: str,
    index_a_retriever,
    index_b_retriever,
    generate_fn,
    log_fn=None,
) -> dict:
    """
    Assign user to A or B; retrieve from corresponding index; generate; log variant + outcome.
    """
    variant = assign_variant(user_id)
    retriever = index_a_retriever if variant == "A" else index_b_retriever
    chunks, scores = retriever(question, top_k=5)
    answer = generate_fn(question, chunks)
    # Log for later analysis
    if log_fn:
        log_fn({
            "user_id": user_id,
            "variant": variant,
            "question": question[:100],
            "num_chunks": len(chunks),
            "answer_length": len(answer),
            # Add latency, feedback when available
        })
    return {"answer": answer, "variant": variant, "chunks": chunks}


# ---------- Example: two "indexes" as simple functions ----------
if __name__ == "__main__":
    def index_a_retriever(q, top_k=5):
        # Simulate: index built with chunk size 512
        return ["Chunk A1 (512 tokens).", "Chunk A2."], [0.9, 0.8]

    def index_b_retriever(q, top_k=5):
        # Simulate: index built with chunk size 1024
        return ["Chunk B1 (1024 tokens, longer).", "Chunk B2."], [0.85, 0.75]

    def generate_fn(q, chunks):
        return "Answer based on " + (chunks[0][:30] + "..." if chunks else "no chunks.")

    logs = []
    def log_fn(record):
        logs.append(record)

    out1 = rag_request_with_ab("user_1", "What is X?", index_a_retriever, index_b_retriever, generate_fn, log_fn)
    out2 = rag_request_with_ab("user_2", "What is Y?", index_a_retriever, index_b_retriever, generate_fn, log_fn)
    print("User 1 variant:", out1["variant"], "| User 2 variant:", out2["variant"])
    print("Logs:", logs)
```

---

## Summary

- **Two variants**: Index A (chunk 512) and Index B (chunk 1024).  
- **Assign** each user/request to A or B (e.g. hash-based 50/50).  
- **Retrieve** from the assigned index; **generate**; **log** variant + metrics (satisfaction, latency, etc.).  
- **Analyze** A vs B and ship the better chunk size.
