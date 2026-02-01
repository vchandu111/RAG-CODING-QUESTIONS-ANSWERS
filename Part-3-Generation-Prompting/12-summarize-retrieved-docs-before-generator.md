# Q12: Summarize Retrieved Docs Before the Generator (Context Compression)

## The Question

> **Code a logic** that **summarizes retrieved documents** *before* passing them to the generator to save on **token costs** and **latency**.

---

## Concepts

- **Problem**: We retrieve 5–10 long chunks and paste them into the prompt. That uses many tokens and can be slow. The model also may "lose" the middle (Lost in the Middle).
- **Context compression**: **Summarize** each chunk (or the whole set) with a small/fast model (or an LLM) *before* sending to the main generator. So we send **short summaries** instead of full text → fewer tokens, faster, and often similar quality.
- **Ways to do it**: (1) Summarize each chunk separately. (2) Summarize the concatenation of all chunks in one go. (3) Use a small "compressor" model that keeps only the most relevant sentences.

---

## Approach

1. **Retrieve** as usual: get e.g. top 10 chunks (each can be 200–500 tokens).

2. **Compress**  
   - **Per-chunk**: For each chunk, call a summarizer: "In 1–2 sentences, summarize this for answering user questions." Append the short summary to a list.  
   - **Or one-shot**: Concatenate all chunks, then one call: "Summarize the following in a short paragraph, keeping only information relevant to answering questions."  
   Use the **summaries** as the context string.

3. **Build the prompt** for the main LLM using the **compressed** context (summaries) instead of raw chunks.

4. **Optional**: Keep a mapping (summary → original chunk IDs) so you can still cite the original sources in the final answer.

5. **Result**: Fewer tokens → lower cost and lower latency; quality may stay similar if the summaries keep the key facts.

---

## Python Implementation

```python
# Install: pip install langchain langchain-openai (or use OpenAI client)

import os
from typing import Optional

def summarize_chunks_with_llm(chunks: list[str], max_sentences_per_chunk: int = 2, api_key: Optional[str] = None) -> list[str]:
    """
    Summarize each chunk in 1-2 sentences. Returns list of summaries.
    Uses OpenAI; you can replace with a local/small model.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize the following text in at most {max_sentences_per_chunk} sentences. Keep only facts useful for answering user questions.

Text:
{chunk[:2000]}

Summary:"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        summary = resp.choices[0].message.content.strip()
        summaries.append(summary)
    return summaries

def compress_context_single_call(chunks: list[str], api_key: Optional[str] = None) -> str:
    """Compress all chunks in one LLM call: one short paragraph keeping key facts."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    combined = "\n\n---\n\n".join([f"[{i+1}] {c[:1500]}" for i, c in enumerate(chunks)])
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    prompt = f"""Condense the following passages into one short paragraph. Keep only information that would help answer user questions. Do not add new information.

Passages:
{combined}

Condensed paragraph:"""
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ---------- No-API fallback: truncate by sentence (simple "compression") ----------
def compress_truncate(chunks: list[str], max_sentences: int = 3) -> list[str]:
    """Simple compression: keep first N sentences per chunk. No API."""
    result = []
    for c in chunks:
        sents = [s.strip() for s in c.replace("?", ".").replace("!", ".").split(".") if s.strip()][:max_sentences]
        result.append(". ".join(sents) + ("." if sents else ""))
    return result


# ---------- Example ----------
if __name__ == "__main__":
    chunks = [
        "Python is a programming language. It is used for data science and ML. Many companies use Python.",
        "Refunds are allowed within 30 days. Contact support@example.com with your order ID. No refunds after 30 days.",
    ]
    compressed = compress_truncate(chunks, max_sentences=2)
    print("Compressed (truncate):")
    for i, c in enumerate(compressed, 1):
        print(f"  {i}. {c}")
    print("\nToken saving: shorter text = fewer tokens when sent to the main LLM.")
```

---

## Summary

- **Before** sending retrieved chunks to the generator, **summarize** them (per-chunk or one combined call) or **truncate** (e.g. first N sentences).
- **Send** the compressed context to the main LLM → **fewer tokens**, **lower cost**, **faster**; optionally keep chunk IDs for citations.
