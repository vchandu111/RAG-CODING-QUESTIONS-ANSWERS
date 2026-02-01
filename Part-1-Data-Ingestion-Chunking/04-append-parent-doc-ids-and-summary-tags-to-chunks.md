# Q4: Append Parent Document IDs and Summary Tags to Chunks

## The Question

> **Write code to append parent document IDs and summary tags to child chunks.** How does this help during the retrieval phase?

---

## Concepts

- **Chunks** = small pieces of a larger document. Each chunk is a "child"; the original doc is the "parent."
- **Parent document ID**: A unique ID for the source document (e.g. `doc_001`, or a path). Stored with **every chunk** that came from that doc.
- **Summary tags**: Short labels for the chunk (or the section), e.g. "introduction", "pricing", "API reference". Can be from a small model or rules.
- **Why it helps at retrieval time**:  
  - **Filtering**: "Only return chunks from document X" or "only from section Y."  
  - **Deduplication**: If 5 chunks come from the same parent, you can show the parent once or fetch the full doc.  
  - **Ranking / UX**: Sort by doc, or show "From: Document A, Section: Pricing."

---

## Approach

1. **Split the document into chunks** (e.g. with a recursive or semantic splitter). Each chunk is a string.

2. **Assign a parent document ID** to the document (e.g. filename, UUID, or `doc_123`). Every chunk from this doc gets the same `parent_id` in metadata.

3. **Optional: add section/summary tags per chunk**  
   - **Rule-based**: First chunk → "intro", last → "conclusion"; or use headings if you parsed them.  
   - **Model-based**: Call a small LLM/summarizer: "In one phrase, what is this chunk about?" and store that as a tag.

4. **Build a list (or DB rows)** where each item has:  
   `{ "chunk_text": "...", "parent_id": "...", "tags": ["summary phrase", "section"] }`.

5. **At retrieval time**:  
   - Store this metadata in your vector DB (e.g. Chroma, Pinecone) so you can **filter** by `parent_id` or `tags`.  
   - After retrieval, you can group by `parent_id`, or only return chunks where `tags` contains "pricing", etc.

---

## Python Implementation

```python
import uuid
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnrichedChunk:
    chunk_text: str
    parent_id: str
    tags: list[str]
    chunk_index: int  # order within parent

def add_parent_id_and_tags(
    chunks: list[str],
    parent_id: Optional[str] = None,
    tag_fn=None,
) -> list[EnrichedChunk]:
    """
    Give each chunk a parent_id and optional tags.
    tag_fn: optional function(chunk_text, index) -> list[str]
    """
    pid = parent_id or str(uuid.uuid4())
    enriched = []
    for i, text in enumerate(chunks):
        tags = tag_fn(text, i) if tag_fn else []
        enriched.append(EnrichedChunk(
            chunk_text=text,
            parent_id=pid,
            tags=tags,
            chunk_index=i,
        ))
    return enriched


def simple_section_tag(chunk_text: str, index: int) -> list[str]:
    """Rule-based: first chunk = intro, last = conclusion (you'd pass total later in real code)."""
    tags = []
    if index == 0:
        tags.append("intro")
    # Could add "conclusion" for last chunk if we had total_chunks
    if "price" in chunk_text.lower() or "cost" in chunk_text.lower():
        tags.append("pricing")
    if "api" in chunk_text.lower() or "endpoint" in chunk_text.lower():
        tags.append("api")
    return tags


# ---------- Example: enrich chunks and show how retrieval could use metadata ----------
if __name__ == "__main__":
    # Simulate chunks from one document
    chunks = [
        "Welcome to our product. This document explains the main features.",
        "Our API has two endpoints: GET /users and POST /users.",
        "Pricing: $10/month for basic, $20 for pro.",
    ]

    enriched = add_parent_id_and_tags(
        chunks,
        parent_id="doc_manual_001",
        tag_fn=simple_section_tag,
    )

    for c in enriched:
        print(f"parent_id={c.parent_id}, index={c.chunk_index}, tags={c.tags}")
        print(f"  text: {c.chunk_text[:50]}...")
        print()

    # How retrieval uses it (pseudo-code):
    # 1. Vector DB stores: { "text": c.chunk_text, "parent_id": c.parent_id, "tags": c.tags }
    # 2. Query: "Only get chunks where parent_id = 'doc_manual_001' and 'pricing' in tags"
    # 3. Or: after retrieval, group results by parent_id to show "from Document X"
```

---

## How This Helps at Retrieval (Summary)

| Use case | How metadata helps |
|----------|--------------------|
| **Filter by doc** | Only retrieve chunks where `parent_id` is in a list of allowed docs (e.g. user's access). |
| **Filter by section** | Only retrieve chunks where `tags` contains "pricing" or "api". |
| **Deduplication** | If 3 chunks from same `parent_id` are in top‑10, you can return the parent doc once. |
| **UX** | Show "Source: Document A, Section: Pricing" using `parent_id` + `tags`. |

So: **append parent_id and tags to every chunk** when you build your index; then use them in filters and in the UI at retrieval time.
