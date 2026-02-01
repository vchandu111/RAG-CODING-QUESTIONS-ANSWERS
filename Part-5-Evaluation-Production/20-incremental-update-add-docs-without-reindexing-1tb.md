# Q20: Incremental Update — Add New Docs Without Re-Indexing 1TB

## The Question

> **How do you code an incremental update script** that **adds new documents** to the vector DB **without re-indexing** the entire **1-terabyte** corpus?

---

## Concepts

- **Full re-index**: Re-chunk and re-embed **all** documents (1 TB), then replace the whole index. Slow and expensive; we want to avoid it when we only add a few new docs.
- **Incremental update**: **Only** process **new** or **changed** documents: chunk them, embed them, and **add** them to the existing vector DB (insert new vectors + metadata). The rest of the index stays as-is.
- **Implementation**: (1) **Track** which docs are already indexed (e.g. a list of doc IDs or a "last_updated" table). (2) **Discover** new/changed files (e.g. scan a folder, compare with tracked list, or use file hashes). (3) **Process** only new docs: load → chunk → embed → **add** to vector DB (e.g. `collection.add(ids, embeddings, metadatas)`). (4) **Update** the tracker (e.g. add new doc IDs). No need to touch the rest of the corpus.

---

## Approach

1. **Tracker**: Keep a set (or DB table) of **indexed doc IDs** (e.g. file path or content hash). When the script runs, we compare "current files" with this set to find **new** and optionally **changed** (e.g. by mtime or hash).

2. **List new/changed docs**:  
   - Scan your source (e.g. directory, S3 prefix, DB).  
   - For each doc, compute an ID (path or hash). If ID not in tracker (or content changed), mark as "to index."

3. **Process only those docs**:  
   - Load text (e.g. read file, parse PDF).  
   - Chunk (recursive or semantic).  
   - Embed each chunk (same model as existing index).  
   - Build lists: `ids`, `embeddings`, `metadatas` (e.g. `parent_id`, `chunk_index`).

4. **Insert into vector DB**:  
   - Call the DB’s **add** (or **upsert**) API: `add(ids=..., embeddings=..., metadatas=...)`.  
   - Do **not** delete or re-build the rest of the index.  
   - If the DB has a max batch size, loop over batches.

5. **Update tracker**: Add the newly indexed doc IDs to the tracker (and optionally update "last_indexed" time).  
   - Next run: only new/changed docs again.

6. **Optional — Deletes**: If a document is removed from the source, remove its chunk IDs from the vector DB (e.g. delete by `parent_id`) and remove its ID from the tracker.

---

## Python Implementation (Chroma-style; same idea for Pinecone/Milvus)

```python
# Install: pip install chromadb sentence-transformers

import hashlib
import os
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# Persist "already indexed" doc IDs (simple file-based set)
TRACKER_FILE = "indexed_docs.txt"

def load_indexed_ids(tracker_path: str = TRACKER_FILE) -> set[str]:
    if not os.path.exists(tracker_path):
        return set()
    with open(tracker_path) as f:
        return set(line.strip() for line in f if line.strip())

def save_indexed_ids(ids: set[str], tracker_path: str = TRACKER_FILE):
    with open(tracker_path, "w") as f:
        for doc_id in ids:
            f.write(doc_id + "\n")

def doc_id_from_path(path: str) -> str:
    """Use path as ID, or hash for shorter ID."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]

def get_new_docs(source_dir: str, indexed_ids: set[str]) -> list[str]:
    """List file paths in source_dir that are not in indexed_ids."""
    new = []
    for p in Path(source_dir).rglob("*"):
        if p.is_file() and p.suffix in (".txt", ".md"):
            doc_id = doc_id_from_path(str(p))
            if doc_id not in indexed_ids:
                new.append(str(p))
    return new

def chunk_text(text: str, max_chars: int = 512) -> list[str]:
    """Simple chunking: split by paragraph then by size."""
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for i in range(0, len(para), max_chars):
            chunks.append(para[i : i + max_chars])
    return chunks or [text]

def incremental_index(
    source_dir: str,
    collection,
    model: SentenceTransformer,
    tracker_path: str = TRACKER_FILE,
    max_chars: int = 512,
) -> int:
    """
    Find new docs in source_dir, chunk + embed, add to collection, update tracker.
    Returns number of new docs indexed.
    """
    indexed = load_indexed_ids(tracker_path)
    new_paths = get_new_docs(source_dir, indexed)
    if not new_paths:
        return 0

    all_ids = []
    all_embeddings = []
    all_metadatas = []
    all_docs = []

    for path in new_paths:
        doc_id = doc_id_from_path(path)
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunks = chunk_text(text, max_chars=max_chars)
        embeddings = model.encode(chunks, convert_to_numpy=True)
        for i, (c, e) in enumerate(zip(chunks, embeddings)):
            all_ids.append(f"{doc_id}_chunk_{i}")
            all_embeddings.append(e.tolist())
            all_metadatas.append({"parent_id": doc_id, "path": path})
            all_docs.append(c)
        indexed.add(doc_id)

    if all_ids:
        # Chroma add; for large batches, chunk into 1000s
        collection.add(ids=all_ids, embeddings=all_embeddings, metadatas=all_metadatas, documents=all_docs)
    save_indexed_ids(indexed, tracker_path)
    return len(new_paths)


# ---------- Example ----------
if __name__ == "__main__":
    # Create a small test dir and index
    test_dir = "test_docs"
    os.makedirs(test_dir, exist_ok=True)
    Path(test_dir, "new_doc.txt").write_text("This is a new document. It has two sentences.")

    client = chromadb.PersistentClient(path="./chroma_incremental")
    coll = client.get_or_create_collection("docs")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    n = incremental_index(test_dir, coll, model)
    print("New docs indexed:", n)
    print("Indexed IDs:", load_indexed_ids())
```

---

## Summary

- **Tracker**: Keep a set of **indexed doc IDs** (or content hashes).  
- **New docs**: List source files; only process those **not** in the tracker (and optionally changed).  
- **Process**: Chunk + embed **only** new docs; **add** (insert) into the vector DB; **update** the tracker.  
- **Result**: New documents are searchable without re-indexing the full 1 TB corpus.
