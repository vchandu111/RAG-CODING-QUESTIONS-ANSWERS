# Q8: Metadata Filtering for "Read" Permissions

## The Question

> How do you **implement metadata filtering** in a vector DB like **Pinecone or Milvus** to ensure a user only retrieves documents they have **"read" permissions** for?

---

## Concepts

- **Vector DB**: Stores embeddings and optional **metadata** (e.g. `doc_id`, `owner`, `allowed_roles`). You can **filter** by metadata before or during similarity search.
- **Read permissions**: Each document (or chunk) has metadata like `allowed_users: ["alice", "bob"]` or `department: "engineering"`. When user "alice" asks a question, we only want to search chunks where "alice" is allowed (e.g. `allowed_users` contains "alice" or `department` matches her department).
- **Flow**: At query time, you pass **filter** = { "allowed_users": { "$contains": "alice" } } (syntax depends on DB). The DB returns only matching vectors. So the user never sees chunks they’re not allowed to read.

---

## Approach

1. **When indexing**: For each chunk, store metadata such as:  
   `allowed_users`, `document_id`, `department`, etc. (whatever your permission model is).

2. **Permission model**: Decide the rule. Examples:  
   - "User can read chunk if their `user_id` is in `chunk.allowed_users`."  
   - "User can read if `chunk.department` equals `user.department`."

3. **At query time**: Build a **filter dict** from the current user (e.g. `user_id = "alice"`).  
   - Pinecone: `filter = {"allowed_users": {"$in": ["alice"]}}` or `{"department": {"$eq": "eng"}}`.  
   - Chroma: `where = {"allowed_users": {"$contains": "alice"}}`.  
   Syntax varies by DB; check the docs.

4. **Call the vector DB**  
   `query(embedding, top_k=10, filter=filter)`.  
   Only chunks that pass the filter are considered.

5. **Return** only those chunks to the LLM. So the answer is grounded only in documents the user is allowed to read.

---

## Python Implementation (Chroma — same idea for Pinecone/Milvus)

```python
# Install: pip install chromadb

import chromadb
from chromadb.config import Settings

def create_collection_with_metadata():
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection("docs", metadata={"description": "RAG chunks with permissions"})
    return coll

def add_documents_with_permissions(collection, ids: list[str], documents: list[str], embeddings: list, allowed_users: list[list[str]]):
    """
    allowed_users[i] = list of user_ids that can read documents[i].
    """
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=[{"allowed_users": ",".join(allowed_users[i])} for i in range(len(ids))],
    )

def query_with_filter(collection, query_embedding: list, top_k: int, allowed_user: str):
    """
    Only return chunks where allowed_user is in the chunk's allowed_users.
    Chroma stores list as comma-separated string; we use $contains for the string.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"allowed_users": {"$contains": allowed_user}},
    )
    return results

# ---------- Example ----------
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    coll = create_collection_with_metadata()
    ids = ["doc1", "doc2", "doc3"]
    docs = [
        "Alice can see this. Internal policy for team A.",
        "Bob can see this. Internal policy for team B.",
        "Alice and Bob can see this. Shared policy.",
    ]
    allowed = [
        ["alice"],
        ["bob"],
        ["alice", "bob"],
    ]
    embs = model.encode(docs, convert_to_numpy=True).tolist()
    add_documents_with_permissions(coll, ids, docs, embs, allowed)

    query = "What is the internal policy?"
    q_emb = model.encode(query, convert_to_numpy=True).tolist()

    # As user "alice": should get doc1 and doc3
    out_alice = query_with_filter(coll, q_emb, top_k=5, allowed_user="alice")
    print("As alice:", out_alice["ids"], out_alice["documents"])

    # As user "bob": should get doc2 and doc3
    out_bob = query_with_filter(coll, q_emb, top_k=5, allowed_user="bob")
    print("As bob:", out_bob["ids"], out_bob["documents"])
```

---

## Pinecone / Milvus (concept only)

- **Pinecone**: Use `filter` in `query()`: e.g. `{"allowed_users": {"$in": ["alice"]}}`. Index must have metadata indexed.
- **Milvus**: Use `expr` in search: e.g. `expr='allowed_users == "alice"'` or use their filter syntax for list membership.

---

## Summary

- **Store** permission metadata with each chunk (e.g. `allowed_users`, `department`).  
- **At query time**, build a filter from the current user (e.g. `user_id in allowed_users`).  
- **Call** vector DB with `query(embedding, top_k, filter=...)` so only allowed chunks are returned.
