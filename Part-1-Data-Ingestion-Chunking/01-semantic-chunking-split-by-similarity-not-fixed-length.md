# Q1: Semantic Chunking — Split by Meaning, Not Fixed Length

## The Question

> Write a function to split a document based on **semantic similarity** rather than fixed token length. How do you decide the threshold for a "new" chunk?

---

## Concepts

- **Fixed-size chunking**: Split text every N characters or tokens. Simple, but can cut sentences or ideas in half.
- **Semantic chunking**: Group text so that **similar meaning** stays together. Better for RAG because each chunk is a coherent idea.
- **Embedding**: A list of numbers that represents the "meaning" of a sentence. Similar sentences → similar embeddings.
- **Threshold**: A number (e.g. 0.7). If the next sentence is "less similar" than this to the current chunk, we start a **new chunk**.

---

## Approach

1. **Split the document into sentences.**  
   We work sentence-by-sentence so we don't break mid-sentence.

2. **Get an embedding for each sentence.**  
   Use a small model (e.g. `sentence-transformers`) to turn each sentence into a vector.

3. **Start the first chunk** with the first sentence.

4. **For each next sentence:**  
   - Get its embedding.  
   - Compare it to the **current chunk's embedding** (e.g. average of all sentence embeddings in the chunk).  
   - **Similarity** = cosine similarity between the two vectors (0 to 1; higher = more similar).

5. **Decide: same chunk or new chunk?**  
   - If similarity **≥ threshold** → add the sentence to the **current chunk**.  
   - If similarity **< threshold** → **close the current chunk** and start a **new chunk** with this sentence.

6. **How to choose the threshold?**  
   - **0.5–0.6**: Fewer, larger chunks (more context per chunk, less granular).  
   - **0.7–0.8**: Balanced (good default).  
   - **0.85+**: More, smaller chunks (very similar content only).  
   Start with **0.7** and tune on a few documents; lower if chunks are too small, raise if they're too big.

---

## Python Implementation

```python
# Install: pip install sentence-transformers numpy

from sentence_transformers import SentenceTransformer
import numpy as np

def get_embedding(model, text: str) -> np.ndarray:
    """Turn one piece of text into a vector (embedding)."""
    return model.encode(text, convert_to_numpy=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How similar are two vectors? 1 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_chunk(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.7,
    sentence_split_fn=None,
) -> list[str]:
    """
    Split text into chunks by meaning.
    - threshold: if similarity(next_sent, current_chunk) < this, start a new chunk.
    """
    if sentence_split_fn is None:
        # Simple split: period, question mark, exclamation + space
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    else:
        sentences = sentence_split_fn(text)

    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_embedding = get_embedding(model, sentences[0])

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_embedding = get_embedding(model, sent)
        similarity = cosine_similarity(sent_embedding, current_chunk_embedding)

        if similarity >= threshold:
            # Same topic → add to current chunk
            current_chunk_sentences.append(sent)
            # Update chunk embedding = average of all sentences in chunk
            all_embeddings = np.array([get_embedding(model, s) for s in current_chunk_sentences])
            current_chunk_embedding = np.mean(all_embeddings, axis=0)
        else:
            # New topic → save current chunk and start new one
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sent]
            current_chunk_embedding = sent_embedding

    chunks.append(" ".join(current_chunk_sentences))
    return chunks


# ---------- Example usage ----------
if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast

    doc = """
    RAG stands for Retrieval-Augmented Generation. You first retrieve relevant documents.
    Then you pass them to an LLM to generate an answer. Embeddings are vectors that represent meaning.
    Python is a programming language. It is used for data science and ML.
    """
    doc = " ".join(doc.split())

    chunks = semantic_chunk(doc, model, threshold=0.7)
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c[:80]}...")
```

---

## How to Decide the Threshold (Summary)

| Goal | Typical threshold |
|------|-------------------|
| Fewer, longer chunks | 0.5–0.6 |
| Balanced (good default) | 0.7–0.75 |
| More, shorter chunks | 0.8–0.85 |

Use **0.7** first; then adjust based on your chunk sizes and retrieval quality.
