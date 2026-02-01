# Q3: Recursive Character Splitter — Why Better Than Fixed-Size?

## The Question

> **Explain and implement a recursive splitter.** Why is it often superior to a simple fixed-size splitter?

---

## Concepts

- **Fixed-size splitter**: Cut text every N characters. Fast and simple, but it can **cut in the middle of a word or sentence**, so chunks are messy.
- **Recursive splitter**: Try to split at **natural boundaries** first (e.g. paragraph, then sentence, then word, then character). So we avoid breaking words/sentences when possible.
- **Idea**: Split by the **largest separator** (e.g. `\n\n`) so that chunks are ~max size. If a piece is still too long, split again by the **next** separator (e.g. `\n`), then `. `, then ` `, then character. So we only break words when necessary.

---

## Approach

1. **Define a list of separators** from "biggest" to "smallest":  
   e.g. `["\n\n", "\n", ". ", " ", ""]`.  
   The last one `""` means "split every character" so we never exceed max size.

2. **Function: split_one(text, separator)**  
   - If separator is `""`, split into list of characters (or use a safe fallback like splitting every `max_chars` chars).  
   - Otherwise split by separator and return list of parts.

3. **Function: recursive_split(text, separators, max_chars)**  
   - If `len(text) <= max_chars`, return `[text]`.  
   - Take the **first** separator in the list.  
   - Split text by that separator.  
   - For each part: if it's still longer than `max_chars`, call **recursive_split(part, rest_of_separators, max_chars)**.  
   - If all parts are small enough, merge them into chunks of size ≤ max_chars (by concatenating until adding the next part would exceed max_chars).  
   - If the first separator didn't produce small enough parts (e.g. one paragraph is huge), try the **next** separator (e.g. `\n`, then `. `, etc.).

4. **Why it's better**: Chunks tend to end at paragraph or sentence boundaries, so they're cleaner and easier for the model to use. Fixed-size often cuts mid-sentence.

---

## Python Implementation

```python
def split_by_separator(text: str, separator: str) -> list[str]:
    """Split text by a separator. If separator is '', split into chars (for last resort)."""
    if separator == "":
        return list(text)  # or we handle this in merge step
    return [t.strip() for t in text.split(separator) if t.strip()]


def merge_into_chunks(parts: list[str], max_chars: int, separator: str) -> list[str]:
    """Combine small parts into chunks of size <= max_chars."""
    if not parts:
        return []
    chunks = []
    current = parts[0]
    sep_join = separator if separator else ""
    for part in parts[1:]:
        if len(current) + len(sep_join) + len(part) <= max_chars:
            current = current + sep_join + part
        else:
            chunks.append(current)
            current = part
    chunks.append(current)
    return chunks


def recursive_split(
    text: str,
    separators: list[str],
    max_chars: int = 512,
    overlap: int = 0,
) -> list[str]:
    """
    Split text by trying separators in order: paragraph, newline, sentence, space, then char.
    Keeps chunks under max_chars and avoids breaking in the middle of words/sentences when possible.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sep = separators[0]
    rest_seps = separators[1:]

    parts = split_by_separator(text, sep)
    if sep == "":
        # Last resort: split by fixed size
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars - overlap)]

    # Try to merge parts into chunks of size <= max_chars
    chunks = []
    for part in parts:
        if len(part) <= max_chars:
            chunks.append(part)
        else:
            # Part still too long: recurse with next separator
            chunks.extend(recursive_split(part, rest_seps, max_chars, overlap))

    # Now merge small consecutive chunks so we don't return tiny pieces
    merged = merge_into_chunks(chunks, max_chars, sep)
    return merged


# ---------- Default separators (like LangChain's RecursiveCharacterTextSplitter) ----------
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------- Example ----------
if __name__ == "__main__":
    long_text = """
    First paragraph. It has two sentences. We want to split by paragraphs first.

    Second paragraph. Also multiple sentences. So we get nice chunks.

    Third paragraph. This one is very long. "Lorem ipsum" style. We need to split again. So we use newline or sentence. The recursive splitter will try the next separator.
    """

    chunks = recursive_split(long_text.strip(), DEFAULT_SEPARATORS, max_chars=100)
    for i, c in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(c)} chars) ---")
        print(c)
        print()
```

---

## Why Recursive Is Usually Better Than Fixed-Size

| Fixed-size | Recursive |
|------------|-----------|
| Cuts anywhere → mid-word, mid-sentence | Prefers paragraph → sentence → word → character |
| Chunks can be hard to read | Chunks are coherent (full sentences/paragraphs) |
| Same logic for all documents | Reuses same idea for any language/text |
| Very simple to implement | Slightly more code, but better quality for RAG |

So: **use a recursive splitter** when you care about chunk quality for retrieval and generation; use fixed-size only when you need the simplest possible thing.
