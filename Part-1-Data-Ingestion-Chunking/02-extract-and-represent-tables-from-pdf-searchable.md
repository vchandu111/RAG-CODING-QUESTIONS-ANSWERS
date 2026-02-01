# Q2: Extract and Represent Tables from PDF (Searchable)

## The Question

> How would you code a pipeline to **extract and represent a complex table within a PDF** so it remains searchable? (e.g., Markdown conversion or Table-to-Text)

---

## Concepts

- **PDFs** store layout (positions of text), not structure (this is a table, this is a heading). So we need a library that can **detect tables** and **read cells**.
- **Table → Markdown**: Each row becomes a line; columns are separated by `|`. Easy to embed and search as text.
- **Table → plain text**: Describe the table in sentences (e.g. "Row 1: Name=Alice, Age=30"). Also searchable.
- **Searchable** here means: after we convert to Markdown or text, we can **chunk and embed** it like any other document so RAG can retrieve it.

---

## Approach

1. **Load the PDF**  
   Use a library that can extract text and optionally detect tables (e.g. `pdfplumber` or `pypdf` + `pdf2image` + OCR if needed).

2. **Extract tables**  
   `pdfplumber` has `page.find_tables()` which returns table regions. For each table, get the cell grid (rows × columns).

3. **Convert each table to a string format**  
   - **Markdown**: Build a string like `| Col1 | Col2 |\n|------|------|\n| A | B |`.  
   - **Table-to-text**: For each row, write something like "Row N: col1=..., col2=...". Good for small tables or when you want natural language.

4. **Combine with rest of page text**  
   So the final document has: "Paragraph... Table as Markdown... Next paragraph." That way the whole page is one searchable blob.

5. **Optional: chunk and embed**  
   Split this combined text into chunks (e.g. by section or by fixed size), then embed for RAG.

---

## Python Implementation

```python
# Install: pip install pdfplumber

import pdfplumber
from pathlib import Path

def table_to_markdown(table_rows: list[list[str]]) -> str:
    """
    Turn a 2D grid of cells (list of rows, each row = list of cell strings)
    into a Markdown table string.
    """
    if not table_rows:
        return ""

    # First row = header
    header = table_rows[0]
    sep = "| " + " | ".join("---" for _ in header) + " |"
    lines = ["| " + " | ".join(str(c).strip() for c in header) + " |", sep]

    for row in table_rows[1:]:
        # Pad row if shorter than header
        row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(str(c).strip() for c in row) + " |")

    return "\n".join(lines)


def table_to_text(table_rows: list[list[str]]) -> str:
    """Describe table in plain text: Row N: col1=..., col2=..."""
    if not table_rows:
        return ""
    header = table_rows[0]
    lines = []
    for i, row in enumerate(table_rows[1:], start=1):
        row = row + [""] * (len(header) - len(row))
        pairs = [f"{h}={v}" for h, v in zip(header, row)]
        lines.append(f"Row {i}: " + ", ".join(pairs))
    return "\n".join(lines)


def extract_page_content(pdf_path: str, page_num: int = 0) -> str:
    """
    Extract one page: body text + all tables as Markdown.
    Returns a single string you can later chunk and embed.
    """
    parts = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        # Raw text (non-table)
        text = page.extract_text()
        if text and text.strip():
            parts.append(text.strip())

        # Tables
        tables = page.find_tables()
        for table in tables:
            rows = table.extract()
            if rows:
                parts.append(table_to_markdown(rows))

    return "\n\n".join(parts)


def extract_full_pdf(pdf_path: str, table_format: str = "markdown") -> str:
    """
    Extract all pages. Each page = text + tables (as markdown or text).
    table_format: "markdown" or "text"
    """
    all_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                all_pages.append(text.strip())

            for table in page.find_tables():
                rows = table.extract()
                if rows:
                    if table_format == "markdown":
                        all_pages.append(table_to_markdown(rows))
                    else:
                        all_pages.append(table_to_text(rows))

    return "\n\n".join(all_pages)


# ---------- Example ----------
if __name__ == "__main__":
    # Use a PDF you have, or create a simple one for testing
    pdf_path = "sample.pdf"
    if Path(pdf_path).exists():
        content = extract_full_pdf(pdf_path, table_format="markdown")
        print(content[:1500])
    else:
        # Demo with a fake table
        fake_table = [
            ["Name", "Age", "Role"],
            ["Alice", "30", "Engineer"],
            ["Bob", "25", "Designer"],
        ]
        print(table_to_markdown(fake_table))
        print("\n--- As text ---\n")
        print(table_to_text(fake_table))
```

---

## Summary

- **Pipeline**: PDF → extract text + detect tables → convert each table to **Markdown** or **table-to-text** → concatenate into one document → chunk & embed for RAG.
- **Searchable**: Because the result is plain text/Markdown, your normal RAG indexing (chunking + embedding) makes it searchable.
