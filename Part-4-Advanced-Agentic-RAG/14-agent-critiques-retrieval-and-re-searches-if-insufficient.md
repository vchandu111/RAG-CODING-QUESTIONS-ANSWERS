# Q14: Agentic RAG — Agent Critiques Retrieval and Re-Searches

## The Question

> **Design a LangGraph or CrewAI workflow** where the agent **"critiques" its own retrieval** and **re-runs the search** if the context is insufficient.

---

## Concepts

- **Basic RAG**: Retrieve once → generate answer. If retrieval was bad, the answer is bad.
- **Agentic RAG**: An **agent** (a loop) can: (1) retrieve, (2) **check** if the retrieved context is enough to answer the question, (3) if not, **re-query** (e.g. different query or more results) and try again, (4) if yes, generate the answer.
- **Critique**: A small step that answers: "Is the retrieved context sufficient to answer the user question? Yes/No (and why)." We can use an LLM or a simple heuristic (e.g. similarity score too low).
- **LangGraph / CrewAI**: Frameworks to build such a **graph** of steps (retrieve → critique → if insufficient → retrieve again → … → generate).

---

## Approach

1. **Start**: User question.

2. **Retrieve**: Run the retriever (vector or hybrid) with the user question; get top‑k chunks and scores.

3. **Critique**:  
   - **Option A (LLM)**: Ask the LLM: "Given this question and these chunks, can you answer the question from the chunks alone? Reply YES or NO and one sentence why."  
   - **Option B (heuristic)**: If best similarity &lt; threshold, say "insufficient."

4. **Decision**:  
   - If **sufficient** → go to **Generate**: call the LLM with question + chunks; return answer.  
   - If **insufficient** → **Re-query**: e.g. use query expansion (3 variations), retrieve again with the new queries, merge results; then go back to **Critique** (or directly to Generate with a max retry, e.g. 2).

5. **Loop**: Repeat retrieve → critique until "sufficient" or max iterations (e.g. 2). Then generate.

6. **LangGraph**: Nodes = retrieve, critique, generate; edges = critique → generate (if yes), critique → retrieve (if no). CrewAI: similar idea with agents and tasks.

---

## Python Implementation (LangGraph-style state machine)

```python
# Install: pip install langgraph langchain langchain-openai (or use simple state dict)

from typing import TypedDict, Literal
from enum import Enum

class State(TypedDict):
    question: str
    chunks: list[str]
    scores: list[float]
    sufficient: bool
    attempt: int
    max_attempts: int

def retrieve(state: State, retriever_fn) -> State:
    """Retrieve top-k chunks for state['question']."""
    question = state["question"]
    chunks, scores = retriever_fn(question, top_k=5)
    state["chunks"] = chunks
    state["scores"] = scores
    return state

def critique(state: State, critique_fn) -> State:
    """
    Decide if chunks are sufficient to answer the question.
    critique_fn(question, chunks) -> True/False
    """
    sufficient = critique_fn(state["question"], state["chunks"], state.get("scores", []))
    state["sufficient"] = sufficient
    return state

def generate(state: State, generate_fn) -> State:
    """Generate answer from question + chunks."""
    if state["chunks"] and state["sufficient"]:
        answer = generate_fn(state["question"], state["chunks"])
    else:
        answer = "I couldn't find enough relevant information in the documents."
    state["answer"] = answer
    return state

def agentic_rag_loop(
    question: str,
    retriever_fn,
    critique_fn,
    generate_fn,
    max_attempts: int = 2,
) -> dict:
    """
    Simple loop: retrieve -> critique -> if not sufficient and attempt < max, retrieve again (e.g. with expanded query) -> else generate.
    """
    state = {
        "question": question,
        "chunks": [],
        "scores": [],
        "sufficient": False,
        "attempt": 0,
        "max_attempts": max_attempts,
    }
    while state["attempt"] < max_attempts:
        state["attempt"] += 1
        state = retrieve(state, retriever_fn)
        state = critique(state, critique_fn)
        if state["sufficient"]:
            break
        # Optional: expand query and retry (simplified: just retry with same query here)
    state = generate(state, generate_fn)
    return {"answer": state["answer"], "chunks": state["chunks"], "attempts": state["attempt"]}


# ---------- Example: mock retriever and critique ----------
if __name__ == "__main__":
    def mock_retriever(question: str, top_k: int = 5):
        # Simulate: first attempt returns weak chunks, second returns better
        if "attempt" not in agentic_rag_loop.__code__.co_freevars:  # simplified
            return ["Weak chunk 1.", "Weak chunk 2."], [0.3, 0.25]
        return ["Refunds are allowed within 30 days. Contact support."], [0.9]

    def mock_critique(question, chunks, scores):
        return (scores[0] >= 0.6) if scores else False

    def mock_generate(question, chunks):
        return "Based on the documents: " + (chunks[0][:50] + "..." if chunks else "No info.")

    # We need to pass attempt into retriever for demo; use a simple closure
    attempt = [0]
    def retriever_with_attempt(q, top_k=5):
        attempt[0] += 1
        if attempt[0] == 1:
            return ["Weak chunk."], [0.35]
        return ["Refunds within 30 days. Contact support."], [0.85]

    result = agentic_rag_loop(
        "What is the refund policy?",
        retriever_with_attempt,
        mock_critique,
        mock_generate,
        max_attempts=2,
    )
    print("Answer:", result["answer"])
    print("Attempts:", result["attempts"])
```

---

## Summary

- **Agentic RAG** = retrieve → **critique** (is context enough?) → if no, **re-retrieve** (e.g. expanded query) → repeat until yes or max attempts → **generate**.
- **Critique**: LLM ("Can you answer from these chunks? YES/NO") or heuristic (e.g. best score &lt; threshold).
- **LangGraph/CrewAI**: Implement this as a graph (retrieve → critique → generate or back to retrieve). The Python above is a minimal loop version of the same idea.
