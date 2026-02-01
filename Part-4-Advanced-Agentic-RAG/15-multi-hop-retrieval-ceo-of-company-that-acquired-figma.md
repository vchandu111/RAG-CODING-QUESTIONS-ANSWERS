# Q15: Multi-Hop Retrieval — "CEO of Company That Acquired Figma?"

## The Question

> **Write a logic** for a query like **"Who is the CEO of the company that acquired Figma?"** (Requires **two-step retrieval**.)

---

## Concepts

- **Single-hop**: One query → retrieve chunks → answer. "Who acquired Figma?" → one search.
- **Multi-hop**: The answer needs **two** steps. Step 1: "Which company acquired Figma?" → e.g. "Adobe." Step 2: "Who is the CEO of Adobe?" → retrieve again with that and get the CEO name.
- **Logic**: (1) **Decompose** the question into sub-questions (manually or with an LLM). (2) **Answer** the first sub-question with retrieval + LLM. (3) **Use that answer** as part of the next query; retrieve again and answer. (4) Combine or return the final answer.

---

## Approach

1. **Question**: "Who is the CEO of the company that acquired Figma?"

2. **Decompose** (manually or LLM):  
   - Q1: "Which company acquired Figma?"  
   - Q2: "Who is the CEO of [answer of Q1]?"

3. **Step 1**: Retrieve for Q1; get chunks; generate answer (e.g. "Adobe"). Store as **intermediate_answer_1**.

4. **Step 2**: Build Q2 using the intermediate answer: "Who is the CEO of Adobe?" Retrieve for Q2; get chunks; generate answer (e.g. "Shantanu Narayen"). Store as **intermediate_answer_2**.

5. **Final answer**: "The CEO of the company that acquired Figma (Adobe) is Shantanu Narayen." Or return the two sub-answers and let the user see the chain.

6. **General**: For N hops, repeat: decompose → for each sub-question in order, retrieve + generate, plug the answer into the next sub-question.

---

## Python Implementation

```python
# Decompose with LLM or use a simple rule; then retrieve + generate for each step.

from typing import Callable, Optional

def decompose_multi_hop(question: str, llm_decompose_fn: Optional[Callable] = None) -> list[str]:
    """
    Split "Who is the CEO of the company that acquired Figma?"
    into ["Which company acquired Figma?", "Who is the CEO of {answer1}?"]
    If no LLM, return a simple heuristic or the single question.
    """
    if llm_decompose_fn:
        return llm_decompose_fn(question)
    # Simple heuristic: if "company that" or "that acquired" etc., we might need 2 steps
    # For demo, return a fixed 2-step for this exact style
    if "CEO" in question and "acquired" in question:
        return [
            "Which company acquired Figma?",
            "Who is the CEO of that company?",  # placeholder; we'll fill with answer1
        ]
    return [question]

def answer_one_hop(sub_question: str, retriever_fn, generate_fn, top_k: int = 5) -> str:
    """Retrieve + generate for one sub-question. Returns answer string."""
    chunks, _ = retriever_fn(sub_question, top_k=top_k)
    if not chunks:
        return ""
    return generate_fn(sub_question, chunks)

def multi_hop_retrieval(
    question: str,
    retriever_fn: Callable,
    generate_fn: Callable,
    decompose_fn: Optional[Callable] = None,
    max_hops: int = 3,
) -> dict:
    """
    Decompose question into sub-questions; for each, retrieve + generate;
    plug previous answer into next sub-question; return final answer and chain.
    """
    sub_questions = decompose_multi_hop(question, decompose_fn)[:max_hops]
    chain = []
    current_question = question
    for i, sq in enumerate(sub_questions):
        # Replace placeholder like "that company" with previous answer
        if i > 0 and chain:
            sq = sq.replace("that company", chain[-1]["answer"]).replace("the company", chain[-1]["answer"])
        answer = answer_one_hop(sq, retriever_fn, generate_fn)
        chain.append({"sub_question": sq, "answer": answer})
    # Final answer: last hop or combine
    final = chain[-1]["answer"] if chain else ""
    return {"final_answer": final, "chain": chain}


# ---------- Example with mock ----------
if __name__ == "__main__":
    # Mock: first query -> "Adobe"; second query -> "Shantanu Narayen"
    def mock_retriever(q, top_k=5):
        if "acquired Figma" in q or "Figma" in q:
            return ["Adobe announced the acquisition of Figma in 2022."], [0.9]
        if "CEO" in q and "Adobe" in q:
            return ["Adobe CEO is Shantanu Narayen."], [0.9]
        return ["No relevant doc."], [0.1]

    def mock_generate(q, chunks):
        return chunks[0].split(".")[0] + "." if chunks else "Unknown."

    result = multi_hop_retrieval(
        "Who is the CEO of the company that acquired Figma?",
        mock_retriever,
        mock_generate,
    )
    print("Chain:", result["chain"])
    print("Final:", result["final_answer"])
```

---

## Summary

- **Multi-hop** = answer in order: first sub-question → retrieve + answer → plug that answer into the next sub-question → retrieve + answer again.
- **Decompose** the user question into sub-questions (LLM or rules); then **for each** in order: retrieve, generate, plug answer into next.
- **Code**: Loop over sub-questions; each step calls retriever + generator; substitute previous answer into the next question text.
