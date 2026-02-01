# Q17: Calculate Faithfulness and Answer Relevance (RAGAS)

## The Question

> **Write code** to calculate **Faithfulness** and **Answer Relevance** using a framework like **RAGAS** or **TruLens**.

---

## Concepts

- **Faithfulness**: Does the **answer** stick to the **context** (retrieved chunks)? If the model adds facts not in the context, faithfulness is low. We measure by: "Given the answer and the context, rate how much of the answer is supported by the context (e.g. 0–1)." RAGAS does this with an LLM or a small model.
- **Answer Relevance**: Is the **answer** actually **relevant** to the **question**? We measure by: "Given the question and the answer, rate how relevant the answer is (e.g. 0–1)." RAGAS uses an LLM to score this.
- **RAGAS**: A library that computes these (and more) for RAG. You pass: question, context (retrieved chunks), answer (from your RAG). It returns scores like faithfulness and answer_relevance.

---

## Approach

1. **Prepare one eval example**: question (string), context (list of chunk strings), answer (string from your RAG).

2. **Faithfulness**:  
   - RAGAS (or similar) uses an LLM to extract "statements" from the answer and check each against the context.  
   - Score = (number of statements supported by context) / (total statements).  
   - You call something like `ragas.evaluate([example], metrics=[faithfulness])`.

3. **Answer Relevance**:  
   - RAGAS uses an LLM: "Given this question and answer, how relevant is the answer (0–1)?"  
   - You call something like `ragas.evaluate([example], metrics=[answer_relevance])`.

4. **Average** over many examples to get average Faithfulness and average Answer Relevance for your RAG.

5. **TruLens**: Similar idea — you define "faithfulness" and "relevance" as custom metrics or use built-ins; TruLens runs your RAG and scores each response.

---

## Python Implementation (RAGAS)

```python
# Install: pip install ragas openai  # and set OPENAI_API_KEY

from datasets import Dataset
import os

# RAGAS expects a Dataset with columns: question, context (list of strings joined or as list), answer
def evaluate_ragas_single(question: str, context: list[str], answer: str, api_key: str = None):
    """
    Compute Faithfulness and Answer Relevance for one (question, context, answer).
    context = list of retrieved chunk strings; answer = your RAG output.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevance

    # RAGAS wants context as a list of strings (one per chunk) or single string
    context_str = "\n\n".join(context) if isinstance(context, list) else context
    data = {
        "question": [question],
        "context": [[context_str]],  # list of lists for batch
        "answer": [answer],
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevance])
    return {
        "faithfulness": result["faithfulness"],
        "answer_relevance": result["answer_relevance"],
    }


# ---------- Manual version (no RAGAS): use LLM to score 0-1 ----------
def score_faithfulness_manual(question: str, context: list[str], answer: str, llm_fn=None) -> float:
    """
    Ask LLM: "Rate 0-1 how much the answer is supported by the context."
    Returns a single float. For demo we return a placeholder.
    """
    if llm_fn is None:
        # Placeholder: assume 1.0 if answer is short and context exists
        return 1.0 if context and answer else 0.0
    prompt = f"Context:\n{chr(10).join(context[:3])}\n\nAnswer:\n{answer}\n\nRate from 0 to 1 how much the answer is supported by the context. Reply with one number."
    response = llm_fn(prompt)
    try:
        return float(response.strip())
    except ValueError:
        return 0.0

def score_answer_relevance_manual(question: str, answer: str, llm_fn=None) -> float:
    """Ask LLM: "Rate 0-1 how relevant is the answer to the question." """
    if llm_fn is None:
        return 1.0 if answer else 0.0
    prompt = f"Question: {question}\n\nAnswer: {answer}\n\nRate from 0 to 1 how relevant the answer is to the question. Reply with one number."
    response = llm_fn(prompt)
    try:
        return float(response.strip())
    except ValueError:
        return 0.0


# ---------- Example ----------
if __name__ == "__main__":
    question = "What is the refund policy?"
    context = ["Refunds are allowed within 30 days. Contact support@example.com."]
    answer = "Refunds are allowed within 30 days. You can contact support@example.com."

    # With RAGAS (requires OPENAI_API_KEY and pip install ragas)
    try:
        scores = evaluate_ragas_single(question, context, answer)
        print("RAGAS scores:", scores)
    except Exception as e:
        print("RAGAS not run (missing key or install):", e)
        f = score_faithfulness_manual(question, context, answer)
        r = score_answer_relevance_manual(question, answer)
        print("Manual placeholder scores: faithfulness={}, answer_relevance={}".format(f, r))
```

---

## Summary

- **Faithfulness** = how much the answer is supported by the context (RAGAS or LLM score 0–1).  
- **Answer Relevance** = how relevant the answer is to the question (RAGAS or LLM score 0–1).  
- **Code**: Use `ragas.evaluate` with metrics `faithfulness` and `answer_relevance` on a dataset of (question, context, answer); or implement simple LLM-based scorers that return 0–1.
