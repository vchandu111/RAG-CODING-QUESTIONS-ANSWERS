# Q10: System Prompt — No Internal Knowledge Without Context

## The Question

> **Write a system prompt** that strictly prevents the LLM from using its **internal knowledge** if the answer is **not present** in the retrieved context.

---

## Concepts

- **RAG**: We give the model **retrieved chunks** (context) and ask it to answer **only from that context**. If the answer isn’t in the context, we want it to say "I don’t know" instead of making something up or using its training data.
- **System prompt**: Instructions we send once (e.g. as "system" role) that define the model’s behavior. We can say: "Only use the provided context. If the answer is not in the context, say so."
- **Guardrails**: Rules like "never cite a source that wasn’t in the context" and "never invent facts" make the RAG answer more trustworthy.

---

## Approach

1. **State the rule clearly**: "You must answer ONLY using the provided context. Do not use your internal knowledge to add facts not in the context."

2. **Define what to do when the answer is missing**: "If the context does not contain enough information to answer the question, say: 'I cannot find this in the provided documents.' Do not guess or infer."

3. **Optional**: "Quote or cite only from the context. Do not invent sources."

4. **Format**: Put this in the **system** message (or the first user message) so every turn follows these rules.

5. **In code**: When you call the LLM, pass **context** (retrieved chunks) in the **user** message and keep the system prompt fixed.

---

## Python Implementation

```python
# System prompt that grounds the model in context only

SYSTEM_PROMPT_NO_INTERNAL_KNOWLEDGE = """You are a helpful assistant that answers questions ONLY using the provided context below.

Rules:
1. Use ONLY information from the "Context" section to answer. Do not use your internal knowledge or training data to add facts that are not in the context.
2. If the context does not contain enough information to answer the question, respond with exactly: "I cannot find this in the provided documents."
3. Do not guess, assume, or make up information. If you are unsure, say you cannot find it.
4. When you answer, you may quote from the context. Do not cite or mention sources that are not in the context.
5. Keep answers concise and grounded in the context."""

def build_user_message(question: str, context_chunks: list[str]) -> str:
    """Build the user message with context + question."""
    context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_chunks)])
    return f"""Context:
{context_block}

Question: {question}

Answer using only the context above. If the answer is not in the context, say "I cannot find this in the provided documents." """


# ---------- Example: how to call an LLM (pseudo-code) ----------
if __name__ == "__main__":
    # With OpenAI-style API:
    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT_NO_INTERNAL_KNOWLEDGE},
    #     {"role": "user", "content": build_user_message("What is the refund policy?", ["Refunds allowed within 30 days."])},
    # ]
    # response = client.chat.completions.create(model="gpt-4", messages=messages)

    # Demo: just print the prompts
    context = ["Refunds are available within 30 days. Contact support@example.com."]
    user_msg = build_user_message("What is the refund policy?", context)
    print("=== System prompt ===")
    print(SYSTEM_PROMPT_NO_INTERNAL_KNOWLEDGE[:200], "...")
    print("\n=== User message (with context) ===")
    print(user_msg)
```

---

## Summary

- **System prompt**: "Answer ONLY from the provided context. If not in context, say 'I cannot find this in the provided documents.' Do not use internal knowledge."
- **User message**: Include the retrieved **context** + the **question** so the model has everything in one place.
- **Result**: The model stays grounded and avoids hallucinating when the answer isn’t in the docs.
