# Q19: Streaming + Async Retrieval for 5-Second Latency

## The Question

> You are seeing **5-second latencies**. Write a **strategy** to implement **streaming** and **asynchronous retrieval** to improve **perceived speed**.

---

## Concepts

- **5-second latency**: Often = time to **retrieve** (e.g. 1–2 s) + time to **generate** (e.g. 3–4 s). Users feel the full 5 s before they see anything.
- **Streaming**: Instead of waiting for the **full** answer, the model **streams** tokens as they’re generated. So the user sees the first word in ~1–2 s (after retrieval + first token), and the rest appears incrementally. **Perceived** latency drops.
- **Asynchronous retrieval**: Start **retrieval** and **generation prep** in parallel where possible. E.g. while the first batch of chunks is being fetched, you can already start the LLM call with a "thinking" placeholder, or pipeline retrieval → stream generation so the user sees output as soon as the first token is ready.
- **Strategy**: (1) **Stream** the LLM response (use the API’s stream=True). (2) **Don’t block** on slow steps: run retrieval in a thread/async task; when done, pass chunks to the generator and start streaming. (3) **Optional**: Cache embeddings or index on a fast server to reduce retrieval time.

---

## Approach

1. **Measure**: Identify where time goes (retrieval vs generation). If generation is 3–4 s, streaming will help a lot; if retrieval is 3 s, optimize retrieval or run it in parallel with something else.

2. **Streaming**:  
   - Use your LLM API’s **streaming** option (e.g. `stream=True`).  
   - Send tokens to the client as they arrive (e.g. Server-Sent Events or WebSocket).  
   - The user sees the first token in ~(retrieval_time + time_to_first_token), not full 5 s.

3. **Async retrieval**:  
   - **Option A**: Start retrieval **asynchronously** (e.g. `asyncio` or a background thread). When retrieval completes, start the LLM call (streaming). Total time is still retrieval + generation, but you can show a "Searching..." state and then stream.  
   - **Option B**: If you have multiple independent steps (e.g. two retrievers), run them **in parallel** and merge; then stream generation.  
   - **Option C**: Pre-warm or cache: keep hot paths fast so retrieval is under 500 ms.

4. **Pipeline**: Request → (async) retrieve → when done → start streaming generate → stream tokens to user. User sees "Searching..." then first token, then rest of answer.

5. **Optional**: **Speculative** or **progressive** display: show "Retrieved N chunks" as soon as retrieval finishes, then stream the answer. So perceived progress is in two steps (retrieval done → answer streaming).

---

## Python Implementation (Conceptual)

```python
# Strategy: async retrieval + streaming generation. Pseudocode with asyncio.

import asyncio
from typing import AsyncIterator

async def retrieve_async(query: str, retriever_fn, top_k: int = 5) -> list[str]:
    """Run retriever in executor so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: retriever_fn(query, top_k)[0])

async def stream_generate_async(query: str, chunks: list[str], stream_llm_fn) -> AsyncIterator[str]:
    """
    stream_llm_fn(question, chunks) yields token strings.
    """
    async for token in stream_llm_fn(query, chunks):
        yield token

async def rag_streaming_pipeline(
    query: str,
    retriever_fn,
    stream_llm_fn,
    top_k: int = 5,
) -> AsyncIterator[str]:
    """
    1. Start retrieval (async).
    2. When retrieval is done, start streaming generation.
    3. Yield tokens as they arrive.
    """
    chunks = await retrieve_async(query, retriever_fn, top_k)
    async for token in stream_generate_async(query, chunks, stream_llm_fn):
        yield token


# ---------- Sync version: show the order of operations ----------
def rag_streaming_sync_strategy():
    """
    Strategy (sync version):
    1. Retrieve (blocking). Consider: cache, faster index, smaller top_k.
    2. Call LLM with stream=True.
    3. For each token chunk from the stream, send to client (e.g. SSE).
    User sees: [wait retrieval] -> [first token] -> [rest of tokens].
    Perceived latency = time until first token (retrieval + TTFT), not full response.
    """
    pass


# ---------- Example: mock async retrieval + mock streaming ----------
if __name__ == "__main__":
    async def mock_retriever(q, top_k=5):
        await asyncio.sleep(0.5)  # simulate 500ms retrieval
        return ["Chunk 1.", "Chunk 2."]

    async def mock_stream_llm(q, chunks):
        answer = "This is the streamed answer."
        for word in answer.split():
            await asyncio.sleep(0.1)
            yield word + " "

    async def main():
        async for token in rag_streaming_pipeline("What is X?", mock_retriever, mock_stream_llm):
            print(token, end="", flush=True)
        print()

    asyncio.run(main())
```

---

## Summary

- **Streaming**: Use the LLM’s **stream=True** and send tokens to the user as they’re generated → **perceived** latency = time to first token (retrieval + TTFT), not full 5 s.  
- **Async retrieval**: Run retrieval in a thread/async task; when done, start streaming generation. Optionally run multiple retrievers in parallel.  
- **Pipeline**: Retrieve (async) → then stream generate → user sees progress quickly and then the answer incrementally.
