# python-QA-agent (CLI)

A command-line question-answering agent that answers questions about a water leak detection product catalogue **using only what the catalogue says**. If the document does not contain the answer, the agent says so.

It is a small RAG (Retrieval-Augmented Generation) system: it retrieves the relevant parts of a markdown file, then asks an LLM running on [Ollama](https://ollama.com) to answer from those parts only.


## How it works

**At startup (once)**
1. **Clean** the markdown: strip image links and page comments, normalise bullets, fix OCR typos (`ofering` → `offering`).
2. **Chunk** it by the document's own structure: one chunk per product (`###`), each prefixed with its category and any section-level text that applies to it, plus one overview chunk per category.
3. **Index** the chunks with BM25 keyword search and, if available, embeddings.

**For every question**
1. **Force in** any product named in the question (typo-tolerant, e.g. `aquascop 550`).
2. **Retrieve** with hybrid search: BM25 + embeddings merged with Reciprocal Rank Fusion. List and recommendation questions retrieve more chunks(10 instead of 6).
3. **Generate** with a strict system prompt at temperature 0: answer only from the excerpts, use a fixed refusal sentence when the answer is missing, never turn "launching Q2 2026" into "available".
4. **Guard**: every word of the answer is checked against the retrieved text and the question. Unsupported wording triggers one rewrite, and any sentence that still contains it is deleted.
5. **Print** the answer plus `[Sources: ...]`.

### Why chunk by product?
Each product is a short, self-contained description, so a whole product is the natural retrieval unit. Fixed-size splits cut products in half and mix neighbours. Copying the section-level bullets (for example "no drilling, even through cast iron lids") into each product chunk keeps them attached to the right products, and the product name in every chunk stops features from being attributed to the wrong product (such as `accelerometer` vs `hydrophone`).

## Setup

Requires **Python 3.9+**. The agent itself uses only the standard library.

```bash
# optional but recommended: local embeddings for better retrieval on vague questions
pip install sentence-transformers
```

1. Put the catalogue markdown in the project folder as `product_overview.md`. Any `.md` file with "product" and "overview" in its name is also found, or pass `--doc PATH`.
2. Create a `.env` file next to the script (`api.env` also works):
   ```
   OLLAMA_API_KEY=your_key_here
   LLM_MODEL=glm-5.3-flash:cloud
   ```
3. Run:
   ```bash
   python qa_agent_onlyollama.py
   ```
   Type `exit` or `quit` to leave.

| Setting | Meaning |
|---|---|
| `OLLAMA_API_KEY` | Uses Ollama's hosted API (`https://ollama.com`). Without it, the agent uses a local Ollama server at `localhost:11434`. |
| `LLM_MODEL` / `--model` | Model name. Defaults: `gpt-oss:120b` (hosted), `llama3.1:8b` (local, run `ollama pull llama3.1:8b` first). |
| `EMBED_BACKEND` | `auto` (default), `st`, `ollama` or `none`. Without an embedding backend the agent uses keyword search only. |
| `--doc PATH` | Path to the markdown knowledge base. |



