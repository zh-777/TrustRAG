
# TrustRAG — RAG-1 Axiom

**RAG-1 Axiom** is the first named TrustRAG architecture. It upgrades the previous
strict document-only RAG experience into a hybrid, evidence-aware assistant.

## Axiom response modes

- **Axiom Auto** — routes each question to General, Grounded, or Hybrid.
- **Grounded Only** — source evidence is mandatory; Axiom abstains if retrieval is weak.
- **General** — normal model knowledge; no uploaded-source grounding is claimed.
- **Hybrid** — combines source-backed facts with clearly exposed general reasoning.
- **Axiom Local** — keeps Auto routing but forces local Ollama-compatible generation.

## What changed from the previous TrustRAG ZIP

- Added `src/axiom_router.py` for deterministic, inspectable query routing.
- Added `src/reranker.py` for FAISS candidate reranking.
- Upgraded `/api/ask` to return `RAG-1 Axiom`, route metadata, response mode,
  source provenance, claim types, GroundCheck results, groundedness, and abstention state.
- Added retrieval thresholds and wider candidate retrieval before reranking.
- Added mode-aware generation prompts for General, Grounded, and Hybrid answers.
- Preserved all existing multi-LLM providers and the local Ollama path.
- Updated the React UI with Axiom identity, mode controls, mode badges, and clearer
  General/Grounded/Hybrid messaging.
- Groundedness is presented as evidence support, never as universal truth.


## Run locally

### 1) Backend

From the project root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

The first GroundCheck request may take longer because the embedding/NLI models need to load.

### 2) Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the local Vite URL, normally `http://localhost:5173`.

### Optional frontend API URL

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

## Bring your own LLM

Choose a provider in the left panel, paste your own API key, and optionally edit the model name. Ollama uses no cloud API key. The **Custom** option accepts any OpenAI-compatible Base URL + model + key.

## Existing research pipeline

The original `src/`, `evaluation/`, and `data/` pipeline is preserved. The former `app.py` Streamlit file is also retained for reference, but the recommended UI is now `frontend/` with `backend/main.py`.
