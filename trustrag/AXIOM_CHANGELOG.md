# RAG-1 Axiom — Change Log

This build implements the initial Axiom architecture described in the Axiom overview.

## Implemented
- General / Grounded / Hybrid query modes.
- Axiom Auto deterministic query router.
- Grounded Only abstention.
- Axiom Local mode using the existing Ollama-compatible provider.
- Multi-provider generation preserved.
- Wider FAISS candidate retrieval + lightweight reranking.
- Retrieval threshold support.
- Richer live-source provenance metadata.
- Mode-aware generation prompts.
- Claim metadata distinguishing grounded vs general output.
- GroundCheck retained for evidence-dependent responses.
- Professional Axiom branding and mode controls in the React frontend.

## Intentionally deferred
- Learned query router.
- Cross-encoder reranker.
- Hybrid BM25 + vector search.
- Fine-tuned Axiom language model.
- Multi-hop retrieval.
- Tool/agent system.

Those belong to Verum, Lumen, Nexus, or later releases rather than RAG-1 Axiom.

## Axiom UI 2.0 — Conversational Interface Refresh
- Reworked the frontend toward a cleaner ChatGPT/Claude-style conversational workspace.
- Added a persistent Axiom mode selector in the top bar: Auto, Grounded, Hybrid, General, Local.
- Added a compact mode selector inside the composer for fast switching while chatting.
- Replaced the previous product logo usage in the assistant experience with the Axiom circular SVG mark.
- Added an animated Axiom thinking state used while a response is being generated.
- Simplified chat bubbles, composer, top navigation, sidebar surfaces, spacing, and responsive behavior.
- Removed the animated star-field from the primary experience for a more professional product surface.
- Backend/RAG behavior and API request shape remain unchanged.
