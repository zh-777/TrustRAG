"""
app.py — TrustRAG Streamlit Web Application
===========================================
Supports DeepSeek (free), Ollama (local), Claude (paid) — switchable from sidebar.
Run: streamlit run app.py
"""

import os
import sys
import time
import streamlit as st

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    from embed import load_model
    return load_model()


@st.cache_resource(show_spinner="Loading FAISS vector store…")
def load_vector_store():
    from vector_store import VectorStore
    vs = VectorStore()
    vs.load()
    return vs


@st.cache_resource(show_spinner="Loading NLI model (roberta-large-mnli)…")
def load_nli():
    from groundcheck import _get_nli_pipeline
    return _get_nli_pipeline()


st.set_page_config(
    page_title="TrustRAG — powered by GroundCheck",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.gc-supported    { background:#d4edda; border-left:4px solid #28a745;
                   padding:6px 10px; margin:4px 0; border-radius:4px; }
.gc-unsupported  { background:#fff3cd; border-left:4px solid #ffc107;
                   padding:6px 10px; margin:4px 0; border-radius:4px; }
.gc-contradiction{ background:#f8d7da; border-left:4px solid #dc3545;
                   padding:6px 10px; margin:4px 0; border-radius:4px; }
.gc-label        { font-size:0.75em; font-weight:700; text-transform:uppercase;
                   letter-spacing:0.05em; margin-bottom:2px; }
.gc-explanation  { font-size:0.8em; color:#555; margin-top:4px; font-style:italic; }
.score-badge     { font-size:0.7em; background:#e9ecef; border-radius:10px;
                   padding:1px 7px; margin-left:6px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("TrustRAG")
    st.caption("Powered by GroundCheck")
    st.markdown("---")

    st.markdown("**LLM Backend**")
    backend_choice = st.selectbox(
        "Choose your LLM",
        ["🆓 DeepSeek (Free API)", "🖥️ Ollama (Local, Free)", "💳 Claude (Anthropic)"],
        index=0,
    )

    if backend_choice.startswith("🆓"):
        os.environ["LLM_BACKEND"] = "deepseek"
        api_key_input = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.environ.get("DEEPSEEK_API_KEY", ""),
            help="Free at platform.deepseek.com — no credit card needed",
        )
        if api_key_input:
            os.environ["DEEPSEEK_API_KEY"] = api_key_input
        st.caption("🆓 Get free key → platform.deepseek.com")

    elif backend_choice.startswith("🖥️"):
        os.environ["LLM_BACKEND"] = "ollama"
        ollama_model = st.text_input(
            "Ollama model name",
            value=os.environ.get("OLLAMA_MODEL", "llama3"),
            help="Must be pulled already: ollama pull llama3",
        )
        os.environ["OLLAMA_MODEL"] = ollama_model
        st.caption("🖥️ Runs fully offline. Start Ollama first.")

    elif backend_choice.startswith("💳"):
        os.environ["LLM_BACKEND"] = "claude"
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="Get key at console.anthropic.com",
        )
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.markdown("---")

    doc_source = st.radio(
        "Document source",
        ["📊 FACTS Dataset (pick a row)", "📋 Paste your own text", "📁 Upload .txt file"],
        index=0,
    )

    top_k = st.slider("Chunks retrieved (top-k)", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.markdown("**About GroundCheck**")
    st.caption(
        "GroundCheck uses `roberta-large-mnli` (NLI) + semantic similarity "
        "to independently verify whether each sentence in the answer is "
        "actually supported by the retrieved source text — catching hallucinations."
    )


# ── Document Selection ─────────────────────────────────────────────────────────
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "facts_grounding_dataset.csv")

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

document_text = ""
suggested_question = ""

if doc_source.startswith("📊"):
    df = load_dataset()
    row_idx = st.sidebar.number_input(
        f"Row index (0 – {len(df)-1})",
        min_value=0, max_value=len(df)-1, value=0, step=1,
    )
    document_text      = str(df.iloc[row_idx]["context_document"])
    suggested_question = str(df.iloc[row_idx]["user_request"])

elif doc_source.startswith("📋"):
    document_text = st.sidebar.text_area(
        "Paste document text here",
        height=200,
        placeholder="Paste your source document here…",
    )

elif doc_source.startswith("📁"):
    uploaded = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded:
        document_text = uploaded.read().decode("utf-8")


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🔍 TrustRAG — powered by GroundCheck")
st.markdown(
    "Ask a question. TrustRAG retrieves relevant passages from the source document, "
    "generates an answer, then GroundCheck independently verifies each sentence for faithfulness."
)

col_doc, col_qa = st.columns([1, 1])

with col_doc:
    st.subheader("📄 Source Document")
    if document_text:
        st.text_area("Document preview", value=document_text[:1500], height=250, disabled=True)
    else:
        st.info("Select a document from the sidebar to get started.")

with col_qa:
    st.subheader("❓ Ask a Question")
    question = st.text_area(
        "Your question",
        value=suggested_question,
        height=80,
        placeholder="Type your question here…",
    )
    run_button = st.button("🚀 Ask TrustRAG", type="primary", disabled=not (document_text and question))


# ── Answer Generation + GroundCheck ───────────────────────────────────────────
if run_button and document_text and question:

    backend = os.environ.get("LLM_BACKEND", "deepseek")

    # Check API key is set for cloud backends
    if backend == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        st.error("Please enter your DeepSeek API key in the sidebar. Get one free at platform.deepseek.com")
        st.stop()
    if backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    st.markdown("---")

    with st.spinner("Embedding document and building temporary index…"):
        from embed import embed_query
        from generate import build_prompt, call_llm, SYSTEM_PROMPT
        from groundcheck import groundcheck
        from ingest import chunk_text
        import numpy as np
        import faiss

        embed_model = load_embed_model()

        doc_chunks_text = chunk_text(document_text)
        doc_chunks = [
            {"text": t, "chunk_id": f"live_chunk_{i}", "source_row_idx": -1,
             "chunk_idx": i, "user_request": question, "domain": "live", "score": 0.0}
            for i, t in enumerate(doc_chunks_text)
        ]

        chunk_vecs = embed_model.encode(
            [c["text"] for c in doc_chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = chunk_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(chunk_vecs)

    with st.spinner("Retrieving relevant passages…"):
        q_vec = embed_query(question, embed_model)
        scores, indices = index.search(q_vec, min(top_k, len(doc_chunks)))
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(doc_chunks[idx])
            chunk["score"] = float(score)
            retrieved.append(chunk)

    with st.spinner(f"Generating answer with {backend.upper()}…"):
        user_prompt = build_prompt(question, retrieved)
        t0 = time.time()
        answer = call_llm(SYSTEM_PROMPT, user_prompt)
        gen_time = time.time() - t0

    with st.spinner("Running GroundCheck faithfulness verification…"):
        gc_result = groundcheck(answer, retrieved)

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("---")
    verdict_color = {
        "FAITHFUL"          : "🟢",
        "PARTIALLY_FAITHFUL": "🟡",
        "UNFAITHFUL"        : "🔴",
    }
    icon = verdict_color.get(gc_result["overall_verdict"], "⚪")

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    col_v1.metric("GroundCheck Verdict", f"{icon} {gc_result['overall_verdict']}")
    col_v2.metric("Faithfulness Score",  f"{gc_result['faithfulness_score']:.0%}")
    col_v3.metric("Generation Time",     f"{gen_time:.1f}s")
    col_v4.metric("LLM Backend",         backend.upper())

    st.subheader("📝 Answer (GroundCheck annotated)")

    verdict_css = {
        "SUPPORTED"    : "gc-supported",
        "UNSUPPORTED"  : "gc-unsupported",
        "CONTRADICTION": "gc-contradiction",
    }
    verdict_labels = {
        "SUPPORTED"    : "✓ Supported",
        "UNSUPPORTED"  : "? Not found in source",
        "CONTRADICTION": "✗ Contradicts source",
    }

    html_parts = []
    for cr in gc_result["claims"]:
        css   = verdict_css.get(cr["verdict"], "gc-unsupported")
        label = verdict_labels.get(cr["verdict"], cr["verdict"])
        nli   = cr["nli"]
        sim   = cr["sim_score"]
        expl_html = (
            f'<div class="gc-explanation">{cr["explanation"]}</div>'
            if cr["explanation"] else ""
        )
        scores_html = (
            f'<span class="score-badge">E:{nli["entailment"]:.2f} '
            f'N:{nli["neutral"]:.2f} C:{nli["contradiction"]:.2f} '
            f'sim:{sim:.2f}</span>'
        )
        html_parts.append(
            f'<div class="{css}">'
            f'<div class="gc-label">{label}{scores_html}</div>'
            f'{cr["claim"]}'
            f'{expl_html}'
            f'</div>'
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    with st.expander("🔬 GroundCheck Detail (per-claim breakdown)"):
        for i, cr in enumerate(gc_result["claims"], 1):
            st.markdown(f"**Claim {i}:** {cr['claim']}")
            cols = st.columns(5)
            cols[0].metric("Verdict",       cr["verdict"])
            cols[1].metric("P(Entail)",     f"{cr['nli']['entailment']:.2%}")
            cols[2].metric("P(Neutral)",    f"{cr['nli']['neutral']:.2%}")
            cols[3].metric("P(Contradict)", f"{cr['nli']['contradiction']:.2%}")
            cols[4].metric("Semantic Sim",  f"{cr['sim_score']:.2%}")
            if cr["explanation"]:
                st.warning(cr["explanation"])
            st.markdown("---")

    with st.expander(f"📚 Retrieved Source Passages (top {len(retrieved)})"):
        for i, c in enumerate(retrieved, 1):
            st.markdown(f"**[{i}]** `{c['chunk_id']}` — relevance: `{c['score']:.3f}`")
            st.text(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
            st.markdown("---")

    with st.expander("🛠️ Raw LLM Prompt (transparency)"):
        st.text(user_prompt)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TrustRAG · powered by GroundCheck · "
    "Built by Zohaib Hassan · Boston Institute of Analytics · "
    "Dataset: Google DeepMind FACTS Grounding (CC-BY-4.0)"
)