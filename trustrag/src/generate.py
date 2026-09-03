"""
src/generate.py — TrustRAG Answer Generation
============================================
Supports 3 LLM backends — switch with the LLM_BACKEND env variable:

  LLM_BACKEND=deepseek  → DeepSeek V3 (FREE tier, recommended)
  LLM_BACKEND=ollama    → Local Ollama (llama3, fully offline)
  LLM_BACKEND=claude    → Anthropic Claude Haiku (paid)

Default: deepseek
"""

import os
from sentence_transformers import SentenceTransformer

from embed import embed_query, load_model
from vector_store import VectorStore

# ── configuration ─────────────────────────────────────────────────────────────
TOP_K       = 5
MAX_TOKENS  = 1024

LLM_BACKEND = os.getenv("LLM_BACKEND", "deepseek").lower()

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

CLAUDE_MODEL = "claude-haiku-4-5"
# ──────────────────────────────────────────────────────────────────────────────



PROVIDER_DEFAULTS = {
    "deepseek": {"model": "deepseek-chat", "base_url": "https://api.deepseek.com"},
    "openai": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
    "gemini": {
    "model": "gemini-3.6-flash","base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "anthropic": {"model": "claude-haiku-4-5", "base_url": ""},
    "groq": {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    "openrouter": {"model": "openai/gpt-4.1-mini", "base_url": "https://openrouter.ai/api/v1"},
    "ollama": {"model": "llama3.2", "base_url": "http://localhost:11434/v1"},
    "custom": {"model": "", "base_url": ""},
}


def call_llm_dynamic(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> str:
    """Call a user-selected LLM without storing their key on the server."""
    provider = provider.lower().strip()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported provider: {provider}")

    defaults = PROVIDER_DEFAULTS[provider]
    chosen_model = model.strip() or defaults["model"]
    chosen_base = base_url.strip() or defaults["base_url"]

    if provider == "anthropic":
        import anthropic
        if not api_key:
            raise EnvironmentError("Anthropic API key is required.")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=chosen_model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    from openai import OpenAI
    if provider == "ollama":
        client = OpenAI(base_url=chosen_base, api_key="ollama")
    else:
        if not api_key:
            raise EnvironmentError(f"API key is required for {provider}.")
        client = OpenAI(base_url=chosen_base, api_key=api_key)

    response = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return response.choices[0].message.content or ""


AXIOM_GENERAL_PROMPT = """You are RAG-1 Axiom, the TrustRAG assistant.
Answer the user's general question using the capabilities of the selected language model.
Do not imply that the answer came from an uploaded source. If the user asks for current,
high-stakes, or source-specific facts that you cannot verify, state the limitation clearly.
Be useful, concise, and technically precise.
"""

AXIOM_GROUNDED_PROMPT = """You are RAG-1 Axiom in GROUNDED mode.

Rules:
1. Source-dependent factual claims must be supported by the SOURCE PASSAGES.
2. Do not silently fill missing source facts with general model knowledge.
3. If the evidence is insufficient, say so explicitly.
4. Prefer citing passage markers like [1], [2] when making source-backed claims.
5. Keep the response concise and factual.
"""

AXIOM_HYBRID_PROMPT = """You are RAG-1 Axiom in HYBRID mode.

Use SOURCE PASSAGES for source-specific facts and clearly distinguish any broader
general explanation or reasoning. Cite passage markers [1], [2], etc. for grounded
claims. Never present general model knowledge as if it came from the source. If a
source-specific fact is unsupported, say that the evidence is insufficient.
"""

def build_axiom_prompt(
    question: str,
    mode: str,
    retrieved_chunks: list[dict] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for an Axiom response mode."""
    mode = (mode or "general").lower()
    retrieved_chunks = retrieved_chunks or []

    if mode == "general":
        return AXIOM_GENERAL_PROMPT, f"USER QUESTION:\n{question}"

    passages_block = "\n\n".join(
        f"[{i+1}] (source: {c.get('source_name') or c.get('chunk_id', 'source')}, "
        f"relevance: {float(c.get('rerank_score', c.get('score', 0.0))):.3f})\n{c.get('text', '')}"
        for i, c in enumerate(retrieved_chunks)
    )
    system = AXIOM_GROUNDED_PROMPT if mode == "grounded" else AXIOM_HYBRID_PROMPT
    instruction = (
        "Answer from the evidence. If evidence is insufficient, abstain."
        if mode == "grounded"
        else "Use evidence for source facts and label broader reasoning clearly."
    )
    user = (
        f"USER QUESTION:\n{question}\n\n"
        f"SOURCE PASSAGES:\n{passages_block}\n\n"
        f"INSTRUCTION:\n{instruction}"
    )
    return system, user


SYSTEM_PROMPT = """You are TrustRAG, a document-grounded question answering assistant.

STRICT RULES — follow all of them without exception:
1. Answer ONLY using information explicitly present in the SOURCE PASSAGES provided below.
2. Do NOT use any outside knowledge, general facts, or anything not stated in the passages.
3. If the passages do not contain enough information to answer the question, say exactly:
   "I cannot answer this question from the provided document."
4. Do not speculate, infer, or extrapolate beyond what the passages directly state.
5. When you quote or closely paraphrase the source, that is ideal — it keeps your answer grounded.
6. Keep your answer concise and factual. Avoid filler phrases.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    passages_block = "\n\n".join(
        f"[{i+1}] (source: {c['chunk_id']}, relevance: {c['score']:.3f})\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    return (
        f"QUESTION: {question}\n\n"
        f"SOURCE PASSAGES:\n{passages_block}\n\n"
        f"Please answer the question using ONLY the source passages above."
    )


def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY is not set.\n"
            "Get a free key at https://platform.deepseek.com\n"
            "Then run: set DEEPSEEK_API_KEY=sk-..."
        )
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return response.choices[0].message.content


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def call_claude(system_prompt: str, user_prompt: str) -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def call_llm(system_prompt: str, user_prompt: str) -> str:
    backend = os.getenv("LLM_BACKEND", LLM_BACKEND).lower()
    if backend == "deepseek":
        return call_deepseek(system_prompt, user_prompt)
    elif backend == "ollama":
        return call_ollama(system_prompt, user_prompt)
    elif backend == "claude":
        return call_claude(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: '{backend}'. Choose: deepseek | ollama | claude")


def generate_answer(
    question: str,
    vs: VectorStore,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> dict:
    q_vec    = embed_query(question, embed_model)
    retrieved = vs.search(q_vec, top_k=top_k)

    if not retrieved:
        return {
            "question"        : question,
            "answer"          : "I cannot answer this question from the provided document.",
            "retrieved_chunks": [],
            "prompt"          : "",
            "backend"         : "none",
        }

    user_prompt = build_prompt(question, retrieved)
    backend     = os.getenv("LLM_BACKEND", LLM_BACKEND).lower()
    answer      = call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "question"        : question,
        "answer"          : answer,
        "retrieved_chunks": retrieved,
        "prompt"          : user_prompt,
        "backend"         : backend,
    }


if __name__ == "__main__":
    import sys
    backend = os.getenv("LLM_BACKEND", LLM_BACKEND)
    print(f"Using LLM backend: {backend.upper()}")
    vs = VectorStore()
    vs.load()
    embed_model = load_model()
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What are my risk factors for dementia if I have hearing loss and eat poorly?"
    )
    print(f"\nQuestion: {question}")
    print("─" * 70)
    result = generate_answer(question, vs, embed_model)
    print(f"\nAnswer (via {result['backend']}):\n{result['answer']}")
    for i, c in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {c['chunk_id']}  score={c['score']:.4f}")