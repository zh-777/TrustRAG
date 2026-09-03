import os
import re
import sys
import time
from typing import Literal

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from embed import load_model, embed_query
from ingest import chunk_text
from groundcheck import groundcheck
from generate import SYSTEM_PROMPT, build_prompt, build_axiom_prompt, call_llm_dynamic
from backend.auth import (
    ChatRenameRequest,
    ChatSaveRequest,
    LoginRequest,
    RegisterRequest,
    create_user_chat,
    delete_user_chat,
    get_user_chat,
    get_user_from_token,
    init_auth_db,
    list_user_chats,
    login_user,
    logout_token,
    register_user,
    rename_user_chat,
    update_user_chat,
)
from backend.media import extract_upload
from axiom_router import route_query
from reranker import rerank

DATA_PATH = os.path.join(ROOT_DIR, "data", "facts_grounding_dataset.csv")

app = FastAPI(
    title="TrustRAG — RAG-1 Axiom API",
    description="Hybrid general/grounded RAG with Axiom Auto routing and GroundCheck verification",
    version="4.0.0-axiom",
)


LOCAL_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


# Production domains are supplied as a comma-separated environment variable.
# Example: TRUSTRAG_ALLOWED_ORIGINS=https://trustrag.vercel.app
PRODUCTION_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("TRUSTRAG_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(LOCAL_ORIGINS + PRODUCTION_ORIGINS)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_embed_model = None
_dataset = None


@app.on_event("startup")
def startup() -> None:
    init_auth_db()


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = load_model()
    return _embed_model


def get_dataset():
    global _dataset
    if _dataset is None:
        _dataset = pd.read_csv(DATA_PATH)
    return _dataset


class LLMConfig(BaseModel):
    provider: Literal[
        "deepseek",
        "gemini",
        "openai",
        "anthropic",
        "groq",
        "openrouter",
        "ollama",
        "custom",
    ] = "gemini"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class AskRequest(BaseModel):
    document_text: str = Field(default="", max_length=500_000)
    question: str = Field(min_length=1, max_length=8_000)
    top_k: int = Field(default=4, ge=1, le=8)
    axiom_mode: Literal["auto", "general", "grounded", "hybrid", "local"] = "auto"
    retrieval_threshold: float = Field(default=0.20, ge=-1.0, le=1.0)
    source_name: str = Field(default="Live knowledge source", max_length=250)
    llm: LLMConfig


# Only short, clearly social messages bypass RAG. Anything mixed with a content
# question falls through to the strict document-grounded pipeline.
_GREETING_PATTERNS = [
    r"^(hi|hello|hey|hiya|hey there|hello there)[!. ]*$",
    r"^(good morning|good afternoon|good evening)[!. ]*$",
    r"^(how are you|how are you doing|how's it going|hows it going)[?!. ]*$",
    r"^(thanks|thank you|thank u|thanku|thx|ty)[!?. ]*$",
    r"^(bye|goodbye|see you|see ya)[!?. ]*$",
]


def _basic_conversation(question: str) -> str | None:
    q = " ".join(question.lower().strip().split())

    if not any(
        re.fullmatch(pattern, q, flags=re.IGNORECASE)
        for pattern in _GREETING_PATTERNS
    ):
        return None

    if q.startswith(("thank", "thanks", "thx", "ty")):
        return (
            "You're welcome. Send me a source or ask a question "
            "about the evidence whenever you're ready."
        )

    if q.startswith(("bye", "goodbye", "see ")):
        return "Goodbye! Your next source will be ready whenever you return."

    if (
        "how are" in q
        or "how's it going" in q
        or "hows it going" in q
    ):
        return (
            "I'm ready to help. Upload or select a source, "
            "then ask me anything you want verified against it."
        )

    return (
        "Hello! I'm TrustRAG. Upload or select a source and I’ll answer "
        "source-related questions strictly from that evidence."
    )


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""

    return authorization.split(" ", 1)[1].strip()


# ---------------------------------------------------------------------------
# Document overview detection
#
# Broad questions such as "what is in this file?" are different from specific
# factual questions. For small documents, they should see the whole source
# instead of depending only on semantic Top-K retrieval.
# ---------------------------------------------------------------------------
def _is_document_overview_question(question: str) -> bool:
    q = " ".join(question.lower().strip().split())

    overview_phrases = (
        "what is in the file",
        "what's in the file",
        "what is in this file",
        "what's in this file",
        "what is in the text file",
        "what's in the text file",
        "what is in the text",
        "what's in the text",
        "what is in this text",
        "what is in the document",
        "what's in the document",
        "what is this document about",
        "what is this file about",
        "what does this document contain",
        "what does this file contain",
        "tell me about this document",
        "tell me about this file",
        "summarize the document",
        "summarize this document",
        "summarize the file",
        "summarize this file",
        "summarize the text",
        "summary of the document",
        "summary of the file",
        "summary of the text",
    )

    return any(phrase in q for phrase in overview_phrases)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "TrustRAG API",
        "version": "4.0.0-axiom",
    }


# ── Local account endpoints ──────────────────────────────────────────────────

@app.post("/api/auth/register")
def auth_register(req: RegisterRequest):
    return register_user(req)


@app.post("/api/auth/login")
def auth_login(req: LoginRequest):
    return login_user(req)


@app.get("/api/auth/me")
def auth_me(
    authorization: str | None = Header(default=None),
):
    token = _bearer_token(authorization)

    user = get_user_from_token(token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid.",
        )

    return {"user": user}


@app.post("/api/auth/logout")
def auth_logout(
    authorization: str | None = Header(default=None),
):
    logout_token(
        _bearer_token(authorization)
    )

    return {"ok": True}


def _account_user(
    authorization: str | None,
) -> dict:

    token = _bearer_token(authorization)

    user = get_user_from_token(token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid.",
        )

    return user


# ── Per-user chat history ───────────────────────────────────────────────────

@app.get("/api/chats")
def chats_list(
    authorization: str | None = Header(default=None),
):
    user = _account_user(authorization)

    return {
        "chats": list_user_chats(
            user["id"]
        )
    }


@app.post("/api/chats")
def chats_create(
    req: ChatSaveRequest,
    authorization: str | None = Header(default=None),
):
    user = _account_user(authorization)

    return {
        "chat": create_user_chat(
            user["id"],
            req,
        )
    }


@app.get("/api/chats/{chat_id}")
def chats_get(
    chat_id: str,
    authorization: str | None = Header(default=None),
):
    user = _account_user(authorization)

    return {
        "chat": get_user_chat(
            user["id"],
            chat_id,
        )
    }


@app.put("/api/chats/{chat_id}")
def chats_update(
    chat_id: str,
    req: ChatSaveRequest,
    authorization: str | None = Header(default=None),
):
    user = _account_user(authorization)

    return {
        "chat": update_user_chat(
            user["id"],
            chat_id,
            req,
        )
    }


@app.patch("/api/chats/{chat_id}/title")
def chats_rename(
    chat_id: str,
    req: ChatRenameRequest,
    authorization: str | None = Header(default=None),
):
    user = _account_user(authorization)

    return {
        "chat": rename_user_chat(
            user["id"],
            chat_id,
            req.title,
        )
    }


@app.delete("/api/chats/{chat_id}")
def chats_delete(
    chat_id: str,
    authorization: str | None = Header(default=None),
):
    user = _account_user(authorization)

    delete_user_chat(
        user["id"],
        chat_id,
    )

    return {"ok": True}


# ── Dataset endpoints ────────────────────────────────────────────────────────

@app.get("/api/dataset/info")
def dataset_info():
    df = get_dataset()

    return {
        "rows": len(df),
        "min_index": 0,
        "max_index": len(df) - 1,
    }


@app.get("/api/dataset/{row_idx}")
def dataset_row(
    row_idx: int,
):
    df = get_dataset()

    if row_idx < 0 or row_idx >= len(df):
        raise HTTPException(
            status_code=404,
            detail="Dataset row not found",
        )

    row = df.iloc[row_idx]

    return {
        "index": row_idx,
        "document_text": str(
            row["context_document"]
        ),
        "suggested_question": str(
            row["user_request"]
        ),
    }


# ── Multimodal source extraction ─────────────────────────────────────────────

@app.post("/api/media/extract")
async def media_extract(
    file: UploadFile = File(...),
    provider: str = Form("gemini"),
    api_key: str = Form(""),
    model: str = Form("gemini-3.6-flash"),
):
    """Turn text/PDF/image/audio/video into evidence text for the normal RAG pipeline."""

    return await extract_upload(
        file=file,
        provider=provider,
        api_key=api_key,
        model=model,
    )


# ── TrustRAG question answering ──────────────────────────────────────────────

@app.post("/api/ask")
def ask(req: AskRequest):
    """RAG-1 Axiom unified query endpoint."""

    conversational = _basic_conversation(req.question)
    if conversational is not None and req.axiom_mode == "auto":
        return {
            "model_family": "TrustRAG",
            "model": "RAG-1 Axiom",
            "answer": conversational,
            "mode": "general",
            "requested_mode": "auto",
            "route": {
                "mode": "general",
                "reason": "Short social message handled by Axiom's built-in conversation layer.",
                "confidence": 1.0,
                "source_available": bool(req.document_text.strip()),
            },
            "backend": "axiom-built-in",
            "provider": "axiom-built-in",
            "generation_time": 0.0,
            "retrieved": [],
            "sources": [],
            "groundcheck": None,
            "groundedness": None,
            "claims": [],
            "raw_prompt": "",
        }

    route = route_query(
        req.question,
        req.document_text,
        requested_mode=req.axiom_mode,
    )
    mode = route.mode

    # "Axiom Local" keeps automatic query routing but forces local generation.
    provider = "ollama" if req.axiom_mode == "local" else req.llm.provider
    model = req.llm.model
    base_url = req.llm.base_url
    api_key = req.llm.api_key

    if provider == "ollama":
        model = model.strip() or "llama3.2"
        base_url = base_url.strip() or "http://localhost:11434/v1"
    elif not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                f"An API key is required for {provider}. "
                "Choose Axiom Local to run with Ollama without a cloud API key."
            ),
        )

    if provider == "custom" and not base_url.strip():
        raise HTTPException(status_code=400, detail="A Base URL is required for a custom provider.")

    try:
        retrieved = []

        if mode in {"grounded", "hybrid"}:
            if not req.document_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Grounded and Hybrid modes require a loaded knowledge source.",
                )

            embed_model = get_embed_model()
            doc_chunks_text = chunk_text(req.document_text)
            if not doc_chunks_text:
                raise HTTPException(status_code=400, detail="The document could not be chunked.")

            doc_chunks = [
                {
                    "text": text,
                    "chunk_id": f"live_chunk_{i}",
                    "source_id": "live_source",
                    "source_name": req.source_name or "Live knowledge source",
                    "source_row_idx": -1,
                    "chunk_idx": i,
                    "domain": "live",
                    "score": 0.0,
                }
                for i, text in enumerate(doc_chunks_text)
            ]

            if _is_document_overview_question(req.question) and len(doc_chunks) <= 12:
                candidates = []
                for chunk in doc_chunks:
                    item = dict(chunk)
                    item["score"] = 1.0
                    candidates.append(item)
            else:
                chunk_vecs = embed_model.encode(
                    [c["text"] for c in doc_chunks],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
                index = faiss.IndexFlatIP(chunk_vecs.shape[1])
                index.add(chunk_vecs)
                q_vec = embed_query(req.question, embed_model)

                candidate_k = min(max(req.top_k * 3, req.top_k), len(doc_chunks))
                scores, indices = index.search(q_vec, candidate_k)
                candidates = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    chunk = dict(doc_chunks[idx])
                    chunk["score"] = float(score)
                    if float(score) >= req.retrieval_threshold:
                        candidates.append(chunk)

            retrieved = rerank(req.question, candidates, top_k=req.top_k)

            if mode == "grounded" and not retrieved:
                return {
                    "model_family": "TrustRAG",
                    "model": "RAG-1 Axiom",
                    "answer": "I could not find enough evidence in the selected knowledge source to answer this reliably.",
                    "mode": "grounded",
                    "requested_mode": req.axiom_mode,
                    "route": route.as_dict(),
                    "backend": provider,
                    "provider": provider,
                    "generation_time": 0.0,
                    "retrieved": [],
                    "sources": [],
                    "groundcheck": {
                        "overall_verdict": "INSUFFICIENT_EVIDENCE",
                        "faithfulness_score": 0.0,
                        "claims": [],
                    },
                    "groundedness": 0.0,
                    "claims": [],
                    "raw_prompt": "",
                    "abstained": True,
                }

            # If Auto selected hybrid but retrieval found nothing useful, be transparent
            # and fall back to GENERAL instead of pretending the answer is grounded.
            if mode == "hybrid" and not retrieved:
                mode = "general"

        system_prompt, user_prompt = build_axiom_prompt(
            req.question,
            mode,
            retrieved,
        )

        t0 = time.time()
        answer = call_llm_dynamic(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        generation_time = time.time() - t0

        gc_result = None
        claims = []
        groundedness = None

        if mode in {"grounded", "hybrid"} and retrieved:
            gc_result = groundcheck(answer, retrieved)
            groundedness = gc_result.get("faithfulness_score")
            for item in gc_result.get("claims", []):
                verdict = item.get("verdict", "UNSUPPORTED")
                claim_type = "grounded"
                status = verdict.lower()
                # In HYBRID mode, a non-contradictory unsupported sentence is exposed
                # as general/unverified rather than being falsely called grounded.
                if mode == "hybrid" and verdict == "UNSUPPORTED":
                    claim_type = "general"
                    status = "not_evidence_scored"
                claims.append({
                    "text": item.get("claim", ""),
                    "type": claim_type,
                    "status": status,
                    "evidence_score": round(float(item.get("sim_score", 0.0)), 4),
                    "entailment": round(float(item.get("nli", {}).get("entailment", 0.0)), 4),
                    "explanation": item.get("explanation", ""),
                })
        else:
            # General answers are explicitly not evidence-scored.
            from groundcheck import decompose_into_claims
            claims = [
                {
                    "text": claim,
                    "type": "general",
                    "status": "not_evidence_scored",
                    "evidence_score": None,
                    "entailment": None,
                    "explanation": "General model knowledge; no uploaded-source grounding claimed.",
                }
                for claim in decompose_into_claims(answer)
            ]

        sources = [
            {
                "source_id": r.get("source_id", "live_source"),
                "source_name": r.get("source_name", req.source_name),
                "chunk_id": r.get("chunk_id"),
                "score": round(float(r.get("score", 0.0)), 4),
                "rerank_score": round(float(r.get("rerank_score", r.get("score", 0.0))), 4),
            }
            for r in retrieved
        ]

        return {
            "model_family": "TrustRAG",
            "model": "RAG-1 Axiom",
            "answer": answer,
            "mode": mode,
            "requested_mode": req.axiom_mode,
            "route": route.as_dict(),
            "backend": provider,
            "provider": provider,
            "generation_time": round(generation_time, 3),
            "retrieved": retrieved,
            "sources": sources,
            "groundcheck": gc_result,
            "groundedness": groundedness,
            "claims": claims,
            "raw_prompt": user_prompt,
            "abstained": False,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Axiom request failed: {exc}") from exc

