# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import logging
import json
import asyncio

from backend.rag_pipeline import generate_with_context

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("faq-assistant")

# ---- FastAPI Setup ----
app = FastAPI(
    title="AI-Powered FAQ Assistant",
    description="Backend for FAQ chatbot",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Session Memory (minimal) ----
SESSION_HISTORY: Dict[str, List[Dict[str, str]]] = {}

# ---- Payload Models ----
class AskPayload(BaseModel):
    question: str
    session_id: Optional[str] = None
    retrieval_mode: Optional[str] = None
    top_k: Optional[int] = None


# ---------- Sanitizer (server-side guardrail) ----------
def sanitize_answer(text: str) -> str:
    """
    Remove any QA/Context echoes before returning/streaming.
    """
    if not text:
        return ""

    s = text.strip()

    # If the model emitted a transcript starting with Q: ... A: ...
    if s[:2].upper() == "Q:" and "A:" in s:
        s = s.rsplit("A:", 1)[-1].strip()

    # Remove anything after these markers if echoed
    for marker in ["Context:", "CONTEXT:", "\nUser:", "\nAssistant:", "\nUSER:", "\nASSISTANT:"]:
        if marker in s:
            s = s.split(marker)[0].strip()

    if s[:2].upper() == "A:":
        s = s[2:].strip()

    return " ".join(s.split())


# ---- Health Check ----
@app.get("/")
def health():
    return {"status": "ok", "message": "FAQ Assistant API is running."}


# ---- Standard Ask Endpoint (non-streaming) ----
@app.post("/ask")
def ask(payload: AskPayload):
    try:
        session_id = payload.session_id or str(uuid.uuid4())
        if session_id not in SESSION_HISTORY:
            SESSION_HISTORY[session_id] = []

        # We do NOT stuff chat history into the "question" anymore
        # to avoid the model echoing Q:/A:. Keep retrieval clean.
        result = generate_with_context(
            question=payload.question,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
        )
        raw_answer = result.get("answer", "")
        answer = sanitize_answer(raw_answer)

        # Update session history
        SESSION_HISTORY[session_id].append(
            {"user": payload.question, "assistant": answer}
        )

        return {
            "session_id": session_id,
            "answer": answer,
            "history": SESSION_HISTORY[session_id],
        }
    except Exception as e:
        log.exception("Ask failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ask failed: {e}")


# ---- Streaming Ask Endpoint ----
@app.post("/ask/stream")
async def ask_stream(payload: AskPayload):
    """
    Streams a *cleaned* answer to the UI token-by-token.
    """
    try:
        session_id = payload.session_id or str(uuid.uuid4())
        if session_id not in SESSION_HISTORY:
            SESSION_HISTORY[session_id] = []

        # Generate complete answer first (retrieval + LLM)
        result = generate_with_context(
            question=payload.question,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
        )
        raw_answer = result.get("answer", "")
        clean = sanitize_answer(raw_answer)

        # Update session memory
        SESSION_HISTORY[session_id].append(
            {"user": payload.question, "assistant": clean}
        )

        # Stream tokens from the cleaned text
        async def token_generator():
            buffer = ""
            for token in clean.split():
                buffer += token + " "
                yield json.dumps({"token": token}) + "\n"
                await asyncio.sleep(0.03)  # small type-on effect
            yield json.dumps({
                "final": buffer.strip(),
                "session_id": session_id,
            }) + "\n"

        return StreamingResponse(token_generator(), media_type="application/json")

    except Exception as e:
        log.exception("Streaming Ask failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Streaming Ask failed: {e}")
