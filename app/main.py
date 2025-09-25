"""
main.py

This module defines the FastAPI backend for the AI-Powered FAQ Assistant.
It provides endpoints for:
1. Streaming Q&A responses via POST `/ask/stream`.
2. Health check via GET `/`.

The backend integrates with the RAG pipeline (`backend.rag_pipeline.generate_answer`)
to retrieve relevant FAQ context and generate answers using a language model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import logging
import json
import asyncio

from backend.rag_pipeline import generate_answer

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("faq-assistant")

# ---- FastAPI Setup ----
app = FastAPI(
    title="AI-Powered FAQ Assistant",
    description="Backend for FAQ chatbot",
    version="1.3.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # NOTE: In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Session Memory ----
# Minimal in-memory session storage to maintain chat history by session_id
SESSION_HISTORY: Dict[str, List[Dict[str, str]]] = {}

# ---- Payload Models ----
class AskPayload(BaseModel):
    """
    Request body schema for the /ask and /ask/stream endpoints.

    Attributes:
        question (str): The user's query.
        session_id (Optional[str]): Optional unique session identifier.
        retrieval_mode (Optional[str]): Retrieval strategy, e.g., 'bm25', 'semantic', or 'hybrid'.
        top_k (Optional[int]): Number of top FAQs to retrieve.
    """
    question: str
    session_id: Optional[str] = None
    retrieval_mode: Optional[str] = None
    top_k: Optional[int] = None


# ---------- Sanitizer (server-side guardrail) ----------
def sanitize_answer(text: str) -> str:
    """
    Clean up the raw model-generated answer by removing unwanted echoes
    such as question text, context, or role labels like 'User:' and 'Assistant:'.

    Args:
        text (str): The raw answer text from the LLM.

    Returns:
        str: Sanitized answer with only the relevant response content.
    """
    if not text:
        return ""

    s = text.strip()

    # If the model returned a Q/A transcript like "Q: ... A: ..."
    if s[:2].upper() == "Q:" and "A:" in s:
        s = s.rsplit("A:", 1)[-1].strip()

    # Remove extra markers or role labels if present
    for marker in ["Context:", "CONTEXT:", "\nUser:", "\nAssistant:", "\nUSER:", "\nASSISTANT:"]:
        if marker in s:
            s = s.split(marker)[0].strip()

    # Strip leading "A:" if present
    if s[:2].upper() == "A:":
        s = s[2:].strip()

    # Normalize whitespace
    return " ".join(s.split())


# ---- Health Check Endpoint ----
@app.get("/")
def health():
    """
    Simple health check endpoint to verify API is running.

    Returns:
        dict: Status message indicating the API is operational.
    """
    return {"status": "ok", "message": "FAQ Assistant API is running."}


# ---- Streaming Ask Endpoint ----
@app.post("/ask/stream")
async def ask_stream(payload: AskPayload):
    """
    Handle a single user question and stream back the answer token-by-token.

    Steps:
        1. Validate session_id or generate a new one.
        2. Pass the question to the RAG pipeline.
        3. Sanitize the LLM-generated answer.
        4. Update session history.
        5. Stream tokens gradually to simulate typing.

    Args:
        payload (AskPayload): Incoming request containing the question and optional parameters.

    Returns:
        StreamingResponse: Stream of JSON tokens representing the generated answer.
    """
    try:
        session_id = payload.session_id or str(uuid.uuid4())
        if session_id not in SESSION_HISTORY:
            SESSION_HISTORY[session_id] = []

        # Generate complete answer (retrieval + LLM)
        result = generate_answer(
            question=payload.question,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
        )
        raw_answer = result.get("answer", "")
        clean = sanitize_answer(raw_answer)

        # Update session history
        SESSION_HISTORY[session_id].append(
            {"user": payload.question, "assistant": clean}
        )

        # Stream tokens asynchronously
        async def token_generator():
            """
            Generator that yields tokens of the final answer one by one.
            Creates a type-on effect in the frontend.
            """
            buffer = ""
            for token in clean.split():
                buffer += token + " "
                yield json.dumps({"token": token}) + "\n"
                await asyncio.sleep(0.03)  # Small delay for typing simulation
            # Yield final message
            yield json.dumps({
                "final": buffer.strip(),
                "session_id": session_id,
            }) + "\n"

        return StreamingResponse(token_generator(), media_type="application/json")

    except Exception as e:
        log.exception("Streaming Ask failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Streaming Ask failed: {e}")
