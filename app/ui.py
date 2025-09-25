# app/ui.py
import os
import uuid
import json
import requests
import streamlit as st

st.set_page_config(page_title="AskFAQ", page_icon="🤖", layout="wide")

# ---------- Cleaning helper ----------
def clean_answer(raw_text: str) -> str:
    """
    Clean up the model response to remove any unwanted prefixes like Q:/A: or context echoes.
    """
    text = (raw_text or "").strip()

    if text[:2].upper() == "Q:" and "A:" in text:
        text = text.rsplit("A:", 1)[-1].strip()

    for marker in ["Context:", "CONTEXT:", "\nUser:", "\nAssistant:", "\nUSER:", "\nASSISTANT:"]:
        if marker in text:
            text = text.split(marker)[0].strip()

    if text[:2].upper() == "A:":
        text = text[2:].strip()

    return " ".join(text.split())

# -------------------
# Backend API config
# -------------------
API_URL = os.getenv("FAQ_API_URL", "http://127.0.0.1:8000")
ASK_STREAM_URL = f"{API_URL.rstrip('/')}/ask/stream"

# -------------------
# Session management
# -------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

# -------------------
# Sidebar
# -------------------
st.sidebar.title("AI-Powered FAQ Assistant")
st.sidebar.write(
    "Ask questions about policies, leave, payroll, and more. "
    "The assistant uses your FAQ KB."
)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
    st.session_state.session_id = str(uuid.uuid4())

# -------------------
# Main Chat Window
# -------------------
st.title("AskFAQ")

# Render previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box
user_text = st.chat_input("Type your question…")

if user_text:
    # Add user's message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Streaming assistant response
    with st.chat_message("assistant"):
        response_box = st.empty()
        full_answer = ""

        try:
            with requests.post(
                ASK_STREAM_URL,
                json={
                    "question": user_text,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=120,
            ) as r:
                if r.status_code == 200:
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue

                        if "token" in data:
                            full_answer += data["token"] + " "
                            response_box.markdown(full_answer)
                        elif "final" in data:
                            full_answer = data["final"]
                            response_box.markdown(full_answer)
                else:
                    full_answer = f"Server error {r.status_code}: {r.text}"
                    response_box.error(full_answer)
        except Exception as e:
            full_answer = f"Request failed: {e}"
            response_box.error(full_answer)

    # Clean the final answer and **only save to history**, no second rendering
    cleaned = clean_answer(full_answer)
    st.session_state.messages.append({"role": "assistant", "content": cleaned})
