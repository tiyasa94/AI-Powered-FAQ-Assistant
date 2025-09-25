"""
ui.py

This module defines the Streamlit frontend for the AI-Powered FAQ Assistant.
It provides a clean, interactive interface for users to ask questions about
HR policies, leave, payroll, and other employee-related topics.

Features:
1. Displays a fixed header and a sidebar for chat controls.
2. Provides quick suggestion buttons for frequently asked questions.
3. Allows free-text chat input for custom questions.
4. Streams answers from the backend `/ask/stream` endpoint for a real-time typing effect.
"""

import os
import uuid
import json
import requests
import streamlit as st

# ---- Streamlit Page Configuration ----
st.set_page_config(page_title="AskFAQ", page_icon="", layout="wide")

# -------------------
# Backend API Config
# -------------------
API_URL = os.getenv("FAQ_API_URL", "http://127.0.0.1:8000")  # Default backend URL
ASK_STREAM_URL = f"{API_URL.rstrip('/')}/ask/stream"  # Streaming endpoint

# -------------------
# Session State Setup
# -------------------
# Generate a unique session ID for each chat session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize message history with a greeting from the assistant
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "I’m AskFAQ—your personal assistant for HR related queries. "
                "How can I help you today?"
            )
        }
    ]

# Store the currently selected suggestion question (if any)
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -------------------
# Helper Functions
# -------------------
def clean_answer(text: str) -> str:
    """
    Clean the raw answer text by removing extra prefixes like 'Answer:', 'A:', or Q/A format.

    Args:
        text (str): Raw answer text.

    Returns:
        str: Cleaned answer text.
    """
    s = (text or "").strip()
    lowers = s.lower()

    # Strip 'Answer:' prefix if present
    if lowers.startswith("answer:"):
        s = s[7:].strip()

    # Strip 'A:' prefix if present
    if s.startswith("A:"):
        s = s[2:].strip()

    # If text starts with 'Q:' followed by 'A:', keep only the answer part
    if s.startswith("Q:"):
        parts = s.split("A:", 1)
        if len(parts) == 2:
            s = parts[1].strip()

    return s


def stream_answer(question: str) -> str:
    """
    Call the backend streaming endpoint and display the answer tokens in real-time.

    Args:
        question (str): The user question to send to the backend.

    Returns:
        str: Final cleaned answer text.
    """
    full = ""
    with st.chat_message("assistant"):
        ph = st.empty()
        try:
            # Stream tokens from backend
            with requests.post(
                ASK_STREAM_URL,
                json={
                    "question": question,
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
                        # Handle token-by-token updates
                        if "token" in data:
                            full += data["token"] + " "
                            ph.markdown(full)
                        # Handle final response
                        elif "final" in data:
                            full = data["final"]
                            ph.markdown(full)
                else:
                    full = f"Server error {r.status_code}: {r.text}"
                    ph.error(full)
        except Exception as e:
            full = f"Request failed: {e}"
            ph.error(full)

    return clean_answer(full)


# -------------------
# Sidebar
# -------------------
st.sidebar.title("AskFAQ Assistant")
st.sidebar.write(
    "Ask questions about policies, leave, payroll, and more. "
    "This assistant uses your internal FAQ knowledge base."
)

# Clear chat button resets session state
if st.sidebar.button("Clear chat"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "I’m AskFAQ—your personal assistant for HR related queries. "
                "How can I help you today?"
            )
        }
    ]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pending_question = None


# -------------------
# Header
# -------------------
st.markdown(
    """
    <div style='background:#0B69D3; padding:12px; border-radius:8px;'>
        <h2 style='color:white; text-align:center; margin:0;'>AskFAQ</h2>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------
# Fixed Quick Suggestions
# -------------------
with st.container():
    st.markdown(
        "<h3 style='font-weight:600; color:#333;'>💡 Need Help? Try Asking:</h3>",
        unsafe_allow_html=True
    )

    # Predefined FAQs as clickable buttons
    suggestions = [
        "What documents are needed for reimbursement?",
        "How do I download my payslip?",
        "What is the notice period for resignation?",
        "Are there any educational allowances?"
    ]

    # Display suggestion buttons with equal spacing
    cols = st.columns(len(suggestions), gap="large")
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_question = s  # Capture which suggestion was clicked

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------
# Gather Input (Suggestion or Chat)
# -------------------
query = None

# If a suggestion was clicked, set it as the current query
if st.session_state.pending_question:
    query = st.session_state.pending_question
    st.session_state.pending_question = None

# If the user typed something manually, override the query
user_text = st.chat_input("Type your question…")
if user_text:
    query = user_text

# Append new user message to session history immediately
if query:
    st.session_state.messages.append({"role": "user", "content": query})

# -------------------
# Render Chat History
# -------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------
# Stream Answer for New Questions
# -------------------
if query:
    answer = stream_answer(query)
    st.session_state.messages.append({"role": "assistant", "content": answer})
