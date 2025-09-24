# app/ui.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="AI-Powered FAQ Assistant", page_icon="💬")

API_URL = os.getenv("FAQ_API_URL", "http://127.0.0.1:8000")
ASK_URL = f"{API_URL.rstrip('/')}/ask"

# ---- Sidebar controls (keep it simple) ----
with st.sidebar:
    st.header("AI-Powered FAQ Assistant")
    st.caption("Ask questions about policies, leave, payroll, and more. The assistant uses your FAQ KB.")
    clear = st.button("Clear chat")
    # st.divider()
    # st.caption("Backend:")
    # st.code(ASK_URL, language="text")

if "messages" not in st.session_state or clear:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

st.title("MyBuddy")
# st.caption("Simple HR FAQ chatbot (Streamlit demo)")

# ---- Render history ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---- Ask backend helper ----
def ask_backend(question: str) -> tuple[str, list[dict]]:
    """
    Calls FastAPI /ask endpoint and returns (answer, contexts).
    Expects JSON: {"answer": "...", "contexts": [{"question": "...", "answer": "..."}]}
    """
    try:
        r = requests.post(ASK_URL, json={"question": question}, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("answer", ""), data.get("contexts", [])
    except Exception as e:
        return f"Request failed: {e}", []

# ---- Chat input -> backend ----
prompt = st.chat_input("Type your question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        answer, contexts = ask_backend(prompt)
        status.update(label="Done", state="complete", expanded=False)
        st.write(answer)

        # Optional: show retrieved snippets like the blog demo does
        # if contexts:
        #     with st.expander("Retrieved FAQs"):
        #         for i, c in enumerate(contexts, 1):
        #             st.markdown(f"**{i}. {c.get('question','')}**")
        #             st.write(c.get("answer", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
