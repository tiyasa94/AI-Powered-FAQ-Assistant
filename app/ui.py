# app/ui.py
import os
import base64
import requests
import streamlit as st
from streamlit_float import float_init

st.set_page_config(page_title="AI-Powered FAQ Assistant", page_icon="💬", layout="wide")

API_URL = os.getenv("FAQ_API_URL", "http://127.0.0.1:8000")
ASK_URL = f"{API_URL.rstrip('/')}/ask"

# --- background image helper ---
def set_page_background(image_path: str, mime: str = "jpg"):
    if not os.path.exists(image_path):
        st.warning(f"Background image not found: {image_path}")
        return
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
          /* Full-page background */
          .stApp {{
            background: url("data:image/{mime};base64,{b64}") no-repeat center center fixed;
            background-size: cover;
          }}
          /* Make foreground containers transparent so the bg shows through */
          .stMain, .block-container {{
            background: transparent !important;
          }}
          header, footer {{
            background: transparent !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_page_background("app/assets/hero_bg.png", mime="png")

# --- init floating support ---
float_init()

# === Floating Chat Box (bottom-right) ===
chat_box = st.container()

# 1) Float FIRST to minimize jump/flicker
chat_box.float(
    css="""
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 380px;
    max-height: 75vh;
    padding: 16px;
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(6px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.18);
    overflow: auto;
    """
)

# 2) Then render content inside the floated container
with chat_box:
    st.subheader("Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "What's on your mind?"}
        ]

    # render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # status placeholder shown while we wait for the backend
    status = st.empty()

    # input -> backend /ask
    user_text = st.chat_input("Type your question…")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # 3) Show in-chat status while generating
        status.markdown("**Getting answer…**")

        answer, contexts = "", []
        try:
            r = requests.post(ASK_URL, json={"question": user_text})
            if r.status_code == 200:
                data = r.json()
                answer = data.get("answer", "")
                contexts = data.get("contexts", [])
            else:
                answer = f"Server error {r.status_code}: {r.text}"
        except Exception as e:
            answer = f"Request failed: {e}"

        # clear the status line once we have a result
        status.empty()

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

        # # optional: show retrieved snippets
        # if contexts:
        #     with st.expander("Retrieved FAQs"):
        #         for i, c in enumerate(contexts, 1):
        #             st.markdown(f"**{i}. {c.get('question','')}**")
        #             st.write(c.get("answer", ""))
