import streamlit as st
from chatbot.retrieval_generation.graph import klugekopf_multi_agent_app
import os
import json
from datetime import datetime
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load keys
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
MODEL_NAME = "llama3-8b-8192"

# --- Page ---
st.set_page_config(page_title="Klugekopf Chatbot", layout="wide")
st.title("💬 Klugekopf - Strategic Assistant")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "session_title" not in st.session_state:
    st.session_state.session_title = None
if "show_delete" not in st.session_state:
    st.session_state.show_delete = {}

# --- Sidebar ---
with st.sidebar:
    st.header("Lead Information")
    user_name = st.text_input("Your Name")
    user_email = st.text_input("Your Email")

    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.session_title = None
        st.session_state.show_delete = {}

    st.markdown("---")
    st.subheader("📂 Previous Chats:")

    os.makedirs("data/chat_sessions", exist_ok=True)
    chat_files = sorted(os.listdir("data/chat_sessions"))

    for filename in chat_files:
        with open(f"data/chat_sessions/{filename}", "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            display_title = data.get("title", filename.split(".json")[0])
            messages = data.get("messages", [])
        else:
            display_title = filename.split(".json")[0]
            messages = data

        file_id = filename

        cols = st.columns([8, 1])
        with cols[0]:
            if st.button(display_title, key=f"load_{file_id}"):
                st.session_state.messages = messages
                st.session_state.session_id = filename
                st.session_state.session_title = display_title
        with cols[1]:
            if st.button("⋮", key=f"menu_{file_id}"):
                st.session_state.show_delete[file_id] = not st.session_state.show_delete.get(file_id, False)

        if st.session_state.show_delete.get(file_id, False):
            if st.button(f"❌ Delete `{display_title}`", key=f"delete_{file_id}"):
                os.remove(f"data/chat_sessions/{filename}")
                st.success(f"Deleted `{display_title}`")
                if filename == st.session_state.session_id:
                    st.session_state.messages = []
                    st.session_state.session_id = None
                    st.session_state.session_title = None
                st.rerun()


# --- Chat display ---
st.markdown("---")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='background-color: #2C2F33; color: white; padding: 12px; border-radius: 10px; margin: 8px 0;'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color: #40444B; color: white; padding: 12px; border-radius: 10px; margin: 8px 0;'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

# --- Input ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your message:",
        placeholder="Type your message here and press Enter or click Send",
        height=80
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    # If first message, auto-generate short title
    if not st.session_state.session_id:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Summarize this question into a very short 3-5 word title."},
                {"role": "user", "content": user_input.strip()}
            ]
        )
        short_title = response.choices[0].message.content.strip().lower()
        short_title = re.sub(r'\W+', '_', short_title)[:40]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.session_state.session_title = short_title.replace("_", " ").title()
        st.session_state.session_id = f"{short_title}_{timestamp}.json"

    with st.spinner("Thinking..."):
        result = klugekopf_multi_agent_app.invoke({"query": user_input})
        answer = result["answer"]

    st.session_state.messages.append({"role": "bot", "content": answer})

    # Save nicely
    os.makedirs("data/chat_sessions", exist_ok=True)
    session_data = {
        "title": st.session_state.session_title,
        "messages": st.session_state.messages
    }
    with open(f"data/chat_sessions/{st.session_state.session_id}", "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    if user_name and user_email:
        with open("data/leads.csv", "a", encoding="utf-8") as f:
            f.write(f"{user_name},{user_email}\n")

    st.rerun()
