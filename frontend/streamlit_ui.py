import streamlit as st
from chatbot.retrieval_generation.graph import klugekopf_multi_agent_app
import os
import json
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2
import bcrypt

# --- Load secrets ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
MODEL_NAME = "llama3-8b-8192"

# --- DB connection ---
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
)
cursor = conn.cursor()

# --- Page config ---
st.set_page_config(page_title="Klugekopf Chatbot", layout="wide")
st.title("💬 Klugekopf - Strategic Assistant")

# --- Auth ---
if "user_id" not in st.session_state and "guest_mode" not in st.session_state:
    st.subheader("🔑 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        cursor.execute(
            "SELECT id, password_hash FROM users WHERE username = %s", (username,)
        )
        user = cursor.fetchone()

        if user:
            if bcrypt.checkpw(password.encode(), user[1].encode()):
                st.session_state["user_id"] = user[0]
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid password. Please try again.")
        else:
            st.warning("🚫 User not found. Please sign up below.")

    st.markdown("---")
    st.subheader("🆕 Sign Up")

    new_username = st.text_input("New Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("New Password", type="password")

    if st.button("Sign Up"):
        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (new_username, new_email, hashed),
            )
            conn.commit()
            st.success("✅ Account created! Please log in.")
            st.rerun()
        except psycopg2.IntegrityError as e:
            conn.rollback()  # Roll back the failed transaction!
            msg = str(e)
            if "users_username_key" in msg:
                st.error("❌ Username already exists. Please choose a different one.")
            elif "users_email_key" in msg:
                st.error("❌ Email already exists. Please use a different email.")
            else:
                st.error("❌ Something went wrong. Please try again.")
        except Exception as e:
            st.error("❌ Unexpected error. Please try again later.")

    st.markdown("---")
    if st.button("🔓 Continue as Guest"):
        st.session_state["guest_mode"] = True
        st.success("✅ Guest session started.")
        st.rerun()

    st.stop()

# --- Figure out mode ---
is_guest = "guest_mode" in st.session_state
user_id = st.session_state.get("user_id")

# --- Sidebar: logout or end guest session ---
if is_guest:
    if st.sidebar.button("🚪 End Guest Session"):
        del st.session_state["guest_mode"]
        st.success("Guest session ended.")
        st.rerun()
else:
    if st.sidebar.button("🚪 Logout"):
        del st.session_state["user_id"]
        st.success("Logged out.")
        st.rerun()

# --- Init messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("🗂️ Manage Chat Sessions")

    if st.button("🆕 New Chat"):
        st.session_state.messages = []

    if not is_guest:
        st.markdown("---")
        st.subheader("📂 Previous Chats:")

        cursor.execute(
            "SELECT id, title FROM chat_sessions WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        sessions = cursor.fetchall()

        for s_id, s_title in sessions:
            if st.button(f"📄 {s_title}", key=f"load_{s_id}"):
                cursor.execute(
                    "SELECT messages FROM chat_sessions WHERE id = %s", (s_id,)
                )
                data = cursor.fetchone()
                st.session_state.messages = data[0]
                st.rerun()

# --- Chat display ---
st.markdown("---")

for msg in st.session_state.messages:
    bg_color = "#2C2F33" if msg["role"] == "user" else "#40444B"
    st.markdown(
        f"<div style='background-color: {bg_color}; color: white; padding: 12px; border-radius: 10px; margin: 8px 0;'>{msg['content']}</div>",
        unsafe_allow_html=True,
    )

# --- Input ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your message:", placeholder="Type your message here...", height=80
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    is_first = len(st.session_state.messages) == 1

    if is_first:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Summarize this question in 3-5 words."},
                {"role": "user", "content": user_input.strip()},
            ],
        )
        short_title = response.choices[0].message.content.strip().lower()
        short_title = re.sub(r"\W+", "_", short_title)[:40]
        chat_title = short_title.replace("_", " ").title()
    else:
        if not is_guest:
            cursor.execute(
                "SELECT title FROM chat_sessions WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            )
            result = cursor.fetchone()
            chat_title = result[0] if result else "Untitled"
        else:
            chat_title = "Guest Session"

    with st.spinner("Thinking..."):
        result = klugekopf_multi_agent_app.invoke({"query": user_input})
        answer = result["answer"]

    st.session_state.messages.append({"role": "bot", "content": answer})

    if not is_guest:
        if is_first:
            cursor.execute(
                "INSERT INTO chat_sessions (user_id, title, messages) VALUES (%s, %s, %s)",
                (user_id, chat_title, json.dumps(st.session_state.messages)),
            )
        else:
            cursor.execute(
                "UPDATE chat_sessions SET messages = %s WHERE user_id = %s AND title = %s",
                (json.dumps(st.session_state.messages), user_id, chat_title),
            )
        conn.commit()

    st.rerun()
