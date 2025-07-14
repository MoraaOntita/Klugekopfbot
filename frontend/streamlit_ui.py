import streamlit as st
import sys
import os
import json
import re
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# 📌 Local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.retrieval_generation.graph import klugekopf_multi_agent_app

# --- Load secrets ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Supabase URL or Service Role Key is missing!")

# --- Init clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
MODEL_NAME = "llama3-8b-8192"

# --- Page config ---
st.set_page_config(page_title="Klugekopf Chatbot", layout="wide")
st.title("💬 Klugekopf - Strategic Assistant")

# --- Check session ---
session = supabase.auth.get_session().session
if session:
    st.session_state["user_id"] = session.user.id

# --- Auth flow ---
if "user_id" not in st.session_state and "guest_mode" not in st.session_state:
    auth_mode = st.radio("Choose:", ["🔑 Login", "🆕 Sign Up"], horizontal=True)

    if auth_mode == "🔑 Login":
        st.subheader("Login to your account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            try:
                resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if resp.session:
                    st.session_state["user_id"] = resp.session.user.id
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    else:
        st.subheader("Create a new account")
        new_email = st.text_input("Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            if not new_email or not new_password:
                st.warning("⚠️ Please fill in all fields to sign up.")
            else:
                try:
                    resp = supabase.auth.sign_up({"email": new_email, "password": new_password})
                    if resp.user:
                        st.success("✅ Account created! Please check your email to confirm and then log in.")
                        st.rerun()
                    else:
                        st.error("❌ Could not create account. Maybe email already exists?")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")
    if st.button("🔓 Continue as Guest"):
        st.session_state["guest_mode"] = True
        st.success("✅ Guest session started.")
        st.rerun()

    st.stop()

# --- Determine mode ---
is_guest = "guest_mode" in st.session_state
user_id = st.session_state.get("user_id")

# --- Sidebar logout/end guest ---
if is_guest:
    if st.sidebar.button("🚪 End Guest Session"):
        del st.session_state["guest_mode"]
        st.success("Guest session ended.")
        st.rerun()
else:
    if st.sidebar.button("🚪 Logout"):
        supabase.auth.sign_out()
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

        sessions = (
            supabase.from_("chat_sessions")
            .select("id, title")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        for s in sessions.data:
            if st.button(f"📄 {s['title']}", key=f"load_{s['id']}"):
                chat = (
                    supabase.from_("chat_sessions")
                    .select("messages")
                    .eq("id", s["id"])
                    .execute()
                )
                st.session_state.messages = json.loads(chat.data[0]["messages"])
                st.rerun()

# --- Chat display ---
st.markdown("---")
for msg in st.session_state.messages:
    bg_color = "#2C2F33" if msg["role"] == "user" else "#40444B"
    st.markdown(
        f"<div style='background-color: {bg_color}; color: white; padding: 12px; border-radius: 10px; margin: 8px 0;'>{msg['content']}</div>",
        unsafe_allow_html=True,
    )

# --- Chat input ---
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
        chat_title = "Guest Session" if is_guest else "Untitled"

    with st.spinner("Thinking..."):
        result = klugekopf_multi_agent_app.invoke({"query": user_input})
        answer = result["answer"]

    st.session_state.messages.append({"role": "bot", "content": answer})

    if not is_guest:
        if is_first:
            supabase.from_("chat_sessions").insert(
                {
                    "user_id": user_id,
                    "title": chat_title,
                    "messages": json.dumps(st.session_state.messages),
                }
            ).execute()
        else:
            supabase.from_("chat_sessions").update(
                {
                    "messages": json.dumps(st.session_state.messages),
                }
            ).eq("user_id", user_id).eq("title", chat_title).execute()

    st.rerun()
