import streamlit as st
import sys
import os
import json
import re
from dotenv import load_dotenv
from supabase import create_client, Client
import bcrypt
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

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
MODEL_NAME = "llama3-8b-8192"

st.set_page_config(page_title="Klugekopf Chatbot", layout="wide")
st.title("💬 Klugekopf - Strategic Assistant")

# --- Auth flow ---
if "user_id" not in st.session_state and "guest_mode" not in st.session_state:

    # --- Auth mode toggle ---
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    mode = st.session_state["auth_mode"]

    if mode == "login":
        st.subheader("🔑 Login to your account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            resp = (
                supabase.from_("users").select("*").eq("username", username).execute()
            )
            user = resp.data[0] if resp.data else None

            if user:
                if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
                    st.session_state["user_id"] = user["id"]
                    st.session_state["username"] = user["username"]
                    st.success(f"✅ Welcome {user['username']}! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid password.")
            else:
                st.error("❌ Username not found.")

        st.markdown("---")
        if st.button("🔓 Continue as Guest"):
            st.session_state["guest_mode"] = True
            st.success("✅ Guest session started.")
            st.rerun()

        st.markdown("Don’t have an account? 👉")
        if st.button("Sign Up Here"):
            st.session_state["auth_mode"] = "signup"
            st.rerun()

    elif mode == "signup":
        st.subheader("📝 Create a new account")

        new_username = st.text_input("Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input(
            "New Password", type="password", key="signup_password"
        )

        if st.button("Sign Up"):
            if not new_username or not new_email or not new_password:
                st.warning("⚠️ Please fill in all fields to sign up.")
            elif " " in new_username or len(new_username) < 3:
                st.warning(
                    "⚠️ Username must be at least 3 characters and contain no spaces."
                )
            else:
                hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                try:
                    resp = (
                        supabase.from_("users")
                        .insert(
                            {
                                "username": new_username,
                                "email": new_email,
                                "password_hash": hashed,
                            }
                        )
                        .execute()
                    )

                    if resp.error:
                        msg = str(resp.error)
                        if "duplicate key" in msg.lower():
                            st.error(
                                "❌ Username or Email already exists. Please choose another."
                            )
                        else:
                            st.error(f"❌ Unexpected error: {msg}")
                    else:
                        st.success("✅ Account created! Please log in.")
                        st.session_state["auth_mode"] = "login"
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Sign up error: {e}")

        st.markdown("Already have an account? 👉")
        if st.button("Back to Login"):
            st.session_state["auth_mode"] = "login"
            st.rerun()

        st.markdown("---")
        if st.button("🔓 Continue as Guest"):
            st.session_state["guest_mode"] = True
            st.success("Guest session started.")
            st.rerun()

    st.info(
        "💡 Tip: You can switch to Guest Mode anytime. End it to return to your account."
    )

    st.stop()

# --- Determine mode ---
is_guest = "guest_mode" in st.session_state
user_id = st.session_state.get("user_id")

# --- Sidebar ---
with st.sidebar:
    if user_id and "username" in st.session_state:
        st.header(f"👋 Welcome, {st.session_state['username']}!")

    st.header("🗂️ Manage Chat Sessions")

    if is_guest:
        st.info("🔍 Guest Mode is active. End it to return to your account.")
        if st.button("🚪 End Guest Session"):
            del st.session_state["guest_mode"]
            st.success("Guest session ended.")
            st.rerun()
    else:
        if user_id:
            if st.button("👤 Switch to Guest Mode"):
                st.session_state["guest_mode"] = True
                st.success("✅ Now in Guest Mode. Your account session is paused.")
                st.rerun()
        if st.button("Logout"):
            st.session_state.clear()
            st.success("Logged out.")
            st.rerun()

    if st.button("🆕 New Chat"):
        st.session_state.messages = []

    if not is_guest and user_id:
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

# --- Init messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        if not is_guest:
            last = (
                supabase.from_("chat_sessions")
                .select("title")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            chat_title = last.data[0]["title"] if last.data else "Untitled"
        else:
            chat_title = "Guest Session"

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
