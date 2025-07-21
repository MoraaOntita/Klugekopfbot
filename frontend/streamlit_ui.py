import streamlit as st
import sys
import os
import json
import re
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# --- Local modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.retrieval_generation.graph import klugekopf_multi_agent_app

# --- Load secrets ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
api_key = os.getenv("GROQ_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Supabase URL or ANON Key is missing!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
MODEL_NAME = "llama3-8b-8192"

st.set_page_config(page_title="Klugekopf Chatbot", layout="wide")
st.title("💬 Klugekopf - Strategic Assistant")


# --- Auth helpers ---
def handle_error(msg: str) -> str:
    return f"❌ {msg}"


# --- Auth flow ---
if "user" not in st.session_state and "guest_mode" not in st.session_state:

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    mode = st.session_state["auth_mode"]

    if mode == "login":
        st.subheader("🔑 Login to your account")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input(
            "Password", type="password", key="login_password"
        )

        if st.button("Login"):
            res = supabase.auth.sign_in_with_password(
                {"email": login_email, "password": login_password}
            )

            if res.error:
                st.error(handle_error(res.error.message))
            else:
                st.session_state["user"] = res.user
                st.session_state["access_token"] = res.session.access_token

                # Optionally fetch username from `profiles`
                profile = (
                    supabase.from_("profiles")
                    .select("*")
                    .eq("user_id", res.user.id)
                    .execute()
                )
                if profile.data:
                    st.session_state["username"] = profile.data[0]["username"]
                else:
                    st.session_state["username"] = res.user.email

                st.success(f"✅ Welcome {st.session_state['username']}! Redirecting...")
                st.rerun()

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

        new_email = st.text_input("Email", key="signup_email").strip()
        new_password = st.text_input(
            "Password", type="password", key="signup_password"
        ).strip()
        new_username = st.text_input("Username", key="signup_username").strip().lower()

        if st.button("Sign Up"):
            if not new_email or not new_password or not new_username:
                st.warning("⚠️ Please fill in all fields.")
            elif " " in new_username or len(new_username) < 3:
                st.warning(
                    "⚠️ Username must be at least 3 characters and contain no spaces."
                )
            else:
                res = supabase.auth.sign_up(
                    {"email": new_email, "password": new_password}
                )

                if res.error:
                    st.error(handle_error(res.error.message))
                else:
                    # Create profile
                    supabase.from_("profiles").insert(
                        {"user_id": res.user.id, "username": new_username}
                    ).execute()

                    st.success(f"✅ Account created for {new_username}! Please log in.")
                    st.session_state["auth_mode"] = "login"
                    st.rerun()

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
user = st.session_state.get("user")
username = st.session_state.get("username", "Guest")

# --- Sidebar ---
with st.sidebar:
    if user:
        st.header(f"👋 Welcome, {username}!")

    st.header("🗂️ Manage Chat Sessions")

    if is_guest:
        st.info("🔍 Guest Mode is active. End it to return to your account.")
        if st.button("🚪 End Guest Session"):
            del st.session_state["guest_mode"]
            st.success("Guest session ended.")
            st.rerun()
    else:
        if user:
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

    if not is_guest and user:
        st.markdown("---")
        st.subheader("📂 Previous Chats:")
        sessions = (
            supabase.from_("chat_sessions")
            .select("id, title")
            .eq("user_id", user.id)
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
        f"<div style='background-color: {bg_color}; color: white; padding: 12px; "
        f"border-radius: 10px; margin: 8px 0;'>{msg['content']}</div>",
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
                .eq("user_id", user.id)
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
                    "user_id": user.id,
                    "title": chat_title,
                    "messages": json.dumps(st.session_state.messages),
                }
            ).execute()
        else:
            supabase.from_("chat_sessions").update(
                {
                    "messages": json.dumps(st.session_state.messages),
                }
            ).eq("user_id", user.id).eq("title", chat_title).execute()

    st.rerun()
