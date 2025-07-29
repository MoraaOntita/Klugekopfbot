import streamlit as st
import sys
import os
import json
import re
from frontend.utils import is_strong_password
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# --- Local module import ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.retrieval_generation.graph import klugekopf_multi_agent_app

# --- Load secrets ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
api_key = os.getenv("GROQ_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

MODEL_NAME = "llama3-8b-8192"
EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"

st.set_page_config(page_title="Klugekopf Chatbot", layout="wide")

st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        color: #FFFFFF;
    }

    /* Make headers bolder */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-weight: 600;
    }

    /* Buttons in sidebar */
    [data-testid="stSidebar"] button {
        background-color: #262730;
        color: #FFFFFF;
        border: 1px solid #3c4048;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin-top: 0.3rem;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #3c4048;
        color: #FFFFFF;
    }

    /* Info messages */
    [data-testid="stSidebar"] .stAlert {
        border-radius: 6px;
    }

    /* Columns spacing */
    [data-testid="stSidebar"] .block-container .stHorizontalBlock {
        gap: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.title("💬 Klugekopf - Strategic Assistant")

# --- Auth flow ---
if "user" not in st.session_state and "guest_mode" not in st.session_state:

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    mode = st.session_state["auth_mode"]

    if mode == "login":
        st.subheader("🔑 Login to your account")
        login_email = st.text_input("Email")
        login_password = st.text_input("Password", type="password")

        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password(
                    {"email": login_email, "password": login_password}
                )
                data = res.model_dump()
                user_data = data["user"]
                session_data = data["session"]

                st.session_state["user"] = user_data
                st.session_state["access_token"] = session_data["access_token"]

                # ✅ Attach the JWT once for this client
                supabase.postgrest.auth(st.session_state["access_token"])

                # Get profile info
                profile = (
                    supabase.table("profiles")
                    .select("*")
                    .eq("user_id", user_data["id"])
                    .execute()
                )

                if profile.data and profile.data[0]["username"]:
                    st.session_state["username"] = profile.data[0]["username"]
                else:
                    st.session_state["username"] = user_data["email"]

                st.success(f"✅ Welcome {st.session_state['username']}! Redirecting...")
                st.rerun()

            except Exception as e:
                error_message = str(e)
                if "Email not confirmed" in error_message:
                    st.warning(
                        "📧 Please confirm your email address. Check your inbox for a confirmation link."
                    )
                else:
                    st.error(f"❌ {error_message}")

        st.markdown("---")
        if st.button("Continue as Guest"):
            st.session_state["guest_mode"] = True
            st.rerun()

        st.markdown("Don’t have an account?")
        if st.button("Sign Up"):
            st.session_state["auth_mode"] = "signup"
            st.rerun()

    elif mode == "signup":
        st.subheader("📝 Sign Up")

        new_email = st.text_input("Email").strip()
        new_password = st.text_input("Password", type="password").strip()
        new_username = st.text_input("Username").strip().lower()

        if st.button("Sign Up"):
            # Basic empty field check
            if not new_email or not new_password or not new_username:
                st.warning("⚠️ Please fill in all fields.")
            # Email validation
            elif not re.match(EMAIL_REGEX, new_email):
                st.warning("📧 Invalid email format.")
            # Password strength (min length 6)
            # Password strength validation
            elif not is_strong_password(new_password):
                st.warning(
                    "🔐 Password must be at least 8 characters long and include:\n"
                    "- At least one uppercase letter\n"
                    "- One lowercase letter\n"
                    "- One digit\n"
                    "- One special character (@$!%*#?&)"
                )

            # Username validation: alphanumeric + underscores, no spaces
            elif not re.match(r"^[a-zA-Z0-9_]+$", new_username):
                st.warning(
                    "👤 Username can only contain letters, numbers, and underscores."
                )
            else:
                try:
                    # Create auth user
                    supabase.auth.sign_up(
                        {"email": new_email, "password": new_password}
                    )

                    # ✅ Immediately sign in to get session/JWT
                    login = supabase.auth.sign_in_with_password(
                        {"email": new_email, "password": new_password}
                    )
                    session_data = login.model_dump()["session"]
                    user_id = login.model_dump()["user"]["id"]

                    # ✅ Attach the JWT for DB insert
                    supabase.postgrest.auth(session_data["access_token"])

                    # ✅ Insert profile with same user_id
                    supabase.table("profiles").insert(
                        {"username": new_username, "user_id": user_id}
                    ).execute()

                    st.success(
                        "✅ Account created! Check your email to confirm your account."
                    )
                    st.stop()

                except Exception as e:
                    st.error(f"❌ {e}")

    st.stop()

# --- Determine mode ---
is_guest = "guest_mode" in st.session_state
user = st.session_state.get("user")

# ✅ Ensure JWT is attached if we have it
if user and "access_token" in st.session_state:
    supabase.postgrest.auth(st.session_state["access_token"])

# --- Sidebar ---
with st.sidebar:
    st.title("💼 Klugekopf Chat")
    st.divider()

    st.subheader("✨ Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat"):
            st.session_state.messages = []
            st.session_state.current_session_id = None
    with col2:
        if not is_guest and user:
            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.rerun()

    if not is_guest and user:
        if st.button("🕶️ Switch to Guest Mode"):
            st.session_state["guest_mode"] = True
            st.rerun()
    elif is_guest:
        if st.button("🚪 End Guest Session"):
            del st.session_state["guest_mode"]
            st.rerun()

    st.divider()
    st.subheader("📂 Your Chats")

    if not is_guest and user:
        try:
            sessions = (
                supabase.table("chat_sessions")
                .select("id, title")
                .eq("user_id", user["id"])
                .order("created_at", desc=True)
                .execute()
            )

            if not sessions.data:
                st.info("No saved chats yet.")
            else:
                for s in sessions.data:
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        if st.button(f"📄 {s['title']}", key=f"load_{s['id']}"):
                            try:
                                chat = (
                                    supabase.table("chat_sessions")
                                    .select("messages")
                                    .eq("id", s["id"])
                                    .execute()
                                )
                                if chat.data:
                                    st.session_state.messages = json.loads(
                                        chat.data[0]["messages"]
                                    )
                                    st.session_state.current_session_id = s["id"]
                                    st.rerun()
                                else:
                                    st.warning("⚠️ No messages found for this session.")
                            except Exception as e:
                                st.error(f"❌ Failed to load chat messages: {str(e)}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{s['id']}"):
                            try:
                                supabase.table("chat_sessions").delete().eq(
                                    "id", s["id"]
                                ).execute()
                                st.success(f"🗑️ Deleted: {s['title']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to delete session: {str(e)}")

        except Exception as e:
            st.error(f"❌ Failed to load chat sessions: {str(e)}")

# --- Init messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat display ---
st.markdown("---")
for msg in st.session_state.messages:
    bg = "#2C2F33" if msg["role"] == "user" else "#40444B"
    st.markdown(
        f"<div style='background: {bg}; color: white; padding:10px; border-radius:8px;'>{msg['content']}</div>",
        unsafe_allow_html=True,
    )

# --- Input ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message:", height=80)
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    is_first = len(st.session_state.messages) == 1

    if is_first:
        summary = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Summarize this in 3-5 words."},
                {"role": "user", "content": user_input.strip()},
            ],
        )
        title = summary.choices[0].message.content.strip().title()
    else:
        title = "Chat Session"

    with st.spinner("Thinking..."):
        result = klugekopf_multi_agent_app.invoke(
            {
                "session_id": st.session_state.get("current_session_id"),
                "query": user_input,
            }
        )
    st.session_state.messages.append({"role": "bot", "content": result["answer"]})

    if not is_guest and user:
        if is_first:
            new_session = (
                supabase.table("chat_sessions")
                .insert(
                    {
                        "user_id": user["id"],
                        "title": title,
                        "messages": json.dumps(st.session_state.messages),
                    }
                )
                .execute()
            )
            st.session_state.current_session_id = new_session.data[0]["id"]
        else:
            supabase.table("chat_sessions").update(
                {"messages": json.dumps(st.session_state.messages)}
            ).eq("id", st.session_state["current_session_id"]).execute()

    st.rerun()
