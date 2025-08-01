from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import yaml
import argparse
from typing import TypedDict
from hashlib import sha256

# Make local modules importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.retrieval_generation.prompts import get_klugekopf_system_prompt
from chatbot.retrieval_generation.retriever import retrieve_context

# ============================
# Load env + config
# ============================

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env.")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

base_url = config["llm"]["base_url"]
MODEL_NAME = config["llm"]["model_name"]

client = OpenAI(base_url=base_url, api_key=api_key)

# ============================
# Simple cache (global store)
# ============================

CACHE = {}


def get_cache_key(session_id: str, agent_name: str, input_text: str) -> str:
    hash_input = f"{session_id}:{agent_name}:{input_text}".encode()
    return sha256(hash_input).hexdigest()


# ============================
# State Schema
# ============================


class KlugekopfState(TypedDict):
    session_id: str  # ✅ New: session identifier
    query: str
    rewritten_query: str
    plan: str
    chunks: list[str]
    metadatas: list[dict]
    summary: str
    tool_result: str
    answer: str


# ============================
# Rewrite Agent
# ============================


def rewrite_agent_node(state: KlugekopfState) -> KlugekopfState:
    import re

    def is_greeting(text: str) -> bool:
        return bool(
            re.match(
                r"^(hi|hello|hey|howdy|greetings|good (morning|afternoon|evening))[\s!\.]*$",
                text.strip().lower(),
            )
        )

    query = state["query"]
    session_id = state.get("session_id", "global")
    cache_key = get_cache_key(session_id, "rewrite_agent", query)

    # Skip rewriting if it's a greeting
    if is_greeting(query):
        rewritten_query = query.strip()
        CACHE[cache_key] = rewritten_query
        return {**state, "rewritten_query": rewritten_query}

    # Friendly, personality-aware system prompt
    SYSTEM_PROMPT = """You are a friendly and helpful assistant that rewrites user messages into clear, concise queries for downstream tools.

Instructions:
- If the user input is a greeting (e.g., "Hi", "Hello", "Hey", "Howdy", "Good morning"), return it as-is without rewriting.
- Otherwise, rewrite the question to be clear, professional, and focused, while preserving the user’s intent."""

    if cache_key in CACHE:
        rewritten_query = CACHE[cache_key]
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        rewritten_query = response.choices[0].message.content.strip()
        CACHE[cache_key] = rewritten_query

    return {**state, "rewritten_query": rewritten_query}


# ============================
# Planner Agent
# ============================


def planner_agent_node(state: KlugekopfState) -> KlugekopfState:
    rewritten_query = state["rewritten_query"]
    session_id = state.get("session_id", "global")
    cache_key = get_cache_key(session_id, "planner_agent", rewritten_query)

    SYSTEM_PROMPT = """You are a helpful and structured assistant.

Your job is to break down user questions into a plan of steps the assistant should follow. 
If the message is a greeting (e.g., 'Hi', 'Hello'), no planning is needed — just return a single step like 'reply with a friendly greeting'.

Otherwise, return a 1–3 step plan to answer the query."""

    if cache_key in CACHE:
        plan = CACHE[cache_key]
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Break down this task: {rewritten_query}"},
            ],
        )
        plan = response.choices[0].message.content.strip()
        CACHE[cache_key] = plan

    return {**state, "plan": plan}


# ============================
# Retrieval Agent
# ============================


def retrieval_agent_node(state: KlugekopfState) -> KlugekopfState:
    rewritten_query = state["rewritten_query"]
    session_id = state.get("session_id", "global")
    cache_key = get_cache_key(session_id, "retrieval_agent", rewritten_query)

    if cache_key in CACHE:
        chunks, metadatas = CACHE[cache_key]
    else:
        chunks, metadatas = retrieve_context(rewritten_query)
        CACHE[cache_key] = (chunks, metadatas)

    return {**state, "chunks": chunks, "metadatas": metadatas}


# ============================
# Summarizer Agent
# ============================


def summarizer_agent_node(state: KlugekopfState) -> KlugekopfState:
    chunks = state["chunks"]
    text = "\n\n".join(chunks)
    session_id = state.get("session_id", "global")
    cache_key = get_cache_key(session_id, "summarizer_agent", text)

    if cache_key in CACHE:
        summary = CACHE[cache_key]
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a summarizer. Condense this context clearly.",
                },
                {"role": "user", "content": f"Summarize this context:\n{text}"},
            ],
        )
        summary = response.choices[0].message.content.strip()
        CACHE[cache_key] = summary

    return {**state, "summary": summary}


# ============================
# Tool Agent
# ============================


def tool_agent_node(state: KlugekopfState) -> KlugekopfState:
    tool_result = "Pretend I did a Google Search or DB call here."
    return {**state, "tool_result": tool_result}


# ============================
# Final Answer Agent
# ============================


def klugekopf_agent_node(state: KlugekopfState) -> KlugekopfState:
    import re

    def is_greeting(text: str) -> bool:
        return bool(
            re.match(
                r"^(hi|hello|hey|howdy|greetings|good (morning|afternoon|evening))[\s!\.]*$",
                text.strip().lower(),
            )
        )

    rewritten_query = state["rewritten_query"]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful and friendly AI assistant. 
If the user greets you (e.g., says 'Hi', 'Hello', etc.), respond with a warm greeting and ask how you can help. 
If the user asks who you are, introduce yourself briefly as the Klugekopf assistant. 
Otherwise, answer the user’s question clearly, using any context provided. 
Use Markdown formatting if it improves clarity.""",
            },
            {"role": "user", "content": rewritten_query},
        ],
    )

    llm_output = response.choices[0].message.content.strip()

    return {"answer": llm_output}


# ============================
# Build Graph
# ============================

graph = StateGraph(state_schema=KlugekopfState)

graph.add_node("rewrite_agent", rewrite_agent_node)
graph.add_node("planner_agent", planner_agent_node)
graph.add_node("retrieval_agent", retrieval_agent_node)
graph.add_node("summarizer_agent", summarizer_agent_node)
graph.add_node("tool_agent", tool_agent_node)
graph.add_node("klugekopf_agent", klugekopf_agent_node)

graph.set_entry_point("rewrite_agent")
graph.add_edge("rewrite_agent", "planner_agent")
graph.add_edge("planner_agent", "retrieval_agent")
graph.add_edge("retrieval_agent", "summarizer_agent")
graph.add_edge("summarizer_agent", "tool_agent")
graph.add_edge("tool_agent", "klugekopf_agent")
graph.set_finish_point("klugekopf_agent")

klugekopf_multi_agent_app = graph.compile()

# ============================
# Local test
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    session_id = input("Session ID: ")
    user_query = input("Ask Klugekopf-Bot: ")
    result = klugekopf_multi_agent_app.invoke(
        {"session_id": session_id, "query": user_query}
    )
    print("\nKlugekopf-Bot says:\n")
    print(result["answer"])
