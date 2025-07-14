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
# Simple cache
# ============================

CACHE = {}


def get_cache_key(agent_name: str, input_text: str) -> str:
    hash_input = f"{agent_name}:{input_text}".encode()
    return sha256(hash_input).hexdigest()


# ============================
# State Schema
# ============================


class KlugekopfState(TypedDict):
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
    query = state["query"]
    cache_key = get_cache_key("rewrite_agent", query)
    if cache_key in CACHE:
        rewritten_query = CACHE[cache_key]
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who rewrites user queries for clarity.",
                },
                {"role": "user", "content": f"Rewrite this user query: {query}"},
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
    cache_key = get_cache_key("planner_agent", rewritten_query)
    if cache_key in CACHE:
        plan = CACHE[cache_key]
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a planner. Break the query into clear tasks.",
                },
                {"role": "user", "content": f"Break down this task: {rewritten_query}"},
            ],
        )
        plan = response.choices[0].message.content.strip()
        CACHE[cache_key] = plan

    return {**state, "plan": plan}


# ============================
# Retrieval Agent (Pinecone!)
# ============================


def retrieval_agent_node(state: KlugekopfState) -> KlugekopfState:
    rewritten_query = state["rewritten_query"]
    # You can also cache retrieval results if they don't change often
    cache_key = get_cache_key("retrieval_agent", rewritten_query)
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
    cache_key = get_cache_key("summarizer_agent", text)
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
# Tool Agent (stub)
# ============================


def tool_agent_node(state: KlugekopfState) -> KlugekopfState:
    tool_result = "Pretend I did a Google Search or DB call here."
    return {**state, "tool_result": tool_result}


# ============================
# Final Answer Agent
# ============================


def klugekopf_agent_node(state: KlugekopfState) -> KlugekopfState:
    rewritten_query = state["rewritten_query"]
    summary = state["summary"]
    plan = state["plan"]
    tool_result = state["tool_result"]

    final_prompt = (
        f"Plan:\n{plan}\n\n"
        f"Summary of context:\n{summary}\n\n"
        f"Tool results:\n{tool_result}\n\n"
        f"User question:\n{rewritten_query}\n\n"
        f"Provide a final answer:"
    )

    system_message = get_klugekopf_system_prompt()
    cache_key = get_cache_key("klugekopf_agent", final_prompt)

    if cache_key in CACHE:
        answer = CACHE[cache_key]
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": final_prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        CACHE[cache_key] = answer

    return {**state, "answer": answer}


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

    user_query = input("Ask Klugekopf-Bot: ")
    result = klugekopf_multi_agent_app.invoke({"query": user_query})
    print("\nKlugekopf-Bot says:\n")
    print(result["answer"])
