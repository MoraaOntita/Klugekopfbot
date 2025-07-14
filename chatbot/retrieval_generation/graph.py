from langgraph.graph import StateGraph, END
from chatbot.retrieval_generation.retriever import retrieve_context
from openai import OpenAI
import os
import yaml
import argparse
from dotenv import load_dotenv
from typing import TypedDict

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.retrieval_generation.prompts import get_klugekopf_system_prompt

# ============================
# Load config + env
# ============================

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in env vars.")

# Use env var or default
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

base_url = config["llm"]["base_url"]
MODEL_NAME = config["llm"]["model_name"]

client = OpenAI(base_url=base_url, api_key=api_key)

# ============================
# LLM Client
# ============================

client = OpenAI(base_url=base_url, api_key=api_key)

# ============================
# Define State Schema
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
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who rewrites user queries for clarity and LLM performance.",
            },
            {"role": "user", "content": f"Rewrite this user query: {query}"},
        ],
    )
    rewritten_query = response.choices[0].message.content.strip()
    return {**state, "rewritten_query": rewritten_query}


# ============================
# Planner Agent
# ============================


def planner_agent_node(state: KlugekopfState) -> KlugekopfState:
    rewritten_query = state["rewritten_query"]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a strategic planner. Break the query into clear tasks.",
            },
            {"role": "user", "content": f"Break down this task: {rewritten_query}"},
        ],
    )
    plan = response.choices[0].message.content.strip()
    return {**state, "plan": plan}


# ============================
# Retrieval Agent
# ============================


def retrieval_agent_node(state: KlugekopfState) -> KlugekopfState:
    rewritten_query = state["rewritten_query"]
    chunks, metadatas = retrieve_context(rewritten_query)
    return {**state, "chunks": chunks, "metadatas": metadatas}


# ============================
# Summarizer Agent
# ============================


def summarizer_agent_node(state: KlugekopfState) -> KlugekopfState:
    chunks = state["chunks"]
    text = "\n\n".join(chunks)
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
    return {**state, "summary": summary}


# ============================
# Tool Agent
# ============================


def tool_agent_node(state: KlugekopfState) -> KlugekopfState:
    # Example: call to an external tool or API
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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": final_prompt},
        ],
    )
    return {**state, "answer": response.choices[0].message.content.strip()}


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

    # If you want to use a different config when run directly:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    user_query = input("Ask Klugekopf-Bot: ")
    result = klugekopf_multi_agent_app.invoke({"query": user_query})
    print("\nKlugekopf-Bot says:\n")
    print(result["answer"])
