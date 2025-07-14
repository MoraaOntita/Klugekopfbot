import os
import sys
import yaml
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to PYTHONPATH so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.retrieval_generation.retriever import retrieve_context
from chatbot.retrieval_generation.prompts import get_klugekopf_system_prompt

# -------------------------------
# Load .env and API key
# -------------------------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env. Please set it!")

# -------------------------------
# Load config
# -------------------------------

parser = argparse.ArgumentParser(description="Generate answer using Groq LLM.")
parser.add_argument(
    "--config",
    type=str,
    default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    help="Path to YAML config file",
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

base_url = config["llm"]["base_url"]
model_name = config["llm"]["model_name"]

# -------------------------------
# Groq client — OpenAI compatible
# -------------------------------

client = OpenAI(base_url=base_url, api_key=api_key)

# -------------------------------
# Prompt builder
# -------------------------------


def build_prompt(query: str, context_chunks: list[str]) -> tuple[str, str]:
    context = "\n\n".join(context_chunks)
    system_message = get_klugekopf_system_prompt()
    user_message = f"Context:\n{context}\n\n" f"Question:\n{query}\n\n" f"Answer:"
    return system_message, user_message


# -------------------------------
# Call LLM
# -------------------------------


def generate_answer(system_message: str, user_message: str) -> str:
    """
    Calls Groq's LLaMA3 to generate the answer.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# -------------------------------
# Orchestrate: Retrieve → Prompt → LLM
# -------------------------------


def run_pipeline(query: str) -> str:
    chunks, _ = retrieve_context(query)  # ✅ now uses Pinecone retriever
    system_message, user_message = build_prompt(query, chunks)
    return generate_answer(system_message, user_message)


# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    query = input("Ask Klugekopf-Bot: ")
    answer = run_pipeline(query)
    print("\nKlugekopf-Bot says:\n")
    print(answer)
    print(answer)
