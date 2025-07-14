# chatbot/retrieval_generation/retriever.py

import yaml
import os
from functools import lru_cache
from dotenv import load_dotenv

from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# Config loader
# -------------------------------

def load_config():
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# ✅ Get from env first, fallback to config.yaml
index_name = os.getenv("PINECONE_INDEX_NAME") or config["vector_db"].get("index_name")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME") or config["vector_db"].get("embedding_model_name")

if not index_name:
    raise ValueError("❌ Pinecone index name not set. Add PINECONE_INDEX_NAME to your .env or config.yaml.")

# -------------------------------
# Lazy factory for Pinecone vectorstore
# -------------------------------

@lru_cache(maxsize=1)
def get_vectorstore():
    """
    Lazily initialize the Pinecone vectorstore.
    """
    # 1️⃣ Create embedding model
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 2️⃣ Connect to Pinecone index
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # 3️⃣ Wrap with LangChain Pinecone
    vectorstore = LangchainPinecone(
        index=index,
        embedding=embedding_function,
        text_key="text",
    )

    return vectorstore

# -------------------------------
# Retrieval function
# -------------------------------

def retrieve_context(query: str, n_results: int = 4):
    """
    Retrieve relevant chunks for a given query using the Pinecone vector store.
    """
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": n_results}
    )

    docs = retriever.invoke(query)
    documents = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return documents, metadatas

# -------------------------------
# CLI usage only
# -------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retriever with Pinecone vector DB")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config()
    query = input("Enter your query: ")
    docs, meta = retrieve_context(query)
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}\n{doc}\nSource: {meta[i].get('source', 'N/A')}")
