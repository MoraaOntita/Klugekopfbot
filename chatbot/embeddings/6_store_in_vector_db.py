import os
import json
import yaml
import argparse
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def load_chunks(filepath):
    """Load your chunk metadata — only text and source are needed."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def setup_pinecone(index_name: str, dimension: int):
    """Create Pinecone index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"ℹ️ Pinecone index already exists: {index_name}")

    return pc

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    embeddings_dir = config["data"]["embeddings_dir"]
    embeddings_filename = config["embeddings"]["output_filename"]
    embeddings_path = os.path.join(embeddings_dir, embeddings_filename)

    index_name = config["vector_db"]["index_name"]
    embedding_model_name = config["vector_db"]["embedding_model_name"]

    print("Loading chunk metadata...")
    data = load_chunks(embeddings_path)

    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"✅ Using embedding model: {embedding_model_name}")

    # Get dimension
    sample_embedding = embedding_function.embed_query("test")
    dimension = len(sample_embedding)
    print(f"✅ Detected embedding dimension: {dimension}")

    # Ensure index exists
    setup_pinecone(index_name, dimension)

    documents = [item["text"] for item in data]
    metadatas = [{"source": item["source"]} for item in data]

    # ✅ Let LangChain connect + upsert itself
    vectorstore = LangchainPinecone.from_texts(
        texts=documents,
        embedding=embedding_function,
        metadatas=metadatas,
        index_name=index_name,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    print(f"✅ Stored {len(documents)} chunks in Pinecone VectorStore.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store text chunks in a Pinecone vector database."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    main(args.config)
