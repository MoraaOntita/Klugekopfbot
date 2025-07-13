import os
import json
import yaml
import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_embeddings(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    embeddings_dir = config["data"]["embeddings_dir"]
    embeddings_filename = config["embeddings"]["output_filename"]
    embeddings_path = os.path.join(embeddings_dir, embeddings_filename)

    chroma_db_dir = config["data"]["vector_db_dir"]
    embedding_model_name = config["vector_db"]["embedding_model_name"]

    print("Loading embeddings...")
    data = load_embeddings(embeddings_path)

    print(f"Setting up LangChain Chroma VectorStore at: {chroma_db_dir}")

    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    documents = [item["text"] for item in data]
    embeddings = [item["embedding"] for item in data]
    metadatas = [{"source": item["source"]} for item in data]
    ids = [item["chunk_id"] for item in data]

    vectorstore = Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=embedding_function
    )

    vectorstore.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    vectorstore.persist()

    print(f"✅ Stored {len(documents)} chunks in Chroma VectorStore.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store embeddings in a Chroma vector database.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    main(args.config)