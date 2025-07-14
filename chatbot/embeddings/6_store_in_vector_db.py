import os
import json
import yaml
import argparse
from langchain_community.vectorstores import FAISS
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

    faiss_db_path = config["vector_db"]["persist_directory"]
    embedding_model_name = config["vector_db"]["embedding_model_name"]

    print("Loading embeddings...")
    data = load_embeddings(embeddings_path)

    print(f"Setting up LangChain FAISS VectorStore at: {faiss_db_path}")

    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    documents = [item["text"] for item in data]
    embeddings = [item["embedding"] for item in data]
    metadatas = [{"source": item["source"]} for item in data]

    # Create FAISS index
    text_embeddings = list(zip(documents, embeddings))

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_function,
        metadatas=metadatas
    )


    # Save to disk
    faiss_index_path = os.path.join(faiss_db_path, "faiss_index")
    vectorstore.save_local(faiss_index_path)

    print(f"✅ Stored {len(documents)} chunks in FAISS VectorStore.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store embeddings in a FAISS vector database."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    main(args.config)
