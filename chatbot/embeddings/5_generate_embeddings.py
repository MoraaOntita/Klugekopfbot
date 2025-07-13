import json
import os
import yaml
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

def load_chunks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_embeddings(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metadata_path = os.path.join(config["data"]["metadata_dir"], "all_chunks_metadata.jsonl")
    embeddings_dir = config["data"]["embeddings_dir"]
    embeddings_filename = config["embeddings"]["output_filename"]
    embeddings_path = os.path.join(embeddings_dir, embeddings_filename)
    embedding_model_name = config["embeddings"]["model_name"]

    os.makedirs(embeddings_dir, exist_ok=True)

    print("Loading chunk metadata...")
    chunks = load_chunks(metadata_path)

    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    embedded_data = []
    for chunk in tqdm(chunks, desc="Generating embeddings"):
        text = chunk["text"]
        embedding = embedding_model.embed_query(text)

        embedded_data.append({
            "chunk_id": chunk["chunk_id"],
            "embedding": embedding,
            "text": text,
            "source": chunk["source"]
        })

    print(f"Saving embeddings to: {embeddings_path}")
    save_embeddings(embedded_data, embeddings_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    main(args.config)