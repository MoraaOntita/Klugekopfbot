import os
from typing import List
import yaml
import argparse

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    extracted_dir = config["data"]["extracted_dir"]
    chunks_dir = config["data"]["chunks_dir"]
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    os.makedirs(chunks_dir, exist_ok=True)

    for filename in os.listdir(extracted_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(extracted_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(chunks_dir, f"{base_name}_chunks.txt")

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text, chunk_size, chunk_overlap)

            with open(output_path, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"--- CHUNK {i+1} ---\n{chunk}\n\n")

            print(f"{len(chunks)} chunks saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk extracted text files.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    main(args.config)
