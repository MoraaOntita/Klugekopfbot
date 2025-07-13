import os
import json
import re
import yaml
import argparse

def extract_chunks_from_text_file(filepath):
    """Extract individual chunks from a text file with '--- CHUNK N ---' headers."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split on headers like --- CHUNK 1 ---
    raw_chunks = re.split(r"--- CHUNK \d+ ---\n", content)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]

def create_chunk_metadata(chunks, source_name):
    """Wrap chunks in metadata structure."""
    chunk_metadata_list = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": f"{source_name}_{str(i+1).zfill(3)}",
            "text": chunk,
            "source": f"{source_name}.docx"
        }
        chunk_metadata_list.append(metadata)
    return chunk_metadata_list

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    chunks_dir = config["data"]["chunks_dir"]
    metadata_dir = config["data"]["metadata_dir"]

    os.makedirs(metadata_dir, exist_ok=True)

    all_chunks_metadata = []

    for filename in os.listdir(chunks_dir):
        if filename.endswith("_chunks.txt"):
            filepath = os.path.join(chunks_dir, filename)
            base_name = filename.replace("_chunks.txt", "")

            print(f"🧩 Processing chunks from: {filename}")
            chunks = extract_chunks_from_text_file(filepath)
            structured_chunks = create_chunk_metadata(chunks, base_name)

            # Save one file per document
            output_path = os.path.join(metadata_dir, f"{base_name}_metadata.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_chunks, f, ensure_ascii=False, indent=2)

            all_chunks_metadata.extend(structured_chunks)

    # Optional: Save a master .jsonl file for embedding
    master_path = os.path.join(metadata_dir, "all_chunks_metadata.jsonl")
    with open(master_path, "w", encoding="utf-8") as f:
        for item in all_chunks_metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Metadata saved to: {metadata_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create metadata from text chunks.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    main(args.config)