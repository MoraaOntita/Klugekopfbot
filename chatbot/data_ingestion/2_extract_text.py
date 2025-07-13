from docx import Document
import os
import yaml
import argparse

def extract_text_from_docx(docx_path: str) -> str:
    """Extract and clean text from a .docx file."""
    doc = Document(docx_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_docs_dir = config["data"]["raw_docs_dir"]
    extracted_dir = config["data"]["extracted_dir"]

    os.makedirs(extracted_dir, exist_ok=True)

    for filename in os.listdir(raw_docs_dir):
        if filename.endswith(".docx"):
            docx_path = os.path.join(raw_docs_dir, filename)
            name = os.path.splitext(filename)[0]
            output_path = os.path.join(extracted_dir, f"{name}.txt")

            print(f"Extracting: {filename}")
            text = extract_text_from_docx(docx_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from docx files.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to the YAML config file"
    )
    args = parser.parse_args()

    main(args.config)
