from docx import Document
import PyPDF2
import os
import yaml
import argparse

def extract_text_from_docx(docx_path: str) -> str:
    """Extract and clean text from a .docx file."""
    doc = Document(docx_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text.strip())
    return "\n".join(text)

def is_pdf(file_path: str) -> bool:
    """Check if file is a PDF by signature."""
    with open(file_path, "rb") as f:
        sig = f.read(4)
    return sig == b"%PDF"

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_docs_dir = config["data"]["raw_docs_dir"]
    extracted_dir = config["data"]["extracted_dir"]

    os.makedirs(extracted_dir, exist_ok=True)

    for filename in os.listdir(raw_docs_dir):
        file_path = os.path.join(raw_docs_dir, filename)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(extracted_dir, f"{name}.txt")

        print(f"Extracting: {filename}")

        if ext.lower() == ".docx":
            try:
                text = extract_text_from_docx(file_path)
            except Exception as e:
                print(f"❌ Failed to extract DOCX: {filename} — {e}")
                continue

        elif ext.lower() == ".pdf" or is_pdf(file_path):
            try:
                text = extract_text_from_pdf(file_path)
            except Exception as e:
                print(f"❌ Failed to extract PDF: {filename} — {e}")
                continue

        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from DOCX and PDF files.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        help="Path to the YAML config file"
    )
    args = parser.parse_args()

    main(args.config)
