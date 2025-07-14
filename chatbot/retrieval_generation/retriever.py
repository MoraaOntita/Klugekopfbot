import yaml
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -------------------------------
# Config loader
# -------------------------------

def load_config():
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# -------------------------------
# Vector DB settings
# -------------------------------

chroma_dir = config["vector_db"]["persist_directory"]
embedding_model_name = config["vector_db"]["embedding_model_name"]

# -------------------------------
# Init embeddings + vector store
# -------------------------------

embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embedding_function
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -------------------------------
# Retrieval function
# -------------------------------

def retrieve_context(query: str, n_results: int = 4):
    """
    Retrieve relevant chunks for a given query using the vector store retriever.
    """
    retriever.search_kwargs["k"] = n_results
    docs = retriever.invoke(query)
    documents = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return documents, metadatas

# -------------------------------
# CLI usage only
# -------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retriever with vector DB")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # If running from CLI with custom config path:
    config = load_config()
    query = input("Enter your query: ")
    docs, meta = retrieve_context(query)
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}\n{doc}\nSource: {meta[i].get('source', 'N/A')}")
