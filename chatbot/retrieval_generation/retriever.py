import yaml
import argparse
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load config
parser = argparse.ArgumentParser(description="Retriever with vector DB")
parser.add_argument(
    "--config",
    type=str,
    default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    help="Path to YAML config file"
)
args = parser.parse_args()
config = load_config(args.config)

# Get settings
chroma_dir = config["vector_db"]["persist_directory"]
embedding_model_name = config["vector_db"]["embedding_model_name"]

# Init embeddings + vector store
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embedding_function
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

def retrieve_context(query: str, n_results: int = 4):
    """
    Retrieve relevant chunks for a given query using the vector store retriever.
    """
    retriever.search_kwargs["k"] = n_results
    docs = retriever.invoke(query)
    documents = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return documents, metadatas

if __name__ == "__main__":
    query = input("Enter your query: ")
    docs, meta = retrieve_context(query)
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}\n{doc}\nSource: {meta[i].get('source', 'N/A')}")