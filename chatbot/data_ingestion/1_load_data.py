import gdown
import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    help="Path to YAML config"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

raw_docs_dir = config["data"]["raw_docs_dir"]
files = config["data"]["files"]

os.makedirs(raw_docs_dir, exist_ok=True)

for file_info in files:
    url = file_info["url"]
    output_path = os.path.join(raw_docs_dir, file_info["output"])
    print(f"Downloading {file_info['name']}...")
    gdown.download(url=url, output=output_path, quiet=False)
