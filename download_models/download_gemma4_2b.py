"""
Download google/gemma-4-E2B from Hugging Face to the Seagate drive.

Usage:
    python download_gemma4_2b.py

HF_TOKEN is read from backend/LLM/.env automatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(Path(__file__).parent / ".env")

MODEL_REPO   = "google/gemma-4-E2B"
DOWNLOAD_DIR = "/media/vector/Seagate-2TB/RBT/models/gemma-4-E2B"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")


def main():
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not found. Make sure backend/LLM/.env contains:\n"
            "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
        )

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading  : {MODEL_REPO}")
    print(f"Destination  : {DOWNLOAD_DIR}")
    print("Downloading ~4 GB — this may take a few minutes...\n")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=DOWNLOAD_DIR,
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )

    print(f"\nDone. Model saved to: {DOWNLOAD_DIR}")
    print(f"\nTo serve it, set in backend/LLM/.env:")
    print(f"  MODEL_PATH={DOWNLOAD_DIR}")
    print(f"  MAX_MODEL_LEN=8192")
    print(f"Then restart the LLM server.")


if __name__ == "__main__":
    main()
