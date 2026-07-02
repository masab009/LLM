"""
Download VECTORinc/UrduLLM-4B-Distilled from Hugging Face to the Seagate drive.

Usage:
    python download_model.py

Set your token via environment variable (recommended):
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    python download_model.py

Or replace the placeholder below directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# ── Configuration ────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODEL_REPO = "VECTORinc/UrduLLM-4B-Distilled"

DOWNLOAD_DIR = "/media/vector/Seagate-2TB/RBT/models/UrduLLM-4B-Distilled"

# ─────────────────────────────────────────────────────────────────────────────


def main():
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not found. Make sure backend/LLM/.env contains:\n"
            "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
        )

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading  : {MODEL_REPO}")
    print(f"Destination  : {DOWNLOAD_DIR}")
    print("This may take a while depending on your connection speed...\n")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=DOWNLOAD_DIR,
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )

    print(f"\nDone. Model saved to: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
