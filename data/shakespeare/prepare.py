"""
Prepare the tiny Shakespeare dataset for GPT-2 BPE tokenization (tiktoken).

This writes:
  - train.bin / val.bin: uint16 GPT-2 token IDs

The output format matches what diffusionGPT/train.py expects.
"""

import os
import urllib.request

import numpy as np
import tiktoken


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
THIS_DIR = os.path.dirname(__file__)


def _download_if_missing(path: str) -> None:
    if os.path.exists(path):
        return
    print(f"downloading {DATA_URL} to {path}...")
    with urllib.request.urlopen(DATA_URL) as r:
        text = r.read().decode("utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    input_file_path = os.path.join(THIS_DIR, "input.txt")
    _download_if_missing(input_file_path)

    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    split = int(n * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(THIS_DIR, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(THIS_DIR, "val.bin"))


if __name__ == "__main__":
    main()

