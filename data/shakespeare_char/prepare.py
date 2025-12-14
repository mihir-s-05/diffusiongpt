"""
Prepare the tiny Shakespeare dataset for character-level language modeling.

This writes:
  - train.bin / val.bin: uint16 token IDs
  - meta.pkl: vocab + encoder/decoder tables (stoi/itos)

The output format matches what diffusionGPT/train.py expects.
"""

import os
import pickle
import urllib.request

import numpy as np


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

    print(f"length of dataset in characters: {len(data):,}")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    n = len(data)
    split = int(n * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids.tofile(os.path.join(THIS_DIR, "train.bin"))
    val_ids.tofile(os.path.join(THIS_DIR, "val.bin"))

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(THIS_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()

