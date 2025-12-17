"""
Prepare WikiText-103 (raw) into GPT-2 BPE token ids (train.bin/val.bin).

Usage:
  python data/wikitext103/prepare.py
"""

import os

import numpy as np
import tiktoken

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(
        "Missing dependency for WikiText prep: `datasets`.\n"
        "Install with: pip install \"datasets<4\""
    ) from e

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "Missing dependency for WikiText prep: `tqdm`.\n"
        "Install with: pip install tqdm"
    ) from e


THIS_DIR = os.path.dirname(__file__)

_DEFAULT_NUM_PROC = 1 if os.name == "nt" else 8
num_proc = int(os.environ.get("WIKITEXT_NUM_PROC", str(_DEFAULT_NUM_PROC)))

enc = tiktoken.get_encoding("gpt2")


def process(example: dict) -> dict:
    text = example.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return {"ids": [], "len": 0}
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


def write_split(dset, filename: str) -> None:
    arr_len = int(np.sum(dset["len"], dtype=np.uint64))
    path = os.path.join(THIS_DIR, filename)
    dtype = np.uint16
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=(arr_len,))

    total_shards = 1024
    idx = 0
    for shard_idx in tqdm(range(total_shards), desc=f"writing {filename}"):
        batch = dset.shard(num_shards=total_shards, index=shard_idx, contiguous=True).with_format("numpy")
        if len(batch["ids"]) == 0:
            continue
        chunk = np.concatenate(batch["ids"]) if len(batch["ids"]) > 1 else batch["ids"][0]
        if chunk.size == 0:
            continue
        arr[idx : idx + len(chunk)] = chunk
        idx += len(chunk)

    arr.flush()
    if idx != arr_len:
        raise RuntimeError(f"wrote {idx} tokens but expected {arr_len} for {filename}")


def main() -> None:
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")

    train = raw["train"]
    val = raw["validation"]

    train_tok = train.map(process, remove_columns=train.column_names, num_proc=num_proc, desc="tokenizing train")
    val_tok = val.map(process, remove_columns=val.column_names, num_proc=num_proc, desc="tokenizing val")

    train_tok = train_tok.filter(lambda x: x["len"] > 0, num_proc=num_proc, desc="filtering train")
    val_tok = val_tok.filter(lambda x: x["len"] > 0, num_proc=num_proc, desc="filtering val")

    print(f"train documents: {len(train_tok):,}")
    print(f"val documents:   {len(val_tok):,}")
    print(f"train tokens:    {int(np.sum(train_tok['len'], dtype=np.uint64)):,}")
    print(f"val tokens:      {int(np.sum(val_tok['len'], dtype=np.uint64)):,}")

    write_split(train_tok, "train.bin")
    write_split(val_tok, "val.bin")


if __name__ == "__main__":
    main()

