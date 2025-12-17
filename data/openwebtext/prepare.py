"""
Prepare OpenWebText into GPT-2 BPE token ids (train.bin/val.bin).
"""

import os
import shutil
import hashlib
from pathlib import Path
import re

import numpy as np
import tiktoken

try:
    import datasets as hf_datasets
    from datasets import load_dataset
    from datasets.exceptions import DatasetGenerationError
except ImportError as e:
    raise SystemExit(
        "Missing dependency for OpenWebText prep: `datasets`.\n"
        "Install with: pip install datasets\n"
        f"Original import error: {e}"
    ) from e

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "Missing dependency for OpenWebText prep: `tqdm`.\n"
        "Install with: pip install tqdm\n"
        f"Original import error: {e}"
    ) from e


THIS_DIR = Path(__file__).resolve().parent


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _ensure_local_hf_cache() -> Path | None:
    """Use a repo-local HuggingFace cache unless the user configured one."""

    if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_DATASETS_CACHE"):
        return None

    local = THIS_DIR / "_hf_cache"
    local.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(local)
    os.environ["HF_HUB_CACHE"] = str(local / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(local / "datasets")
    return local


_DEFAULT_NUM_PROC = 1 if os.name == "nt" else 8
num_proc = int(os.environ.get("OWT_NUM_PROC", str(_DEFAULT_NUM_PROC)))

num_proc_load_dataset = int(os.environ.get("OWT_NUM_PROC_LOAD_DATASET", str(num_proc)))

enc = tiktoken.get_encoding("gpt2")


def _cleanup_zero_byte_openwebtext_tars() -> int:
    """
    HuggingFace cache downloads can occasionally be interrupted (or quarantined),
    leaving behind 0-byte subset tar files. Those will break dataset generation.

    Deleting them is safe: they will be re-downloaded on the next load_dataset().
    """

    hub_root = os.environ.get("HF_HUB_CACHE", "").strip()
    if not hub_root:
        hub_root = str(Path.home() / ".cache" / "huggingface" / "hub")

    snapshots_dir = Path(hub_root) / "datasets--openwebtext" / "snapshots"
    if not snapshots_dir.exists():
        return 0

    deleted = 0
    for tar_path in snapshots_dir.glob("**/subsets/urlsf_subset*.tar"):
        try:
            if tar_path.is_file() and tar_path.stat().st_size == 0:
                tar_path.unlink()
                deleted += 1
        except OSError:
            pass
    return deleted


def _assign_split(text: str, val_mod: int) -> str:
    """
    Deterministically assign example to train/val based on a stable hash of text.

    val_mod=2000 gives ~0.05% validation split (similar to the original script).
    """

    if val_mod <= 0:
        raise ValueError("val_mod must be > 0")
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    v = int.from_bytes(h, byteorder="little", signed=False)
    return "val" if (v % val_mod) == 0 else "train"


def _streaming_prepare(val_mod: int, max_docs: int | None, max_tokens: int | None) -> None:
    """
    Streaming tokenization path (recommended on Windows):
    - avoids generating a large on-disk cached dataset
    - writes train.bin/val.bin incrementally without loading everything into RAM

    Note: Streaming may still download shards, but it typically avoids keeping giant extracted artifacts.
    """

    ds = load_dataset("openwebtext", streaming=True, trust_remote_code=True)

    train_path = THIS_DIR / "train.bin"
    val_path = THIS_DIR / "val.bin"
    for p in (train_path, val_path):
        if p.exists():
            p.unlink()

    buffers: dict[str, list[int]] = {"train": [], "val": []}
    token_counts: dict[str, int] = {"train": 0, "val": 0}
    doc_counts: dict[str, int] = {"train": 0, "val": 0}

    flush_threshold = int(os.environ.get("OWT_FLUSH_TOKENS", "1000000"))

    def flush(split: str) -> None:
        buf = buffers[split]
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint16)
        out_path = train_path if split == "train" else val_path
        with open(out_path, "ab") as f:
            f.write(arr.tobytes(order="C"))
        token_counts[split] += int(arr.size)
        buffers[split].clear()

    it = iter(ds["train"])
    processed = 0
    for ex in it:
        text = ex.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue

        split = _assign_split(text, val_mod=val_mod)
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        buffers[split].extend(ids)
        doc_counts[split] += 1
        processed += 1

        if max_docs is not None and processed >= max_docs:
            break
        if max_tokens is not None and (token_counts["train"] + token_counts["val"] + len(buffers["train"]) + len(buffers["val"])) >= max_tokens:
            break

        if len(buffers[split]) >= flush_threshold:
            flush(split)

        if processed % 10000 == 0:
            pending = len(buffers["train"]) + len(buffers["val"])
            total = token_counts["train"] + token_counts["val"] + pending
            print(
                f"processed docs={processed:,} | tokens_written train={token_counts['train']:,} val={token_counts['val']:,} | pending={pending:,} | total~{total:,}"
            )

    flush("train")
    flush("val")
    print("done.")
    print(f"train docs: {doc_counts['train']:,} | tokens: {token_counts['train']:,} | file: {train_path}")
    print(f"val docs:   {doc_counts['val']:,} | tokens: {token_counts['val']:,} | file: {val_path}")


if __name__ == "__main__":
    local_cache = _ensure_local_hf_cache()

    try:
        datasets_major = int(str(hf_datasets.__version__).split(".", maxsplit=1)[0])
    except Exception:
        datasets_major = None
    if datasets_major is not None and datasets_major >= 4:
        raise SystemExit(
            f"Your installed `datasets` version ({hf_datasets.__version__}) no longer supports dataset scripts "
            "(like OpenWebText's `openwebtext.py`).\n"
            "Fix: pip install \"datasets<4\""
        )

    streaming_default = os.name == "nt"
    use_streaming = _truthy_env("OWT_STREAMING", streaming_default)

    val_mod = int(os.environ.get("OWT_VAL_MOD", "2000"))
    max_docs = os.environ.get("OWT_MAX_DOCS", "").strip()
    max_tokens = os.environ.get("OWT_MAX_TOKENS", "").strip()
    max_docs_i = int(max_docs) if max_docs else None
    max_tokens_i = int(max_tokens) if max_tokens else None

    if use_streaming:
        _streaming_prepare(val_mod=val_mod, max_docs=max_docs_i, max_tokens=max_tokens_i)
        delete_cache = _truthy_env("OWT_DELETE_CACHE", os.name == "nt")
        if delete_cache and local_cache and local_cache.exists():
            try:
                shutil.rmtree(local_cache, ignore_errors=True)
                print(f"Deleted isolated HF cache: {local_cache}")
            except OSError:
                pass
        raise SystemExit(0)

    deleted = _cleanup_zero_byte_openwebtext_tars()
    if deleted:
        print(f"Deleted {deleted} zero-byte OpenWebText subset tar file(s) from HF cache; re-downloading as needed.")

    try:
        try:
            dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)
        except TypeError:
            dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    except DatasetGenerationError as e:
        msg = str(e)
        m = re.search(r"Invalid argument: '([^']+urlsf_subset\\d+\\.tar)'", msg)
        if m:
            bad = m.group(1)
            raise SystemExit(
                "OpenWebText generation failed while reading a cached subset tar.\n"
                f"Try deleting this file and rerunning:\n  {bad}\n\n"
                "If this keeps happening, set a shorter cache path (e.g. HF_HOME=C:\\hf) and/or reduce workers:\n"
                "  set OWT_NUM_PROC=1\n  set OWT_NUM_PROC_LOAD_DATASET=1"
            ) from e
        raise

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        out = {"ids": ids, "len": len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
