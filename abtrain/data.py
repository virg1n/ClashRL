"""Data plumbing: download IL_Replay from Hugging Face, and stage shards into /dev/shm.

    python -m abtrain.data download   # pinned revision, verified against SHA256SUMS
    python -m abtrain.data stage      # copy shards to SHM (skipped if /dev/shm is too small)

Both are restartable: finished work is detected and skipped.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

from .common import HF_DATASET, HF_REVISION, RAW, SHARDS, SHM, log


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def raw_ready() -> bool:
    return (RAW / ".complete").exists() and any((RAW / "replays").glob("*.parquet"))


def download() -> None:
    if raw_ready():
        log(f"IL_Replay already present in {RAW}", "data")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    for attempt in range(8):
        try:
            snapshot_download(repo_id=HF_DATASET, repo_type="dataset", revision=HF_REVISION, local_dir=str(RAW),
                              allow_patterns=["replays/*.parquet", "manifest.json", "README.md", "SHA256SUMS",
                                              "compatibility.json"], max_workers=8)
            break
        except Exception as exc:
            log(f"download attempt {attempt + 1} failed: {exc!r}", "data")
            time.sleep(min(600, 30 * 2**attempt))
    else:
        raise SystemExit("IL_Replay download failed")
    sums = RAW / "SHA256SUMS"
    bad = []
    if sums.exists():
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            digest, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
            if not rel.startswith("replays/"):
                continue
            p = RAW / rel
            if not p.exists() or _sha256(p) != digest:
                bad.append(rel)
    if bad:
        for rel in bad:  # a fresh attempt re-fetches them
            (RAW / rel).unlink(missing_ok=True)
        raise SystemExit(f"{len(bad)} files failed the checksum: {bad[:5]}")
    n = len(list((RAW / "replays").glob("*.parquet")))
    if not n:
        raise SystemExit("no parquet files downloaded")
    (RAW / ".complete").write_text(str(time.time()))
    log(f"IL_Replay ready: {n} parquet parts in {RAW}", "data")


def stage() -> None:
    """Copy shards into /dev/shm (RAM) so BC never reads them over NFS."""
    dst = SHM / "shards"
    if (dst / ".complete").exists():
        log("shards already staged", "data")
        return
    files = sorted(p for p in SHARDS.rglob("*.npz") if not p.name.startswith("."))
    need = sum(p.stat().st_size for p in files)
    dst.mkdir(parents=True, exist_ok=True)
    have = sum(p.stat().st_size for p in dst.rglob("*.npz"))
    free = shutil.disk_usage(dst).free
    if need - have > free - 20 * 2**30:  # keep 20 GB of headroom for rollouts
        log(f"not staging: shards need {need / 2**30:.1f} GB, {free / 2**30:.1f} GB free in {dst}; "
            f"training reads from {SHARDS}", "data")
        shutil.rmtree(dst, ignore_errors=True)
        return
    t0 = time.time()
    for p in files:
        q = dst / p.relative_to(SHARDS)
        if q.exists() and q.stat().st_size == p.stat().st_size:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        tmp = q.with_name(f".{q.name}.tmp{os.getpid()}")
        shutil.copyfile(p, tmp)
        os.replace(tmp, q)
    (dst / ".complete").write_text(str(time.time()))
    log(f"staged {len(files)} shards ({need / 2**30:.1f} GB) to {dst} in {(time.time() - t0) / 60:.1f} min", "data")


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["download"])[0]
    {"download": download, "stage": stage}[cmd]()


if __name__ == "__main__":
    main()
