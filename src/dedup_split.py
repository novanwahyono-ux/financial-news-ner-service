"""
dedup_split.py — Dedup near-duplicate dan split data 70/15/15
==============================================================
Sesuai Fase 2 brief (Bagian 5):
  1. Grup duplikat / near-duplicate:
     - Hash eksak (normalized body) → duplikat pasti
     - MinHash LSH (datasketch) → near-duplicate (Jaccard >= threshold)
     Duplikat/near-dup dikelompokkan terlebih dahulu; split dilakukan
     per GRUP agar tidak ada kebocoran antar split.
  2. Split PER ARTIKEL/GRUP — seed deterministik (config.SEED)
     Rasio: 70% train / 15% val / 15% test
  3. Hasil disimpan ke: data/splits/splits.json
     Format: {"train": [...], "val": [...], "test": [...]}
     List berisi original_id unik per split.
  4. Verifikasi tidak ada kebocoran: tidak ada original_id yang ada
     di lebih dari satu split.

Penggunaan:
    python -m src.dedup_split          # jalankan dedup + split
    python -m src.dedup_split --dry-run  # tampilkan statistik saja
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from src.config import (
    DATA_SPLITS_DIR,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data_loader import load_all_csvs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Konstanta
# ──────────────────────────────────────────────────────────────
MINHASH_NUM_PERM: int = 128          # jumlah permutasi MinHash
MINHASH_THRESHOLD: float = 0.85     # Jaccard >= 0.85 → near-duplicate
NGRAM_SIZE: int = 3                  # n-gram karakter untuk shingling

SPLITS_FILE: Path = DATA_SPLITS_DIR / "splits.json"
DEDUP_REPORT_FILE: Path = DATA_SPLITS_DIR / "dedup_report.json"


# ──────────────────────────────────────────────────────────────
# Normalisasi teks
# ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip whitespace, hapus tanda baca berlebihan."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def body_hash(body: str) -> str:
    """SHA-256 dari body yang dinormalisasi — untuk dedup eksak."""
    return hashlib.sha256(normalize(body).encode("utf-8")).hexdigest()


def make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    """Buat MinHash dari n-gram karakter teks."""
    m = MinHash(num_perm=num_perm)
    norm = normalize(text)
    shingles = {norm[i : i + NGRAM_SIZE] for i in range(max(1, len(norm) - NGRAM_SIZE + 1))}
    for shingle in shingles:
        m.update(shingle.encode("utf-8"))
    return m


# ──────────────────────────────────────────────────────────────
# Dedup: eksak + near-duplicate → grup
# ──────────────────────────────────────────────────────────────

def build_groups(df_unique: pd.DataFrame) -> dict[str, int]:
    """
    Beri setiap original_id label grup (int).
    Artikel dalam satu grup harus masuk split yang sama.

    Strategi:
    1. Kelompokkan berdasarkan hash body (exact-dup) → awal union-find
    2. Perkaya dengan MinHash LSH untuk near-dup

    Returns
    -------
    dict: {original_id -> group_id}
    """
    ids = df_unique["original_id"].tolist()
    bodies = df_unique["body"].fillna("").tolist()
    n = len(ids)

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    idx_map = {oid: i for i, oid in enumerate(ids)}

    # ── Langkah 1: exact dedup berdasarkan hash body ──────────
    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, body in enumerate(bodies):
        h = body_hash(body)
        hash_to_indices[h].append(i)

    exact_groups = 0
    for indices in hash_to_indices.values():
        if len(indices) > 1:
            for j in range(1, len(indices)):
                union(indices[0], indices[j])
            exact_groups += 1

    logger.info("Exact-dup groups: %d", exact_groups)

    # ── Langkah 2: MinHash LSH untuk near-dup ────────────────
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    minhashes: list[MinHash] = []

    for i, (oid, body) in enumerate(zip(ids, bodies)):
        m = make_minhash(body)
        minhashes.append(m)
        lsh.insert(oid, m)

    near_dup_pairs = 0
    for i, (oid, m) in enumerate(zip(ids, minhashes)):
        candidates = lsh.query(m)
        for candidate_oid in candidates:
            if candidate_oid != oid:
                j = idx_map[candidate_oid]
                if find(i) != find(j):
                    union(i, j)
                    near_dup_pairs += 1

    logger.info("Near-dup pairs merged: %d", near_dup_pairs)

    # ── Buat mapping original_id → group_id ──────────────────
    root_to_group: dict[int, int] = {}
    group_counter = 0
    id_to_group: dict[str, int] = {}

    for i, oid in enumerate(ids):
        root = find(i)
        if root not in root_to_group:
            root_to_group[root] = group_counter
            group_counter += 1
        id_to_group[oid] = root_to_group[root]

    total_groups = group_counter
    multi_member_groups = sum(
        1 for g in set(id_to_group.values())
        if list(id_to_group.values()).count(g) > 1
    )
    logger.info(
        "Total grup: %d (%d grup dengan >1 anggota)", total_groups, multi_member_groups
    )
    return id_to_group


# ──────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────

def split_groups(
    id_to_group: dict[str, int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """
    Split original_id ke train/val/test berdasarkan grup.

    1. Kumpulkan semua grup unik
    2. Shuffle grup dengan seed deterministik
    3. Alokasikan grup ke split berdasarkan rasio
    4. Semua original_id dalam satu grup masuk ke split yang sama

    Returns
    -------
    dict: {"train": [...], "val": [...], "test": [...]}
    """
    rng = np.random.default_rng(seed)

    # Grup unik → list of original_ids
    group_to_ids: dict[int, list[str]] = defaultdict(list)
    for oid, grp in id_to_group.items():
        group_to_ids[grp].append(oid)

    group_ids = np.array(sorted(group_to_ids.keys()))
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    # test = sisa

    train_groups = group_ids[:n_train]
    val_groups = group_ids[n_train : n_train + n_val]
    test_groups = group_ids[n_train + n_val :]

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for grp in train_groups:
        splits["train"].extend(group_to_ids[grp])
    for grp in val_groups:
        splits["val"].extend(group_to_ids[grp])
    for grp in test_groups:
        splits["test"].extend(group_to_ids[grp])

    return splits


# ──────────────────────────────────────────────────────────────
# Verifikasi tidak ada kebocoran
# ──────────────────────────────────────────────────────────────

def verify_no_leakage(splits: dict[str, list[str]]) -> bool:
    """
    Pastikan tidak ada original_id yang muncul di lebih dari satu split.
    Raise AssertionError jika ada kebocoran.
    """
    all_ids: list[str] = []
    for split_name, ids in splits.items():
        all_ids.extend(ids)

    total = len(all_ids)
    unique = len(set(all_ids))

    if total != unique:
        duplicates = total - unique
        logger.error("KEBOCORAN TERDETEKSI: %d original_id ada di lebih dari satu split!", duplicates)
        return False

    logger.info("Verifikasi kebocoran: OK (tidak ada original_id ganda antar split)")
    return True


# ──────────────────────────────────────────────────────────────
# Laporan statistik split
# ──────────────────────────────────────────────────────────────

def print_split_report(splits: dict[str, list[str]], total_unique: int) -> None:
    total_split = sum(len(v) for v in splits.values())
    sep = "=" * 60
    print(f"\n{sep}")
    print("  LAPORAN SPLIT DATA")
    print(sep)
    print(f"\n  Seed         : {SEED}")
    print(f"  Threshold NearDup: {MINHASH_THRESHOLD} (Jaccard)")
    print(f"  Total ID unik (original_id): {total_unique}")
    print(f"  Total dalam splits : {total_split}")
    print()
    for split_name, ids in splits.items():
        pct = len(ids) / total_split * 100 if total_split else 0
        print(f"  {split_name.upper():<8} : {len(ids):>4} ID  ({pct:.1f}%)")
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    logger.info("=== Fase 2: Dedup & Split ===")

    df = load_all_csvs()

    # Ambil baris unik berdasarkan original_id
    df_unique = df.drop_duplicates(subset=["original_id"]).reset_index(drop=True)
    logger.info("Artikel unik (berdasar original_id): %d", len(df_unique))

    logger.info("Membangun grup duplikat/near-dup...")
    id_to_group = build_groups(df_unique)

    logger.info("Membagi ke split train/val/test (seed=%d)...", SEED)
    splits = split_groups(id_to_group)

    ok = verify_no_leakage(splits)
    if not ok:
        raise RuntimeError("Split gagal: ada kebocoran antar split!")

    print_split_report(splits, total_unique=len(df_unique))

    if dry_run:
        logger.info("Dry-run: tidak ada file yang disimpan.")
        return

    # Simpan splits.json
    output = {
        "meta": {
            "seed": SEED,
            "minhash_threshold": MINHASH_THRESHOLD,
            "minhash_num_perm": MINHASH_NUM_PERM,
            "ngram_size": NGRAM_SIZE,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "total_unique_ids": len(df_unique),
            "split_counts": {k: len(v) for k, v in splits.items()},
        },
        "train": sorted(splits["train"]),
        "val": sorted(splits["val"]),
        "test": sorted(splits["test"]),
    }

    with open(SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Splits disimpan ke: %s", SPLITS_FILE)

    # Simpan dedup report
    group_sizes = defaultdict(int)
    for grp in id_to_group.values():
        group_sizes[grp] += 1

    dedup_report = {
        "total_original_ids": len(df_unique),
        "total_groups": len(set(id_to_group.values())),
        "singleton_groups": sum(1 for s in group_sizes.values() if s == 1),
        "multi_member_groups": sum(1 for s in group_sizes.values() if s > 1),
        "largest_group_size": max(group_sizes.values()),
    }
    with open(DEDUP_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(dedup_report, f, ensure_ascii=False, indent=2)
    logger.info("Dedup report disimpan ke: %s", DEDUP_REPORT_FILE)


def run(df: pd.DataFrame | None = None) -> dict:
    """Run deduplication and return both supported split-file layouts."""
    df = load_all_csvs() if df is None else df
    df_unique = df.drop_duplicates(subset=["original_id"]).reset_index(drop=True)
    id_to_group = build_groups(df_unique)
    splits = split_groups(id_to_group)

    if not verify_no_leakage(splits):
        raise RuntimeError("Split gagal: ada kebocoran antar split!")

    group_sizes = defaultdict(int)
    for group in id_to_group.values():
        group_sizes[group] += 1

    output = {
        "meta": {
            "seed": SEED,
            "minhash_threshold": MINHASH_THRESHOLD,
            "minhash_num_perm": MINHASH_NUM_PERM,
            "ngram_size": NGRAM_SIZE,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "total_unique_ids": len(df_unique),
            "split_counts": {name: len(ids) for name, ids in splits.items()},
            "ratios": {
                "train": TRAIN_RATIO,
                "val": VAL_RATIO,
                "test": TEST_RATIO,
            },
            "num_perm": MINHASH_NUM_PERM,
            "shingle_size": NGRAM_SIZE,
            "similarity_threshold": MINHASH_THRESHOLD,
            "unique_articles": len(df_unique),
            "near_duplicate_groups": len(group_sizes),
            "articles_in_multi_member_groups": len(df_unique) - len(group_sizes),
            "counts": {name: len(ids) for name, ids in splits.items()},
        },
        "train": sorted(splits["train"]),
        "val": sorted(splits["val"]),
        "test": sorted(splits["test"]),
        "splits": {name: sorted(ids) for name, ids in splits.items()},
    }
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def load_splits(path: Path = SPLITS_FILE) -> dict[str, list[str]]:
    """Load split IDs from either the flat or nested split-file layout."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if "splits" in data:
        return data["splits"]
    return {name: data[name] for name in ("train", "val", "test")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedup & split dataset NER")
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan statistik tanpa menyimpan file")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
