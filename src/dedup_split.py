import json
import re
from collections import defaultdict

import numpy as np
from datasketch import MinHash, MinHashLSH

from src.config import (
    NUM_PERM,
    SEED,
    SHINGLE_SIZE,
    SIMILARITY_THRESHOLD,
    SPLIT_RATIOS,
    SPLITS_DIR,
    SPLITS_PATH,
)
from src.data_loader import load_all, unique_articles


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def shingles(text, k=SHINGLE_SIZE):
    t = normalize(text)
    if len(t) <= k:
        return {t}
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def make_minhash(text, num_perm=NUM_PERM):
    m = MinHash(num_perm=num_perm)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m

class _UnionFind:

    def __init__(self, n):
        self.p = list(range(n))   # awalnya tiap elemen jadi grup sendiri

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_groups(texts, threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERM):
    mhs = [make_minhash(t, num_perm) for t in texts]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(mhs):
        lsh.insert(str(i), m)
    uf = _UnionFind(len(texts))
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            uf.union(i, int(j))
    return [uf.find(i) for i in range(len(texts))]

def split_groups(group_ids, ratios=SPLIT_RATIOS, seed=SEED):
    sizes = defaultdict(int)
    for g in group_ids:
        sizes[g] += 1

    total = sum(sizes.values())
    targets = {name: r * total for name, r in ratios.items()}

    order = sorted(sizes)                  # biar urutan awalnya pasti
    rng = np.random.default_rng(seed)
    rng.shuffle(order)                     # acak, tapi hasil acaknya selalu sama

    counts = {name: 0 for name in ratios}
    assign = {}
    for g in order:
        name = max(ratios, key=lambda k: (targets[k] - counts[k]) / targets[k])
        assign[g] = name
        counts[name] += sizes[g]
    return assign

def verify_no_leakage(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]) & set(splits[b])
            assert not overlap, f"KEBOCORAN {a} & {b}: {len(overlap)} id"


def run(df=None):
    df = load_all() if df is None else df
    uniq = unique_articles(df)
    groups = build_groups(uniq["body"].fillna("").tolist())
    assign = split_groups(groups)

    splits = {name: [] for name in SPLIT_RATIOS}
    for oid, g in zip(uniq["original_id"], groups):
        splits[assign[g]].append(oid)

    verify_no_leakage(splits)

    n_groups = len(set(groups))
    result = {
        "meta": {
            "seed": SEED,
            "ratios": SPLIT_RATIOS,
            "num_perm": NUM_PERM,
            "shingle_size": SHINGLE_SIZE,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "unique_articles": len(uniq),
            "near_duplicate_groups": n_groups,
            "articles_in_multi_member_groups": len(uniq) - n_groups,
            "counts": {name: len(ids) for name, ids in splits.items()},
        },
        "splits": splits,
    }
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_splits(path=SPLITS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["splits"]


if __name__ == "__main__":
    hasil = run()
    print(json.dumps(hasil["meta"], indent=2))
    print("Zero leakage OK. Tersimpan:", SPLITS_PATH)