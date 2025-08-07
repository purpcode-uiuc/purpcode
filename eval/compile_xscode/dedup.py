#!/usr/bin/env python

# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0


import hashlib
import json
import struct
import time
from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Union

import fire
import numpy as np
from datasets import Dataset, load_dataset
from scipy.integrate import quad as integrate
from tqdm import tqdm

SEED = 42
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)


def ngrams(sequence, n, min_length):
    if len(sequence) < min_length:
        return []
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def embed_func(messages, idx, **kwargs):
    content = messages[0]["content"]

    hashvalues = np.ones(kwargs["num_perm"], dtype=np.uint64) * MAX_HASH
    tokens = {
        " ".join(t)
        for t in ngrams(content.split(), kwargs["ngram"], kwargs["min_length"])
    }

    hv = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )
    a, b = kwargs["permutations"]
    phv = np.bitwise_and(
        ((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)

    Hs = [
        bytes(hashvalues[start:end].byteswap().data)
        for start, end in kwargs["hashranges"]
    ]
    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(threshold, num_perm):
    def false_positive_probability(threshold, b, r):
        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold, b, r):
        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp + fn
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


def deduplicate(
    generation_path: str,
    messages_column="messages",
    ngram=3,
    num_perm=250,
    threshold=0.3,
    min_length=2,
    batch_size=10000,
    output="deduplicated",
    save_duplicates=True,
):
    start_time = time.time()
    with open(generation_path, "r") as f:
        ds = [json.loads(line) for line in f]

    ds = Dataset.from_dict({key: [d[key] for d in ds] for key in ds[0]})
    print(f"Loaded {len(ds)} examples")

    B, R = optimal_param(threshold, num_perm)
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES = [defaultdict(set) for _ in range(B)]

    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_perm)
        ],
        dtype=np.uint64,
    ).T

    embedded = ds.map(
        embed_func,
        fn_kwargs={
            "num_perm": num_perm,
            "hashranges": HASH_RANGES,
            "ngram": ngram,
            "permutations": PERMUTATIONS,
            "min_length": min_length,
        },
        input_columns=[messages_column],
        with_indices=True,
        batch_size=batch_size,
    )

    uf = UnionFind()

    # Process in batches
    for i in tqdm(range(0, len(embedded), batch_size), desc="Processing batches"):
        batch = embedded[i : i + batch_size]
        for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
            for H, hashtable in zip(Hs, HASH_TABLES):
                hashtable[H].add(key)

    for table in HASH_TABLES:
        for cluster in table.values():
            if len(cluster) > 1:
                idx = min(cluster)
                for x in cluster:
                    uf.union(x, idx)

    embedded = embedded.map(
        lambda _, idx: {"__cluster__": uf.find(idx)},
        with_indices=True,
        batch_size=batch_size,
    )

    # Count unique clusters and find cluster representatives
    cluster_counts = {}
    cluster_representatives = {}
    for record_idx, record in enumerate(embedded):
        cluster_id = record["__cluster__"]
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        # The representative is the one with the minimum original index (cluster_id)
        if cluster_id not in cluster_representatives:
            cluster_representatives[cluster_id] = cluster_id

    clusters_with_duplicates = sum(1 for count in cluster_counts.values() if count > 1)
    total_duplicates_removed = sum(
        count - 1 for count in cluster_counts.values() if count > 1
    )

    # Separate duplicates and kept data
    kept_data = embedded.filter(
        lambda record, idx: record["__cluster__"] == idx,
        with_indices=True,
        batch_size=batch_size,
    )

    duplicate_data = None
    if save_duplicates and total_duplicates_removed > 0:
        duplicate_data = embedded.filter(
            lambda record, idx: record["__cluster__"] != idx,
            with_indices=True,
            batch_size=batch_size,
        )

        # Add similarity information to duplicates
        duplicate_data = duplicate_data.map(
            lambda record: {
                **record,
                "__similar_to_idx__": record["__cluster__"],
                "__duplicate_cluster__": record["__cluster__"],
            },
            batch_size=batch_size,
        )

    kept_data = kept_data.remove_columns(["__cluster__", "__signatures__", "__id__"])

    output_path = Path(output)
    if output_path.suffix != ".jsonl":
        # Include threshold in filename
        stem = output_path.stem
        output_path = output_path.parent / f"{stem}_threshold_{threshold}.jsonl"

    # Save kept data
    with open(output_path, "w") as f:
        for item in kept_data:
            f.write(json.dumps(item) + "\n")

    # Save duplicates if requested
    duplicates_saved = 0
    if save_duplicates and total_duplicates_removed > 0 and duplicate_data is not None:
        duplicate_data = duplicate_data.remove_columns(
            ["__cluster__", "__signatures__", "__id__"]
        )

        duplicate_path = output_path.with_name(
            f"{output_path.stem}_duplicates{output_path.suffix}"
        )
        with open(duplicate_path, "w") as f:
            for item in duplicate_data:
                f.write(json.dumps(item) + "\n")
        duplicates_saved = len(duplicate_data)
        print(f"Duplicates saved: {duplicate_path}")

    original_count = len(ds)
    final_count = len(kept_data)

    print(f"=== DEDUPLICATION STATS ===")
    print(
        f"Parameters: ngram={ngram}, num_perm={num_perm}, threshold={threshold}, min_length={min_length}"
    )
    print(f"Original samples: {original_count:,}")
    print(f"Duplicate clusters found: {clusters_with_duplicates:,}")
    print(f"Duplicate samples removed: {total_duplicates_removed:,}")
    print(f"Final samples kept: {final_count:,}")
    print(f"Retention rate: {final_count/original_count:.2%}")
    print(f"Duplicate rate: {total_duplicates_removed/original_count:.2%}")
    print(f"Processing time: {time.time() - start_time:.2f}s")
    print(f"Output saved: {output_path}")
    if save_duplicates and total_duplicates_removed > 0:
        print(f"Duplicates saved: {duplicates_saved:,} samples with similarity info")
    print(f"==========================")


def dedup_main(generation_path: str = "xscode.jsonl"):

    deduplication_path = generation_path.replace(".jsonl", ".deduplicated.jsonl")

    if Path(deduplication_path).exists():
        print(f"Deduplicated file already exists: {deduplication_path}")
        return deduplication_path

    deduplicate(
        generation_path=generation_path,
        messages_column="messages",
        ngram=3,
        num_perm=250,
        threshold=0.4,
        min_length=2,
        batch_size=10000,
        output=deduplication_path,
        save_duplicates=True,
    )

    return deduplication_path


if __name__ == "__main__":
    fire.Fire(dedup_main)
