"""Geometry Admission Test: automaton, key system, and data generation.

12-state, 4-symbol permutation automaton per the Stage A specification in
research/OPEN_CAPABILITY_FILE_GEOMETRY_ADMISSION_STAGE_A_2026_07_25.md
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

NUM_STATES = 12
NUM_OPS = 4
OP_NAMES = ["a", "b", "c", "d"]

STATE_TOKENS = list(range(NUM_STATES))       # 0..11
OP_TOKENS = list(range(12, 16))              # 12..15
PAD_TOKEN = 16
VOCAB_SIZE = 17

DEVELOPMENT_KEY_JSON = {
    "a": [1, 5, 4, 6, 3, 7, 8, 0, 9, 2, 10, 11],
    "b": [8, 10, 11, 5, 6, 1, 9, 3, 4, 0, 7, 2],
    "c": [11, 5, 1, 0, 6, 10, 2, 7, 9, 4, 8, 3],
    "d": [1, 7, 2, 4, 10, 5, 9, 0, 3, 8, 6, 11],
}


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key_from_json(key_json: dict) -> np.ndarray:
    perms = np.zeros((NUM_OPS, NUM_STATES), dtype=np.int64)
    for i, op in enumerate(OP_NAMES):
        perms[i] = key_json[op]
    return perms


def generate_key_from_seed(seed_bytes: bytes) -> dict:
    domain_seeds = []
    for i, op in enumerate(OP_NAMES):
        h = hashlib.sha256(seed_bytes + op.encode("utf-8")).digest()
        domain_seeds.append(int.from_bytes(h[:16], "little"))

    key_json = {}
    for i, op in enumerate(OP_NAMES):
        rng = np.random.Generator(np.random.PCG64DXSM(domain_seeds[i]))
        perm = rng.permutation(NUM_STATES).tolist()
        key_json[op] = perm
    return key_json


def generate_sealed_key(key_index: int) -> tuple[dict, str]:
    seed_bytes = os.urandom(32)
    seed_hash = hashlib.sha256(seed_bytes).hexdigest()
    key_json = generate_key_from_seed(seed_bytes)
    return key_json, seed_hash


def simulate_automaton(key: np.ndarray, s0: int, ops: np.ndarray) -> int:
    state = s0
    for op in ops:
        state = key[op, state]
    return int(state)


def simulate_batch(key: np.ndarray, s0: np.ndarray, ops: np.ndarray) -> np.ndarray:
    batch_size = s0.shape[0]
    states = s0.copy()
    seq_len = ops.shape[1]
    for t in range(seq_len):
        for i in range(batch_size):
            if ops[i, t] >= 0:
                states[i] = key[ops[i, t], states[i]]
    return states


def encode_example(s0: int, ops: list[int]) -> tuple[list[int], int]:
    input_ids = [s0] + [op + 12 for op in ops]
    return input_ids, len(input_ids)


def generate_eval_set(
    key: np.ndarray,
    n_examples: int,
    length_range: tuple[int, int],
    rng: np.random.Generator,
) -> list[dict]:
    examples = []
    for _ in range(n_examples):
        s0 = rng.integers(0, NUM_STATES)
        length = rng.integers(length_range[0], length_range[1] + 1)
        ops = rng.integers(0, NUM_OPS, size=length)
        label = simulate_automaton(key, int(s0), ops)
        input_ids, seq_len = encode_example(int(s0), ops.tolist())
        examples.append({
            "input_ids": input_ids,
            "label": label,
            "s0": int(s0),
            "ops": ops.tolist(),
            "length": int(length),
        })
    return examples


def generate_direct_edges(key: np.ndarray) -> list[dict]:
    examples = []
    for s in range(NUM_STATES):
        for op in range(NUM_OPS):
            label = int(key[op, s])
            input_ids = [s, op + 12]
            examples.append({
                "input_ids": input_ids,
                "label": label,
                "s0": s,
                "ops": [op],
                "length": 1,
            })
    return examples


def generate_all_eval_sets(key: np.ndarray, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "dev_in_range": generate_eval_set(key, 20000, (1, 16), rng),
        "dev_extrapolation": generate_eval_set(key, 20000, (17, 32), rng),
        "stress_long": generate_eval_set(key, 20000, (33, 64), rng),
        "direct_edges": generate_direct_edges(key),
    }


def hash_eval_set(examples: list[dict]) -> str:
    canonical = json.dumps(examples, sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical)


class AutomatonTrainDataset(IterableDataset):
    def __init__(self, key: np.ndarray, seed: int, max_length: int = 16):
        self.key = key
        self.seed = seed
        self.max_length = max_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = self.seed + worker_info.id
        else:
            seed = self.seed
        rng = np.random.default_rng(seed)

        while True:
            s0 = rng.integers(0, NUM_STATES)
            length = rng.integers(1, self.max_length + 1)
            ops = rng.integers(0, NUM_OPS, size=length)
            label = simulate_automaton(self.key, int(s0), ops)
            input_ids = [int(s0)] + [int(op) + 12 for op in ops]
            yield {
                "input_ids": input_ids,
                "label": label,
            }


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for b in batch:
        ids = b["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [PAD_TOKEN] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(b["label"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def generate_anchors(
    n_anchors: int = 2048,
    length_range: tuple[int, int] = (8, 24),
    protocol_id: str = "OCF_GAT_ANCHORS_V1",
) -> list[dict]:
    seed_hex = sha256_hex(protocol_id)
    seed_int = int(seed_hex[:16], 16)
    rng = np.random.default_rng(seed_int)

    anchors = []
    for _ in range(n_anchors):
        s0 = int(rng.integers(0, NUM_STATES))
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        ops = rng.integers(0, NUM_OPS, size=length).tolist()
        input_ids = [s0] + [op + 12 for op in ops]
        canonical = json.dumps(input_ids, separators=(",", ":"))
        anchor_hash = sha256_hex(protocol_id + "||" + canonical)
        anchors.append({
            "input_ids": input_ids,
            "s0": s0,
            "ops": ops,
            "length": length,
            "hash": anchor_hash,
        })
    return anchors


def partition_anchors_into_banks(
    anchors: list[dict],
    n_banks: int = 32,
) -> list[list[dict]]:
    sorted_anchors = sorted(anchors, key=lambda a: a["hash"])
    bank_size = len(sorted_anchors) // n_banks
    banks = []
    for i in range(n_banks):
        start = i * bank_size
        end = start + bank_size
        banks.append(sorted_anchors[start:end])
    return banks


def audit_edge_coverage(
    key: np.ndarray,
    anchors: list[dict],
) -> dict:
    edge_counts = np.zeros((NUM_STATES, NUM_OPS), dtype=np.int64)
    for a in anchors:
        state = a["s0"]
        for op in a["ops"]:
            edge_counts[state, op] += 1
            state = int(key[op, state])

    return {
        "edge_counts": edge_counts.tolist(),
        "min_count": int(edge_counts.min()),
        "max_count": int(edge_counts.max()),
        "total_traversals": int(edge_counts.sum()),
        "all_covered": bool(edge_counts.min() > 0),
        "min_above_400": bool(edge_counts.min() >= 400),
    }


if __name__ == "__main__":
    dev_key = key_from_json(DEVELOPMENT_KEY_JSON)
    print(f"Development key loaded: {dev_key.shape}")

    expected_hash = sha256_hex("GAT_STAGE_A_DEV_KEY_V1")
    print(f"Expected key derivation hash: {expected_hash}")

    for s in range(3):
        for op in range(NUM_OPS):
            result = simulate_automaton(dev_key, s, np.array([op]))
            print(f"  state={s}, op={OP_NAMES[op]} -> {result} (expected: {DEVELOPMENT_KEY_JSON[OP_NAMES[op]][s]})")

    eval_sets = generate_all_eval_sets(dev_key, seed=42)
    for name, examples in eval_sets.items():
        h = hash_eval_set(examples)
        print(f"  {name}: {len(examples)} examples, hash={h[:16]}...")

    anchors = generate_anchors()
    print(f"\nAnchors: {len(anchors)}")
    banks = partition_anchors_into_banks(anchors)
    print(f"Banks: {len(banks)} x {len(banks[0])}")

    coverage = audit_edge_coverage(dev_key, anchors)
    print(f"Edge coverage: min={coverage['min_count']}, max={coverage['max_count']}, all_covered={coverage['all_covered']}, min>=400={coverage['min_above_400']}")
