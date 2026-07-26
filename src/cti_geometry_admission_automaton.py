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


def generate_sealed_key(key_index: int) -> tuple[dict, bytes, str]:
    seed_bytes = os.urandom(32)
    seed_hash = hashlib.sha256(seed_bytes).hexdigest()
    key_json = generate_key_from_seed(seed_bytes)
    return key_json, seed_bytes, seed_hash


def paired_key_from_transposition(
    base_key_json: dict,
    calibrated_op: str,
    withheld_op: str,
    source_u: int,
    source_v: int,
) -> tuple[dict, dict]:
    """Construct a CM-CKS partner key by transposing two outputs of one withheld permutation.

    Returns (partner_key_json, pair_metadata).
    """
    assert withheld_op in OP_NAMES, f"Unknown op: {withheld_op}"
    assert calibrated_op in OP_NAMES, f"Unknown op: {calibrated_op}"
    assert withheld_op != calibrated_op, "Cannot transpose the calibrated operator"
    assert 0 <= source_u < NUM_STATES and 0 <= source_v < NUM_STATES
    assert source_u != source_v, "Transposition requires distinct states"

    partner = {}
    for op in OP_NAMES:
        partner[op] = list(base_key_json[op])

    perm = partner[withheld_op]
    perm[source_u], perm[source_v] = perm[source_v], perm[source_u]

    for op in OP_NAMES:
        assert len(set(partner[op])) == NUM_STATES, f"Not a permutation: {op}"
    assert partner[calibrated_op] == base_key_json[calibrated_op]

    op_idx = OP_NAMES.index(withheld_op)
    changed_edges = [
        {"op": withheld_op, "state": source_u,
         "edge_index": source_u * NUM_OPS + op_idx,
         "base_output": base_key_json[withheld_op][source_u],
         "partner_output": partner[withheld_op][source_u]},
        {"op": withheld_op, "state": source_v,
         "edge_index": source_v * NUM_OPS + op_idx,
         "base_output": base_key_json[withheld_op][source_v],
         "partner_output": partner[withheld_op][source_v]},
    ]

    base_hash = sha256_hex(json.dumps(base_key_json, sort_keys=True))
    partner_hash = sha256_hex(json.dumps(partner, sort_keys=True))

    metadata = {
        "calibrated_op": calibrated_op,
        "withheld_op": withheld_op,
        "transposition": [source_u, source_v],
        "changed_edges": changed_edges,
        "base_key_hash": base_hash,
        "partner_key_hash": partner_hash,
        "calibration_identical": base_key_json[calibrated_op] == partner[calibrated_op],
        "num_differing_entries": sum(
            1 for op in OP_NAMES
            for s in range(NUM_STATES)
            if base_key_json[op][s] != partner[op][s]
        ),
    }
    assert metadata["calibration_identical"]
    assert metadata["num_differing_entries"] == 2

    return partner, metadata


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
    has_labels = "label" in batch[0]
    labels = [] if has_labels else None
    for b in batch:
        ids = b["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [PAD_TOKEN] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        if has_labels:
            labels.append(b["label"])
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    if has_labels:
        result["labels"] = torch.tensor(labels, dtype=torch.long)
    return result


ANCHOR_COUNT = 2048
ANCHOR_LENGTH_RANGE = (8, 32)
ANCHOR_PROTOCOL_ID = "OCF_GAT_ANCHORS_R7_V2"


def generate_anchors(
    n_anchors: int = ANCHOR_COUNT,
    length_range: tuple[int, int] = ANCHOR_LENGTH_RANGE,
    protocol_id: str = ANCHOR_PROTOCOL_ID,
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


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def apply_permutation_power(key: np.ndarray, op: int, state: int, power: int) -> int:
    """Apply pi_op^power to state. power >= 0."""
    s = state
    for _ in range(power):
        s = int(key[op, s])
    return s


def apply_inverse_permutation_power(key: np.ndarray, op: int, state: int, power: int) -> int:
    """Apply pi_op^{-power} to state."""
    inv_perm = inverse_permutation(key[op])
    s = state
    for _ in range(power):
        s = int(inv_perm[s])
    return s


def generate_calibration_set(key: np.ndarray, key_slot: int) -> list[dict]:
    """Generate the 64 labeled calibration sequences for a key.

    key_slot determines which operation is labeled: q_k = key_slot % 4.
    All operations in all 64 sequences use only the labeled operation.
    """
    q_k = key_slot % NUM_OPS
    examples = []

    for L in [1, 2, 4, 8, 16]:
        for s0 in range(NUM_STATES):
            ops = [q_k] * L
            label = simulate_automaton(key, s0, np.array(ops))
            input_ids = [s0] + [op + 12 for op in ops]
            examples.append({
                "input_ids": input_ids,
                "label": label,
                "s0": s0,
                "ops": ops,
                "length": L,
            })

    extra = [(0, 3), (3, 5), (6, 7), (9, 11)]
    for s0, L in extra:
        ops = [q_k] * L
        label = simulate_automaton(key, s0, np.array(ops))
        input_ids = [s0] + [op + 12 for op in ops]
        examples.append({
            "input_ids": input_ids,
            "label": label,
            "s0": s0,
            "ops": ops,
            "length": L,
        })

    assert len(examples) == 64
    return examples


def generate_withheld_eval_set(
    key: np.ndarray,
    key_slot: int,
    key_index: int,
) -> tuple[list[dict], list[dict]]:
    """Generate the 4000 withheld evaluation examples and 36 direct withheld probes.

    Returns (withheld_sequences, direct_probes).
    """
    q_k = key_slot % NUM_OPS

    withheld_edges = []
    for s in range(NUM_STATES):
        for x in range(NUM_OPS):
            if x != q_k:
                withheld_edges.append((s, x))
    assert len(withheld_edges) == 36

    edge_hash = lambda e: sha256_hex(f"GAT_WITHHELD_EXTRA_V1||{key_index}||{e[0]}_{e[1]}")
    sorted_edges = sorted(withheld_edges, key=edge_hash)

    allocation = {}
    for e in withheld_edges:
        allocation[e] = 111
    for i in range(4):
        allocation[sorted_edges[i]] += 1

    examples = []
    for edge in withheld_edges:
        s_target, x_target = edge
        n_examples = allocation[edge]

        pair_seed_str = f"GAT_WITHHELD_PAIRS_V1||{key_index}||{s_target}_{x_target}"
        pair_seed = int(sha256_hex(pair_seed_str)[:16], 16)
        rng = np.random.default_rng(pair_seed)

        all_pairs = [(p, r) for p in range(16) for r in range(16)]
        rng.shuffle(all_pairs)

        for idx in range(n_examples):
            p, r = all_pairs[idx % len(all_pairs)]

            s0 = apply_inverse_permutation_power(key, q_k, s_target, p)

            ops = [q_k] * p + [x_target] + [q_k] * r
            label = simulate_automaton(key, s0, np.array(ops))
            input_ids = [s0] + [op + 12 for op in ops]

            examples.append({
                "input_ids": input_ids,
                "label": label,
                "s0": s0,
                "ops": ops,
                "length": len(ops),
                "target_edge": (s_target, x_target),
                "p": p,
                "r": r,
            })

    assert len(examples) == 4000

    direct_probes = []
    for s, x in withheld_edges:
        label = int(key[x, s])
        input_ids = [s, x + 12]
        direct_probes.append({
            "input_ids": input_ids,
            "label": label,
            "s0": s,
            "ops": [x],
            "length": 1,
            "target_edge": (s, x),
        })
    assert len(direct_probes) == 36

    return examples, direct_probes


def generate_stage_b_dev_keys() -> list[dict]:
    """Generate the two Stage B development keys."""
    keys = []
    for i in range(2):
        seed_str = f"GAT_STAGE_B_DEV_KEY_V1|{i}"
        seed_bytes = hashlib.sha256(seed_str.encode()).digest()
        key_json = generate_key_from_seed(seed_bytes)
        keys.append({
            "key_json": key_json,
            "seed_hash": hashlib.sha256(seed_bytes).hexdigest(),
            "derivation": seed_str,
            "slot": i,
        })
    return keys


def generate_bank_order_permutation(n_banks: int = 32, n_steps: int = 5000) -> list[int]:
    """Generate frozen bank order for installer training."""
    seed_hex = sha256_hex("GAT_INSTALLER_BANK_ORDER_V1")
    seed_int = int(seed_hex[:16], 16)
    rng = np.random.default_rng(seed_int)

    perm = rng.permutation(n_banks).tolist()
    order = []
    for step in range(n_steps):
        order.append(perm[step % n_banks])
    return order


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
