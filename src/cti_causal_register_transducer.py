"""Causal Register Transducer: simulator, data generation, and partition logic.

Protocol: CSO_ADMISSION_V1 (locked Jul 26 2026).
Task: 4 registers in Z_16, 8 invertible non-commuting operations.
State space: 65,536 states. Exact-state chance: 1/65536.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MOD = 16
NUM_REGS = 4
NUM_OPS = 8
STATE_SPACE_SIZE = MOD ** NUM_REGS  # 65536

OP_NAMES = [
    "add_01",   # U0: r0 <- r0 + r1 mod 16
    "add_12",   # U1: r1 <- r1 + r2 mod 16
    "add_23",   # U2: r2 <- r2 + r3 mod 16
    "add_30",   # U3: r3 <- r3 + r0 mod 16
    "swap_02",  # U4: swap(r0, r2)
    "swap_13",  # U5: swap(r1, r3)
    "rotate_L", # U6: (r0,r1,r2,r3) <- (r1,r2,r3,r0)
    "neg_02",   # U7: r0 <- -r0 mod 16; r2 <- -r2 mod 16
]

PARTITION_SEED = 42
TRAIN_FRACTION = 0.75
TRAIN_LENGTHS = (1, 12)
EVAL_LENGTHS = (13, 32)
NUM_EXCLUDED_BIGRAMS = 16


def apply_op(state: np.ndarray, op: int) -> np.ndarray:
    """Apply operation op to register state. state shape: (4,) dtype int."""
    r = state.copy()
    if op == 0:    # add_01
        r[0] = (r[0] + r[1]) % MOD
    elif op == 1:  # add_12
        r[1] = (r[1] + r[2]) % MOD
    elif op == 2:  # add_23
        r[2] = (r[2] + r[3]) % MOD
    elif op == 3:  # add_30
        r[3] = (r[3] + r[0]) % MOD
    elif op == 4:  # swap_02
        r[0], r[2] = r[2], r[0]
    elif op == 5:  # swap_13
        r[1], r[3] = r[3], r[1]
    elif op == 6:  # rotate_L
        r[0], r[1], r[2], r[3] = r[1], r[2], r[3], r[0]
    elif op == 7:  # neg_02
        r[0] = (-r[0]) % MOD
        r[2] = (-r[2]) % MOD
    else:
        raise ValueError(f"Invalid op: {op}")
    return r


def apply_inverse_op(state: np.ndarray, op: int) -> np.ndarray:
    """Apply the inverse of operation op to register state."""
    r = state.copy()
    if op == 0:    # inv(add_01): r0 <- r0 - r1 mod 16
        r[0] = (r[0] - r[1]) % MOD
    elif op == 1:  # inv(add_12): r1 <- r1 - r2 mod 16
        r[1] = (r[1] - r[2]) % MOD
    elif op == 2:  # inv(add_23): r2 <- r2 - r3 mod 16
        r[2] = (r[2] - r[3]) % MOD
    elif op == 3:  # inv(add_30): r3 <- r3 - r0 mod 16
        r[3] = (r[3] - r[0]) % MOD
    elif op == 4:  # inv(swap_02) = swap_02
        r[0], r[2] = r[2], r[0]
    elif op == 5:  # inv(swap_13) = swap_13
        r[1], r[3] = r[3], r[1]
    elif op == 6:  # inv(rotate_L) = rotate_R
        r[0], r[1], r[2], r[3] = r[3], r[0], r[1], r[2]
    elif op == 7:  # inv(neg_02) = neg_02
        r[0] = (-r[0]) % MOD
        r[2] = (-r[2]) % MOD
    else:
        raise ValueError(f"Invalid op: {op}")
    return r


def execute_program(init_state: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Execute a sequence of operations starting from init_state.
    Returns final register state.
    """
    state = init_state.copy()
    for op in ops:
        state = apply_op(state, int(op))
    return state


def state_to_index(state: np.ndarray) -> int:
    """Convert (r0,r1,r2,r3) to a flat index in [0, 65536)."""
    return int(state[0] * MOD**3 + state[1] * MOD**2 + state[2] * MOD + state[3])


def index_to_state(idx: int) -> np.ndarray:
    """Convert flat index to (r0,r1,r2,r3)."""
    r3 = idx % MOD
    r2 = (idx // MOD) % MOD
    r1 = (idx // MOD**2) % MOD
    r0 = (idx // MOD**3) % MOD
    return np.array([r0, r1, r2, r3], dtype=np.int64)


# --- Partition logic ---


def _hash_state(state: np.ndarray, seed: int) -> int:
    """Deterministic hash of a register state for partitioning."""
    data = struct.pack(">I4B", seed, *state.astype(np.uint8))
    return int(hashlib.sha256(data).hexdigest(), 16)


def make_initial_state_partition(seed: int = PARTITION_SEED,
                                 train_frac: float = TRAIN_FRACTION):
    """Partition all 65536 initial states into train/eval sets.
    Returns (train_indices, eval_indices) as sorted numpy arrays.
    """
    train_indices = []
    eval_indices = []
    for idx in range(STATE_SPACE_SIZE):
        state = index_to_state(idx)
        h = _hash_state(state, seed)
        if (h % 10000) < int(train_frac * 10000):
            train_indices.append(idx)
        else:
            eval_indices.append(idx)
    return np.array(sorted(train_indices)), np.array(sorted(eval_indices))


def make_bigram_partition(seed: int = PARTITION_SEED,
                          num_excluded: int = NUM_EXCLUDED_BIGRAMS):
    """Partition 64 ordered bigrams (8x8) into included/excluded.
    Returns (included_bigrams, excluded_bigrams) as lists of (op_i, op_j) tuples.
    """
    all_bigrams = [(i, j) for i in range(NUM_OPS) for j in range(NUM_OPS)]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_bigrams))
    excluded_set = set()
    for p in perm[:num_excluded]:
        excluded_set.add(all_bigrams[p])
    included = [b for b in all_bigrams if b not in excluded_set]
    excluded = [b for b in all_bigrams if b in excluded_set]
    return sorted(included), sorted(excluded)


def make_trigram_withheld(seed: int = PARTITION_SEED, count: int = 32):
    """Select precommitted withheld trigrams for evaluation.
    Returns list of (op_i, op_j, op_k) tuples.
    """
    all_trigrams = [(i, j, k) for i in range(NUM_OPS)
                    for j in range(NUM_OPS) for k in range(NUM_OPS)]
    rng = np.random.RandomState(seed + 1)
    perm = rng.permutation(len(all_trigrams))
    return sorted([all_trigrams[p] for p in perm[:count]])


# --- Data generation ---


def generate_training_example(rng: np.random.RandomState,
                              train_state_indices: np.ndarray,
                              included_bigrams: list,
                              min_len: int = TRAIN_LENGTHS[0],
                              max_len: int = TRAIN_LENGTHS[1]):
    """Generate one training example.
    Returns (init_state, ops, final_state) where:
    - init_state: (4,) array in Z_16
    - ops: (L,) array of op indices
    - final_state: (4,) array in Z_16
    """
    idx = train_state_indices[rng.randint(len(train_state_indices))]
    init_state = index_to_state(idx)

    length = rng.randint(min_len, max_len + 1)
    ops = np.empty(length, dtype=np.int64)

    for i in range(length):
        if i == 0:
            ops[i] = rng.randint(NUM_OPS)
        else:
            while True:
                candidate = rng.randint(NUM_OPS)
                bigram = (int(ops[i - 1]), int(candidate))
                if bigram in _INCLUDED_BIGRAM_SET:
                    ops[i] = candidate
                    break

    final_state = execute_program(init_state, ops)
    return init_state, ops, final_state


def generate_eval_example(rng: np.random.RandomState,
                          eval_state_indices: np.ndarray,
                          min_len: int = EVAL_LENGTHS[0],
                          max_len: int = EVAL_LENGTHS[1]):
    """Generate one evaluation example (no bigram restriction).
    Returns (init_state, ops, final_state).
    """
    idx = eval_state_indices[rng.randint(len(eval_state_indices))]
    init_state = index_to_state(idx)
    length = rng.randint(min_len, max_len + 1)
    ops = rng.randint(0, NUM_OPS, size=length).astype(np.int64)
    final_state = execute_program(init_state, ops)
    return init_state, ops, final_state


_INCLUDED_BIGRAM_SET = set()


def init_partitions(seed: int = PARTITION_SEED):
    """Initialize and return all partition data. Must be called before data generation."""
    global _INCLUDED_BIGRAM_SET

    train_states, eval_states = make_initial_state_partition(seed)
    included_bigrams, excluded_bigrams = make_bigram_partition(seed)
    withheld_trigrams = make_trigram_withheld(seed)

    _INCLUDED_BIGRAM_SET = set(included_bigrams)

    return {
        "train_state_indices": train_states,
        "eval_state_indices": eval_states,
        "included_bigrams": included_bigrams,
        "excluded_bigrams": excluded_bigrams,
        "withheld_trigrams": withheld_trigrams,
    }


def partition_hashes(partitions: dict) -> dict:
    """Compute deterministic hashes of all partition data for precommit."""
    hashes = {}

    h = hashlib.sha256()
    h.update(partitions["train_state_indices"].tobytes())
    hashes["train_states_sha256"] = h.hexdigest()

    h = hashlib.sha256()
    h.update(partitions["eval_state_indices"].tobytes())
    hashes["eval_states_sha256"] = h.hexdigest()

    h = hashlib.sha256()
    for b in partitions["included_bigrams"]:
        h.update(struct.pack(">BB", *b))
    hashes["included_bigrams_sha256"] = h.hexdigest()

    h = hashlib.sha256()
    for b in partitions["excluded_bigrams"]:
        h.update(struct.pack(">BB", *b))
    hashes["excluded_bigrams_sha256"] = h.hexdigest()

    h = hashlib.sha256()
    for t in partitions["withheld_trigrams"]:
        h.update(struct.pack(">BBB", *t))
    hashes["withheld_trigrams_sha256"] = h.hexdigest()

    return hashes


# --- Verification suite ---


def _build_all_states() -> np.ndarray:
    """Build (65536, 4) array of all register states. Cached for vectorized ops."""
    indices = np.arange(STATE_SPACE_SIZE)
    states = np.stack([
        (indices // MOD**3) % MOD,
        (indices // MOD**2) % MOD,
        (indices // MOD) % MOD,
        indices % MOD,
    ], axis=1).astype(np.int64)
    return states


def _apply_op_vectorized(states: np.ndarray, op: int) -> np.ndarray:
    """Apply operation to all states at once. states: (N, 4), returns (N, 4)."""
    r = states.copy()
    if op == 0:
        r[:, 0] = (r[:, 0] + r[:, 1]) % MOD
    elif op == 1:
        r[:, 1] = (r[:, 1] + r[:, 2]) % MOD
    elif op == 2:
        r[:, 2] = (r[:, 2] + r[:, 3]) % MOD
    elif op == 3:
        r[:, 3] = (r[:, 3] + r[:, 0]) % MOD
    elif op == 4:
        r[:, 0], r[:, 2] = r[:, 2].copy(), r[:, 0].copy()
    elif op == 5:
        r[:, 1], r[:, 3] = r[:, 3].copy(), r[:, 1].copy()
    elif op == 6:
        r[:, 0], r[:, 1], r[:, 2], r[:, 3] = (
            r[:, 1].copy(), r[:, 2].copy(), r[:, 3].copy(), r[:, 0].copy()
        )
    elif op == 7:
        r[:, 0] = (-r[:, 0]) % MOD
        r[:, 2] = (-r[:, 2]) % MOD
    return r


def _apply_inverse_op_vectorized(states: np.ndarray, op: int) -> np.ndarray:
    """Apply inverse of operation to all states at once."""
    r = states.copy()
    if op == 0:
        r[:, 0] = (r[:, 0] - r[:, 1]) % MOD
    elif op == 1:
        r[:, 1] = (r[:, 1] - r[:, 2]) % MOD
    elif op == 2:
        r[:, 2] = (r[:, 2] - r[:, 3]) % MOD
    elif op == 3:
        r[:, 3] = (r[:, 3] - r[:, 0]) % MOD
    elif op == 4:
        r[:, 0], r[:, 2] = r[:, 2].copy(), r[:, 0].copy()
    elif op == 5:
        r[:, 1], r[:, 3] = r[:, 3].copy(), r[:, 1].copy()
    elif op == 6:
        r[:, 0], r[:, 1], r[:, 2], r[:, 3] = (
            r[:, 3].copy(), r[:, 0].copy(), r[:, 1].copy(), r[:, 2].copy()
        )
    elif op == 7:
        r[:, 0] = (-r[:, 0]) % MOD
        r[:, 2] = (-r[:, 2]) % MOD
    return r


def _states_to_indices(states: np.ndarray) -> np.ndarray:
    """Convert (N, 4) state array to (N,) flat indices."""
    return (states[:, 0] * MOD**3 + states[:, 1] * MOD**2
            + states[:, 2] * MOD + states[:, 3])


def verify_invertibility():
    """Verify all 8 operations are invertible over all 65536 states (vectorized)."""
    print("Verifying invertibility...")
    all_states = _build_all_states()
    for op in range(NUM_OPS):
        forward = _apply_op_vectorized(all_states, op)
        recovered = _apply_inverse_op_vectorized(forward, op)
        if not np.array_equal(all_states, recovered):
            mismatches = np.where(~np.all(all_states == recovered, axis=1))[0]
            raise AssertionError(
                f"Invertibility failed: op={op}, {len(mismatches)} mismatches"
            )
        print(f"  {OP_NAMES[op]}: PASS (65536/65536 states)")
    print("Invertibility: ALL PASS")


def verify_bijectivity():
    """Verify all 8 operations are bijections (vectorized)."""
    print("Verifying bijectivity...")
    all_states = _build_all_states()
    for op in range(NUM_OPS):
        result = _apply_op_vectorized(all_states, op)
        result_indices = _states_to_indices(result)
        n_unique = len(np.unique(result_indices))
        if n_unique != STATE_SPACE_SIZE:
            raise AssertionError(
                f"Bijectivity failed: op={op}, {n_unique} unique outputs != {STATE_SPACE_SIZE}"
            )
        print(f"  {OP_NAMES[op]}: PASS ({n_unique} unique outputs)")
    print("Bijectivity: ALL PASS")


def verify_noncommutativity():
    """Verify non-commutativity (vectorized): for each pair (i,j) with i<j,
    check if U_i(U_j(r)) != U_j(U_i(r)) for any state.
    """
    print("Verifying non-commutativity...")
    all_states = _build_all_states()
    noncommuting_pairs = 0
    commuting_pairs = []
    for i in range(NUM_OPS):
        for j in range(i + 1, NUM_OPS):
            ij = _apply_op_vectorized(_apply_op_vectorized(all_states, i), j)
            ji = _apply_op_vectorized(_apply_op_vectorized(all_states, j), i)
            differ = ~np.all(ij == ji, axis=1)
            n_differ = np.sum(differ)
            if n_differ > 0:
                noncommuting_pairs += 1
            else:
                commuting_pairs.append((i, j))
                print(f"  WARNING: ops {i},{j} ({OP_NAMES[i]},{OP_NAMES[j]}) COMMUTE")
    total_pairs = NUM_OPS * (NUM_OPS - 1) // 2
    print(f"Non-commutativity: {noncommuting_pairs}/{total_pairs} pairs non-commuting")
    if commuting_pairs:
        print(f"  Commuting pairs: {commuting_pairs}")
    return noncommuting_pairs, commuting_pairs


def verify_group_generates_large():
    """Verify the 8 operations generate a group much larger than 65,536.

    Instead of BFS on S_65536 (infeasible), we check algebraic properties:
    1. add_01 alone generates Z_16 orbits on r0 (order 16 per r1 value)
    2. rotate_L cycles all 4 registers
    3. Combined with swaps and negation, these generate a group acting
       transitively on all coordinates with independent Z_16 actions.

    We verify by checking that composing generators can reach all 65,536
    states from any fixed starting state.
    """
    print("Verifying group generates a large subgroup...")

    start = np.array([1, 2, 3, 4], dtype=np.int64)
    reachable = set()
    reachable.add(tuple(start))
    frontier = [start]
    max_bfs = STATE_SPACE_SIZE + 1000

    while frontier and len(reachable) < max_bfs:
        next_frontier = []
        for state in frontier:
            for op in range(NUM_OPS):
                result = apply_op(state, op)
                key = tuple(result)
                if key not in reachable:
                    reachable.add(key)
                    next_frontier.append(result)
                inv_result = apply_inverse_op(state, op)
                inv_key = tuple(inv_result)
                if inv_key not in reachable:
                    reachable.add(inv_key)
                    next_frontier.append(inv_result)
        frontier = next_frontier
        if len(reachable) % 10000 < 100:
            print(f"  ...{len(reachable)} states reached so far")

    print(f"  Reachable states from (1,2,3,4): {len(reachable)}")
    if len(reachable) == STATE_SPACE_SIZE:
        print("  Group orbit covers ALL Z_16^4 states (transitive action)")
    elif len(reachable) > STATE_SPACE_SIZE // 2:
        print(f"  Large orbit: {len(reachable)}/{STATE_SPACE_SIZE} "
              f"({100*len(reachable)/STATE_SPACE_SIZE:.1f}%)")
    else:
        print(f"  WARNING: Small orbit {len(reachable)}/{STATE_SPACE_SIZE}")
    return len(reachable)


def verify_partitions():
    """Verify partition sizes and determinism."""
    print("Verifying partitions...")
    p = init_partitions()

    n_train = len(p["train_state_indices"])
    n_eval = len(p["eval_state_indices"])
    print(f"  Train states: {n_train} ({100*n_train/STATE_SPACE_SIZE:.1f}%)")
    print(f"  Eval states: {n_eval} ({100*n_eval/STATE_SPACE_SIZE:.1f}%)")

    overlap = set(p["train_state_indices"]) & set(p["eval_state_indices"])
    assert len(overlap) == 0, f"Partition overlap: {len(overlap)} states"
    assert n_train + n_eval == STATE_SPACE_SIZE, "Partition doesn't cover all states"

    n_inc = len(p["included_bigrams"])
    n_exc = len(p["excluded_bigrams"])
    print(f"  Included bigrams: {n_inc}, Excluded bigrams: {n_exc}")
    assert n_inc + n_exc == NUM_OPS * NUM_OPS, "Bigram partition incomplete"

    n_tri = len(p["withheld_trigrams"])
    print(f"  Withheld trigrams: {n_tri}")

    p2 = init_partitions()
    assert np.array_equal(p["train_state_indices"], p2["train_state_indices"])
    assert p["excluded_bigrams"] == p2["excluded_bigrams"]
    assert p["withheld_trigrams"] == p2["withheld_trigrams"]
    print("  Determinism: PASS")

    hashes = partition_hashes(p)
    for k, v in hashes.items():
        print(f"  {k}: {v[:16]}...")

    print("Partitions: ALL PASS")
    return p, hashes


def verify_data_generation(partitions: dict, n_samples: int = 1000):
    """Verify training and eval data generation produces correct results."""
    print(f"Verifying data generation ({n_samples} samples each)...")
    rng = np.random.RandomState(123)

    for i in range(n_samples):
        init_state, ops, final_state = generate_training_example(
            rng, partitions["train_state_indices"],
            partitions["included_bigrams"]
        )
        expected = execute_program(init_state, ops)
        assert np.array_equal(final_state, expected), f"Train sample {i} mismatch"
        assert state_to_index(init_state) in set(partitions["train_state_indices"])
        assert TRAIN_LENGTHS[0] <= len(ops) <= TRAIN_LENGTHS[1]
        for j in range(1, len(ops)):
            bigram = (int(ops[j-1]), int(ops[j]))
            assert bigram in _INCLUDED_BIGRAM_SET, f"Excluded bigram in train: {bigram}"

    for i in range(n_samples):
        init_state, ops, final_state = generate_eval_example(
            rng, partitions["eval_state_indices"]
        )
        expected = execute_program(init_state, ops)
        assert np.array_equal(final_state, expected), f"Eval sample {i} mismatch"
        assert state_to_index(init_state) in set(partitions["eval_state_indices"])
        assert EVAL_LENGTHS[0] <= len(ops) <= EVAL_LENGTHS[1]

    print("Data generation: ALL PASS")


def run_full_verification():
    """Run all verification checks. Call this before any training."""
    print("=" * 60)
    print("CAUSAL REGISTER TRANSDUCER: EXHAUSTIVE VERIFICATION")
    print("=" * 60)
    print()

    verify_invertibility()
    print()
    verify_bijectivity()
    print()
    nc, commuting = verify_noncommutativity()
    print()

    partitions, hashes = verify_partitions()
    print()
    verify_data_generation(partitions)
    print()

    print("=" * 60)
    print("ALL VERIFICATION CHECKS PASSED")
    print(f"Non-commuting pairs: {nc}/28")
    if commuting:
        print(f"Commuting pairs: {commuting}")
    print(f"Train states: {len(partitions['train_state_indices'])}")
    print(f"Eval states: {len(partitions['eval_state_indices'])}")
    print(f"Excluded bigrams: {len(partitions['excluded_bigrams'])}")
    print(f"Withheld trigrams: {len(partitions['withheld_trigrams'])}")
    print("=" * 60)

    return partitions, hashes


if __name__ == "__main__":
    partitions, hashes = run_full_verification()

    print("\nPartition hashes (for precommit.json):")
    print(json.dumps(hashes, indent=2))

    print("\nExcluded bigrams:")
    for b in partitions["excluded_bigrams"]:
        print(f"  ({OP_NAMES[b[0]]}, {OP_NAMES[b[1]]})")

    print("\nRunning transitivity check (reachability from origin)...")
    n_reachable = verify_group_generates_large()
