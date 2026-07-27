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
NUM_WITHHELD_TRIGRAMS = 32
GROUP_ORDER_MIN = 200_000


# ---------------------------------------------------------------------------
# Scalar operations
# ---------------------------------------------------------------------------

def apply_op(state: np.ndarray, op: int) -> np.ndarray:
    """Apply operation op to register state. state shape: (4,) dtype int."""
    r = state.copy()
    if op == 0:
        r[0] = (r[0] + r[1]) % MOD
    elif op == 1:
        r[1] = (r[1] + r[2]) % MOD
    elif op == 2:
        r[2] = (r[2] + r[3]) % MOD
    elif op == 3:
        r[3] = (r[3] + r[0]) % MOD
    elif op == 4:
        r[0], r[2] = r[2], r[0]
    elif op == 5:
        r[1], r[3] = r[3], r[1]
    elif op == 6:
        r[0], r[1], r[2], r[3] = r[1], r[2], r[3], r[0]
    elif op == 7:
        r[0] = (-r[0]) % MOD
        r[2] = (-r[2]) % MOD
    else:
        raise ValueError(f"Invalid op: {op}")
    return r


def apply_inverse_op(state: np.ndarray, op: int) -> np.ndarray:
    """Apply the inverse of operation op to register state."""
    r = state.copy()
    if op == 0:
        r[0] = (r[0] - r[1]) % MOD
    elif op == 1:
        r[1] = (r[1] - r[2]) % MOD
    elif op == 2:
        r[2] = (r[2] - r[3]) % MOD
    elif op == 3:
        r[3] = (r[3] - r[0]) % MOD
    elif op == 4:
        r[0], r[2] = r[2], r[0]
    elif op == 5:
        r[1], r[3] = r[3], r[1]
    elif op == 6:
        r[0], r[1], r[2], r[3] = r[3], r[0], r[1], r[2]
    elif op == 7:
        r[0] = (-r[0]) % MOD
        r[2] = (-r[2]) % MOD
    else:
        raise ValueError(f"Invalid op: {op}")
    return r


def execute_program(init_state: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Execute a sequence of operations starting from init_state."""
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


# ---------------------------------------------------------------------------
# Partition logic
# ---------------------------------------------------------------------------

def _hash_state(state: np.ndarray, seed: int) -> int:
    """Deterministic hash of a register state for partitioning."""
    data = struct.pack(">I4B", seed, *state.astype(np.uint8))
    return int(hashlib.sha256(data).hexdigest(), 16)


def make_initial_state_partition(seed: int = PARTITION_SEED,
                                 train_frac: float = TRAIN_FRACTION):
    """Partition all 65536 initial states into train/eval sets.
    Uses hash-based assignment (probabilistic, ~74.86% train).
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


def make_trigram_withheld(seed: int = PARTITION_SEED, count: int = NUM_WITHHELD_TRIGRAMS):
    """Select precommitted withheld trigrams for evaluation.
    Returns list of (op_i, op_j, op_k) tuples.
    """
    all_trigrams = [(i, j, k) for i in range(NUM_OPS)
                    for j in range(NUM_OPS) for k in range(NUM_OPS)]
    rng = np.random.RandomState(seed + 1)
    perm = rng.permutation(len(all_trigrams))
    return sorted([all_trigrams[p] for p in perm[:count]])


def init_partitions(seed: int = PARTITION_SEED):
    """Initialize and return all partition data."""
    train_states, eval_states = make_initial_state_partition(seed)
    included_bigrams, excluded_bigrams = make_bigram_partition(seed)
    withheld_trigrams = make_trigram_withheld(seed)

    return {
        "train_state_indices": train_states,
        "eval_state_indices": eval_states,
        "included_bigrams": included_bigrams,
        "excluded_bigrams": excluded_bigrams,
        "withheld_trigrams": withheld_trigrams,
        "included_bigram_set": frozenset(included_bigrams),
        "excluded_bigram_set": frozenset(excluded_bigrams),
        "withheld_trigram_set": frozenset(withheld_trigrams),
    }


def partition_hashes(partitions: dict) -> dict:
    """Compute deterministic hashes of all partition data for precommit.
    Uses explicit big-endian encoding for platform independence.
    """
    hashes = {}

    h = hashlib.sha256()
    for idx in partitions["train_state_indices"]:
        h.update(struct.pack(">I", int(idx)))
    hashes["train_states_sha256"] = h.hexdigest()

    h = hashlib.sha256()
    for idx in partitions["eval_state_indices"]:
        h.update(struct.pack(">I", int(idx)))
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


# ---------------------------------------------------------------------------
# Precommit
# ---------------------------------------------------------------------------

PRECOMMIT_PATH = Path(__file__).resolve().parent.parent / "results" / "causal_organ" / "precommit.json"


def create_precommit(partitions: dict, model_config: Optional[dict] = None) -> dict:
    """Create and write the immutable precommit artifact."""
    hashes = partition_hashes(partitions)

    precommit = {
        "protocol": "CSO_ADMISSION_V1",
        "partition_seed": PARTITION_SEED,
        "train_fraction_threshold": TRAIN_FRACTION,
        "n_train_states": int(len(partitions["train_state_indices"])),
        "n_eval_states": int(len(partitions["eval_state_indices"])),
        "n_included_bigrams": len(partitions["included_bigrams"]),
        "n_excluded_bigrams": len(partitions["excluded_bigrams"]),
        "n_withheld_trigrams": len(partitions["withheld_trigrams"]),
        "excluded_bigrams": partitions["excluded_bigrams"],
        "withheld_trigrams": partitions["withheld_trigrams"],
        "hashes": hashes,
    }

    if model_config is not None:
        precommit["model_config"] = model_config

    h = hashlib.sha256()
    h.update(json.dumps(hashes, sort_keys=True).encode())
    precommit["integrity_sha256"] = h.hexdigest()

    PRECOMMIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRECOMMIT_PATH, "w") as f:
        json.dump(precommit, f, indent=2)
    print(f"Precommit written to {PRECOMMIT_PATH}")
    return precommit


def verify_precommit(partitions: dict) -> bool:
    """Verify current partitions match the frozen precommit. Raises on mismatch."""
    if not PRECOMMIT_PATH.exists():
        raise FileNotFoundError(f"Precommit not found at {PRECOMMIT_PATH}")

    with open(PRECOMMIT_PATH) as f:
        precommit = json.load(f)

    current_hashes = partition_hashes(partitions)
    frozen_hashes = precommit["hashes"]

    for key in frozen_hashes:
        if current_hashes.get(key) != frozen_hashes[key]:
            raise ValueError(
                f"Precommit verification FAILED: {key} mismatch.\n"
                f"  Frozen: {frozen_hashes[key]}\n"
                f"  Current: {current_hashes.get(key)}"
            )

    h = hashlib.sha256()
    h.update(json.dumps(frozen_hashes, sort_keys=True).encode())
    if h.hexdigest() != precommit["integrity_sha256"]:
        raise ValueError("Precommit integrity hash mismatch")

    print("Precommit verification: PASS")
    return True


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _contains_withheld_trigram(ops: np.ndarray, withheld_set: frozenset) -> bool:
    """Check if any window of 3 consecutive ops forms a withheld trigram."""
    for i in range(len(ops) - 2):
        tri = (int(ops[i]), int(ops[i + 1]), int(ops[i + 2]))
        if tri in withheld_set:
            return True
    return False


def generate_training_example(rng: np.random.RandomState,
                              partitions: dict,
                              min_len: int = TRAIN_LENGTHS[0],
                              max_len: int = TRAIN_LENGTHS[1],
                              max_retries: int = 100):
    """Generate one training example.
    Excludes: excluded bigrams AND withheld trigrams.
    Returns (init_state, ops, final_state).
    """
    train_states = partitions["train_state_indices"]
    included_set = partitions["included_bigram_set"]
    withheld_set = partitions["withheld_trigram_set"]

    for _ in range(max_retries):
        idx = train_states[rng.randint(len(train_states))]
        init_state = index_to_state(idx)
        length = rng.randint(min_len, max_len + 1)
        ops = np.empty(length, dtype=np.int64)

        valid = True
        for i in range(length):
            if i == 0:
                ops[i] = rng.randint(NUM_OPS)
            else:
                found = False
                for _ in range(50):
                    candidate = rng.randint(NUM_OPS)
                    bigram = (int(ops[i - 1]), int(candidate))
                    if bigram not in included_set:
                        continue
                    ops[i] = candidate
                    found = True
                    break
                if not found:
                    valid = False
                    break

        if not valid:
            continue

        if _contains_withheld_trigram(ops, withheld_set):
            continue

        final_state = execute_program(init_state, ops)
        return init_state, ops, final_state

    raise RuntimeError("Failed to generate training example after max_retries")


# ---------------------------------------------------------------------------
# Evaluation strata generators
# ---------------------------------------------------------------------------

def generate_length_extrapolation(rng: np.random.RandomState,
                                  partitions: dict,
                                  min_len: int = EVAL_LENGTHS[0],
                                  max_len: int = EVAL_LENGTHS[1]):
    """Eval stratum: lengths 13-32, eval states, unrestricted bigrams."""
    eval_states = partitions["eval_state_indices"]
    idx = eval_states[rng.randint(len(eval_states))]
    init_state = index_to_state(idx)
    length = rng.randint(min_len, max_len + 1)
    ops = rng.randint(0, NUM_OPS, size=length).astype(np.int64)
    final_state = execute_program(init_state, ops)
    return {"init_state": init_state, "ops": ops, "final_state": final_state,
            "stratum": "length_extrapolation"}


def generate_excluded_bigram(rng: np.random.RandomState,
                             partitions: dict,
                             min_len: int = TRAIN_LENGTHS[0],
                             max_len: int = TRAIN_LENGTHS[1]):
    """Eval stratum: must contain at least one excluded bigram."""
    excluded = partitions["excluded_bigrams"]
    train_states = partitions["train_state_indices"]

    idx = train_states[rng.randint(len(train_states))]
    init_state = index_to_state(idx)
    length = max(2, rng.randint(min_len, max_len + 1))

    forced_bigram = excluded[rng.randint(len(excluded))]
    insert_pos = rng.randint(0, length - 1)

    ops = rng.randint(0, NUM_OPS, size=length).astype(np.int64)
    ops[insert_pos] = forced_bigram[0]
    ops[insert_pos + 1] = forced_bigram[1]

    final_state = execute_program(init_state, ops)
    return {"init_state": init_state, "ops": ops, "final_state": final_state,
            "stratum": "excluded_bigram"}


def generate_withheld_trigram(rng: np.random.RandomState,
                              partitions: dict,
                              min_len: int = 3,
                              max_len: int = EVAL_LENGTHS[1]):
    """Eval stratum: must contain at least one withheld trigram."""
    trigrams = partitions["withheld_trigrams"]
    eval_states = partitions["eval_state_indices"]

    idx = eval_states[rng.randint(len(eval_states))]
    init_state = index_to_state(idx)
    length = max(3, rng.randint(min_len, max_len + 1))

    forced_tri = trigrams[rng.randint(len(trigrams))]
    insert_pos = rng.randint(0, length - 2)

    ops = rng.randint(0, NUM_OPS, size=length).astype(np.int64)
    ops[insert_pos] = forced_tri[0]
    ops[insert_pos + 1] = forced_tri[1]
    ops[insert_pos + 2] = forced_tri[2]

    final_state = execute_program(init_state, ops)
    return {"init_state": init_state, "ops": ops, "final_state": final_state,
            "stratum": "withheld_trigram"}


def generate_held_out_state(rng: np.random.RandomState,
                            partitions: dict,
                            min_len: int = TRAIN_LENGTHS[0],
                            max_len: int = TRAIN_LENGTHS[1]):
    """Eval stratum: held-out initial states, train-length, included bigrams."""
    eval_states = partitions["eval_state_indices"]
    included_set = partitions["included_bigram_set"]

    idx = eval_states[rng.randint(len(eval_states))]
    init_state = index_to_state(idx)
    length = rng.randint(min_len, max_len + 1)
    ops = np.empty(length, dtype=np.int64)

    for i in range(length):
        if i == 0:
            ops[i] = rng.randint(NUM_OPS)
        else:
            while True:
                candidate = rng.randint(NUM_OPS)
                if (int(ops[i - 1]), int(candidate)) in included_set:
                    ops[i] = candidate
                    break

    final_state = execute_program(init_state, ops)
    return {"init_state": init_state, "ops": ops, "final_state": final_state,
            "stratum": "held_out_state"}


def generate_full_intersection(rng: np.random.RandomState,
                               partitions: dict):
    """Eval stratum: eval states + long lengths + excluded bigrams.
    The hardest split: everything withheld at once.
    """
    eval_states = partitions["eval_state_indices"]
    excluded = partitions["excluded_bigrams"]

    idx = eval_states[rng.randint(len(eval_states))]
    init_state = index_to_state(idx)
    length = rng.randint(EVAL_LENGTHS[0], EVAL_LENGTHS[1] + 1)

    forced_bigram = excluded[rng.randint(len(excluded))]
    insert_pos = rng.randint(0, max(1, length - 1))

    ops = rng.randint(0, NUM_OPS, size=length).astype(np.int64)
    if length >= 2:
        ops[insert_pos] = forced_bigram[0]
        ops[min(insert_pos + 1, length - 1)] = forced_bigram[1]

    final_state = execute_program(init_state, ops)
    return {"init_state": init_state, "ops": ops, "final_state": final_state,
            "stratum": "full_intersection"}


def generate_eval_batch(rng: np.random.RandomState,
                        partitions: dict,
                        n_per_stratum: int = 200) -> dict:
    """Generate a complete evaluation batch covering all 5 strata."""
    generators = {
        "length_extrapolation": generate_length_extrapolation,
        "excluded_bigram": generate_excluded_bigram,
        "withheld_trigram": generate_withheld_trigram,
        "held_out_state": generate_held_out_state,
        "full_intersection": generate_full_intersection,
    }
    batch = {}
    for name, gen_fn in generators.items():
        batch[name] = [gen_fn(rng, partitions) for _ in range(n_per_stratum)]
    return batch


# ---------------------------------------------------------------------------
# Vectorized operations (for verification)
# ---------------------------------------------------------------------------

def _build_all_states() -> np.ndarray:
    """Build (65536, 4) array of all register states."""
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


# ---------------------------------------------------------------------------
# Generator matrices for group order verification
# ---------------------------------------------------------------------------

def _build_generator_matrices():
    """Build 4x4 integer matrices over Z_16 for each operation.
    All operations are linear (no translation) so M*state = new_state.
    Convention: state is a column vector, new_state = M @ state.
    Equivalent to: state_row @ M^T = new_state_row.
    """
    matrices = []
    inverses = []
    for op in range(NUM_OPS):
        M = np.eye(4, dtype=np.int64)
        M_inv = np.eye(4, dtype=np.int64)
        if op == 0:  # add_01: r0 <- r0 + r1
            M[0, 1] = 1
            M_inv[0, 1] = MOD - 1
        elif op == 1:  # add_12: r1 <- r1 + r2
            M[1, 2] = 1
            M_inv[1, 2] = MOD - 1
        elif op == 2:  # add_23: r2 <- r2 + r3
            M[2, 3] = 1
            M_inv[2, 3] = MOD - 1
        elif op == 3:  # add_30: r3 <- r3 + r0
            M[3, 0] = 1
            M_inv[3, 0] = MOD - 1
        elif op == 4:  # swap_02
            M = np.array([[0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1]], dtype=np.int64)
            M_inv = M.copy()
        elif op == 5:  # swap_13
            M = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0]], dtype=np.int64)
            M_inv = M.copy()
        elif op == 6:  # rotate_L: (r0,r1,r2,r3) <- (r1,r2,r3,r0)
            M = np.array([[0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [1, 0, 0, 0]], dtype=np.int64)
            M_inv = np.array([[0, 0, 0, 1],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]], dtype=np.int64)
        elif op == 7:  # neg_02: r0 <- -r0, r2 <- -r2
            M = np.diag(np.array([MOD - 1, 1, MOD - 1, 1], dtype=np.int64))
            M_inv = M.copy()
        matrices.append(M % MOD)
        inverses.append(M_inv % MOD)
    return matrices, inverses


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------

def verify_invertibility():
    """Verify all 8 operations are invertible over all 65536 states."""
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
    """Verify all 8 operations are bijections."""
    print("Verifying bijectivity...")
    all_states = _build_all_states()
    for op in range(NUM_OPS):
        result = _apply_op_vectorized(all_states, op)
        result_indices = _states_to_indices(result)
        n_unique = len(np.unique(result_indices))
        if n_unique != STATE_SPACE_SIZE:
            raise AssertionError(
                f"Bijectivity failed: op={op}, {n_unique} unique != {STATE_SPACE_SIZE}"
            )
        print(f"  {OP_NAMES[op]}: PASS ({n_unique} unique outputs)")
    print("Bijectivity: ALL PASS")


def verify_noncommutativity():
    """Verify non-commutativity for each pair (i,j) with i<j."""
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


def verify_generator_matrices():
    """Verify generator matrices match scalar operations on all states."""
    print("Verifying generator matrices...")
    all_states = _build_all_states()
    generators, gen_inverses = _build_generator_matrices()

    for op in range(NUM_OPS):
        M = generators[op]
        result_mat = (all_states @ M.T) % MOD
        result_scalar = _apply_op_vectorized(all_states, op)
        assert np.array_equal(result_mat, result_scalar), (
            f"Generator matrix mismatch for {OP_NAMES[op]}"
        )

        M_inv = gen_inverses[op]
        result_inv_mat = (all_states @ M_inv.T) % MOD
        result_inv_scalar = _apply_inverse_op_vectorized(all_states, op)
        assert np.array_equal(result_inv_mat, result_inv_scalar), (
            f"Inverse matrix mismatch for {OP_NAMES[op]}"
        )

    print("  All 8 generators + inverses match scalar ops: PASS")


def verify_group_order():
    """Verify group order >> 65,536 by BFS on 4x4 matrices mod 16.
    Each operation is linear, so we do BFS in GL_4(Z_16).
    """
    print("Verifying group order via matrix BFS...")
    generators, gen_inverses = _build_generator_matrices()
    all_gens = generators + gen_inverses

    identity = np.eye(4, dtype=np.int64)
    seen = {identity.astype(np.int8).tobytes()}
    queue = [identity]
    cap = GROUP_ORDER_MIN + 1000

    while queue and len(seen) < cap:
        mat = queue.pop(0)
        for gen in all_gens:
            product = (mat @ gen) % MOD
            key = product.astype(np.int8).tobytes()
            if key not in seen:
                seen.add(key)
                queue.append(product)
                if len(seen) >= cap:
                    break
        if len(seen) % 50000 == 0 and len(seen) > 0:
            print(f"  ...{len(seen)} distinct matrices found")

    group_order = len(seen)
    capped = len(queue) > 0

    if capped:
        print(f"  Group order >= {group_order} (BFS capped)")
    else:
        print(f"  Exact group order: {group_order}")

    assert group_order >= GROUP_ORDER_MIN, (
        f"Group order {group_order} < {GROUP_ORDER_MIN} (required >> {STATE_SPACE_SIZE})"
    )
    print(f"  PASS: group order {'>' if capped else '='}"
          f" {group_order} >> {STATE_SPACE_SIZE}")
    return group_order, capped


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
    assert n_exc == NUM_EXCLUDED_BIGRAMS

    n_tri = len(p["withheld_trigrams"])
    print(f"  Withheld trigrams: {n_tri}")
    assert n_tri == NUM_WITHHELD_TRIGRAMS

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
    train_states_set = set(partitions["train_state_indices"])
    eval_states_set = set(partitions["eval_state_indices"])
    included_set = partitions["included_bigram_set"]
    withheld_set = partitions["withheld_trigram_set"]

    for i in range(n_samples):
        init_state, ops, final_state = generate_training_example(rng, partitions)
        expected = execute_program(init_state, ops)
        assert np.array_equal(final_state, expected), f"Train sample {i} mismatch"
        assert state_to_index(init_state) in train_states_set
        assert TRAIN_LENGTHS[0] <= len(ops) <= TRAIN_LENGTHS[1]
        for j in range(1, len(ops)):
            bigram = (int(ops[j - 1]), int(ops[j]))
            assert bigram in included_set, f"Excluded bigram in train: {bigram}"
        assert not _contains_withheld_trigram(ops, withheld_set), (
            f"Withheld trigram in train sample {i}"
        )

    for stratum_name, gen_fn in [
        ("length_extrapolation", generate_length_extrapolation),
        ("excluded_bigram", generate_excluded_bigram),
        ("withheld_trigram", generate_withheld_trigram),
        ("held_out_state", generate_held_out_state),
        ("full_intersection", generate_full_intersection),
    ]:
        for i in range(n_samples // 5):
            example = gen_fn(rng, partitions)
            expected = execute_program(example["init_state"], example["ops"])
            assert np.array_equal(example["final_state"], expected), (
                f"{stratum_name} sample {i} mismatch"
            )
            assert example["stratum"] == stratum_name

    print("Data generation: ALL PASS")


def verify_trigram_exclusion(partitions: dict, n_samples: int = 10000):
    """Verify no withheld trigrams appear in training data (high-volume)."""
    print(f"Verifying trigram exclusion ({n_samples} training samples)...")
    rng = np.random.RandomState(456)
    withheld_set = partitions["withheld_trigram_set"]
    violations = 0

    for _ in range(n_samples):
        _, ops, _ = generate_training_example(rng, partitions)
        if _contains_withheld_trigram(ops, withheld_set):
            violations += 1

    assert violations == 0, f"Trigram leakage: {violations}/{n_samples} training samples"
    print(f"  Zero withheld trigrams in {n_samples} training samples: PASS")


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
    verify_generator_matrices()
    print()
    group_order, capped = verify_group_order()
    print()

    partitions, hashes = verify_partitions()
    print()
    verify_data_generation(partitions)
    print()
    verify_trigram_exclusion(partitions)
    print()

    print("=" * 60)
    print("ALL VERIFICATION CHECKS PASSED")
    print(f"Non-commuting pairs: {nc}/28")
    if commuting:
        print(f"Commuting pairs: {commuting}")
    print(f"Group order: {'>' if capped else '='} {group_order}")
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

    print("\nWriting precommit...")
    create_precommit(partitions)
