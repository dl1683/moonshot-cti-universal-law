"""Run the preregistered benchmark bridge: lm-eval on 12 models x 5 tasks.

Skips GPQA (gated) and LFM2-350M (DLL failure).
Results go to results/downstream_bridge/bridge_run_01/<model_slug>/<task>/
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BRIDGE_DIR = RESULTS / "downstream_bridge" / "bridge_run_01"

MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "allenai/OLMo-1B-hf",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "state-spaces/mamba-130m-hf",
    "Zyphra/Zamba2-1.2B",
    "ibm-granite/granite-3.0-1b-a400m-base",
]

TASKS = [
    "leaderboard_bbh",
    "leaderboard_instruction_following",
    "leaderboard_math_hard",
    "leaderboard_musr",
    "leaderboard_mmlu_pro",
]

LM_EVAL = r"C:\Users\devan\AppData\Local\Programs\Python\Python313\Scripts\lm-eval.exe"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def run_task(model_id: str, task: str) -> dict:
    slug = model_slug(model_id)
    out_dir = BRIDGE_DIR / slug / task
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        LM_EVAL,
        "--model", "hf",
        "--model_args", f"pretrained={model_id},dtype=auto",
        "--tasks", task,
        "--batch_size", "auto",
        "--device", "cuda:0",
        "--output_path", str(out_dir),
    ]

    print(f"\n{'='*60}")
    print(f"MODEL: {model_id}")
    print(f"TASK:  {task}")
    print(f"CMD:   {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        elapsed = time.time() - t0
        status = "pass" if result.returncode == 0 else "fail"

        if result.returncode != 0:
            print(f"FAILED (rc={result.returncode}, {elapsed:.1f}s)")
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-10:]:
                print(f"  ERR: {line}")
        else:
            print(f"PASSED ({elapsed:.1f}s)")

        return {
            "model_id": model_id,
            "task": task,
            "status": status,
            "returncode": result.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "output_dir": str(out_dir),
            "error_tail": "\n".join(err_lines[-5:]) if result.returncode != 0 else None,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"TIMEOUT ({elapsed:.1f}s)")
        return {
            "model_id": model_id,
            "task": task,
            "status": "timeout",
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 1),
            "output_dir": str(out_dir),
            "error_tail": "Timed out after 3600s",
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {e}")
        return {
            "model_id": model_id,
            "task": task,
            "status": "error",
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 1),
            "output_dir": str(out_dir),
            "error_tail": str(e),
        }


def main():
    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

    log_path = BRIDGE_DIR / "execution_log.json"
    if log_path.exists():
        all_results = json.loads(log_path.read_text(encoding="utf-8"))
    else:
        all_results = []

    completed = {(r["model_id"], r["task"]) for r in all_results}

    total = len(MODELS) * len(TASKS)
    done = len(completed)

    for model_id in MODELS:
        for task in TASKS:
            if (model_id, task) in completed:
                print(f"SKIP (already done): {model_slug(model_id)} / {task}")
                continue
            done += 1
            print(f"\n[{done}/{total}] Starting {model_slug(model_id)} / {task}")

            result = run_task(model_id, task)
            all_results.append(result)

            log_path.write_text(
                json.dumps(all_results, indent=2),
                encoding="utf-8",
            )

    passed = sum(1 for r in all_results if r["status"] == "pass")
    failed = sum(1 for r in all_results if r["status"] != "pass")
    print(f"\n{'='*60}")
    print(f"BENCHMARK BRIDGE COMPLETE: {passed}/{total} passed, {failed} failed")
    print(f"Results in: {BRIDGE_DIR}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
