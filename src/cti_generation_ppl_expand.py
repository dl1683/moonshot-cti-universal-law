#!/usr/bin/env python -u
"""
Compute WikiText-103 PPL for new models.
Appends to results/cti_generation_ppl.json.
"""
import json, time, gc, sys
import numpy as np
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
PPL_FILE = RESULTS / "cti_generation_ppl.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models to evaluate (key, hf_id, name)
NEW_MODELS = [
    ("qwen2-0.5b",   "Qwen/Qwen2-0.5B",           "Qwen2-0.5B"),
    ("falcon-h1-3b", "tiiuae/Falcon-H1-3B-Base",    "Falcon-H1-3B"),
    ("granite-micro", "ibm-granite/granite-4.0-micro", "Granite-4.0-Micro"),
    ("lfm2.5-1.2b",  "LiquidAI/LFM2.5-1.2B-Base",  "LFM2.5-1.2B"),
]


def compute_perplexity_wikitext(model, tokenizer, max_tokens=100000, seq_len=512):
    """Compute PPL on WikiText-103 validation (non-overlapping sliding window)."""
    from datasets import load_dataset

    print(f"    Loading WikiText-103 validation...", flush=True)
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    full_text = "\n\n".join([x["text"] for x in wiki if x["text"].strip()])
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"][0]
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]
    total_tokens = len(input_ids)
    print(f"    Total tokens: {total_tokens}", flush=True)

    total_loss = 0.0
    total_counted = 0
    model.eval()
    with torch.no_grad():
        for begin in range(0, total_tokens - 1, seq_len):
            end = min(begin + seq_len, total_tokens)
            chunk = input_ids[begin:end].unsqueeze(0).to(DEVICE)
            outputs = model(chunk, labels=chunk)
            n_tok = chunk.shape[1] - 1
            total_loss += outputs.loss.item() * n_tok
            total_counted += n_tok

    if total_counted == 0:
        return float("inf")
    avg_ce = total_loss / total_counted
    ppl = float(np.exp(avg_ce))
    print(f"    PPL = {ppl:.2f} (CE = {avg_ce:.4f}, n = {total_counted})", flush=True)
    return ppl


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load existing
    if PPL_FILE.exists():
        with open(PPL_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    to_run = [(k, hf, name) for k, hf, name in NEW_MODELS if k not in results]
    if not to_run:
        print("All models already computed.")
        return

    print(f"\n{'='*70}")
    print(f"  PPL EVALUATION: {len(to_run)} new models")
    print(f"{'='*70}\n")

    for key, hf_id, name in to_run:
        print(f"\n--- {name} ({hf_id}) ---")
        t0 = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(DEVICE)
            model.eval()

            ppl = compute_perplexity_wikitext(model, tokenizer)
            elapsed = time.time() - t0

            results[key] = {
                "model": name,
                "hf_id": hf_id,
                "ppl": ppl,
                "log_ppl": float(np.log(ppl)),
                "time_s": elapsed,
            }
            print(f"    Done: PPL={ppl:.2f}, time={elapsed:.1f}s")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            elapsed = time.time() - t0
            results[key] = {"model": name, "error": str(e)}
            print(f"    ERROR: {e} ({elapsed:.1f}s)")

        with open(PPL_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for k, v in results.items():
        if 'ppl' in v:
            print(f"  {v['model']:<25} PPL={v['ppl']:>8.2f}  log(PPL)={v['log_ppl']:>6.3f}")


if __name__ == "__main__":
    main()
