"""Local model inference engine for Atlas R2.

Loads a HuggingFace checkpoint, runs generation with energy tracking,
and returns structured results compatible with the cost ledger.
"""

import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
SYSTEMS_CONFIG = REPO / "configs" / "atlas_r2_systems.yaml"


def load_systems_config():
    with open(SYSTEMS_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(system_id, dtype=torch.float16, device="cuda"):
    cfg = load_systems_config()
    local = cfg.get("local_checkpoints", {})
    if system_id not in local:
        raise ValueError(f"Unknown system: {system_id}. "
                         f"Available: {list(local.keys())}")

    spec = local[system_id]
    hf_id = spec["hf_id"]
    revision = spec.get("hf_revision")

    tok = AutoTokenizer.from_pretrained(
        hf_id, revision=revision, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=revision, dtype=dtype,
        device_map=device, trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok, spec


def generate(model, tok, prompt, max_new_tokens=128, temperature=0.0):
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        if temperature == 0.0:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
            )
        else:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature,
                pad_token_id=tok.pad_token_id,
            )
    wall_seconds = time.perf_counter() - t0

    output_ids = out[0][input_len:]
    text = tok.decode(output_ids, skip_special_tokens=True)

    return {
        "text": text,
        "input_tokens": input_len,
        "output_tokens": len(output_ids),
        "wall_seconds": round(wall_seconds, 3),
    }


def batch_generate(model, tok, prompts, max_new_tokens=128, temperature=0.0):
    results = []
    for prompt in prompts:
        r = generate(model, tok, prompt, max_new_tokens, temperature)
        results.append(r)
    return results


if __name__ == "__main__":
    import json

    system_id = "qwen3_0.6b"
    print(f"Loading {system_id}...")
    model, tok, spec = load_model(system_id)
    print(f"  {spec['hf_id']} ({spec['params_b']}B) on {model.device}")

    prompts = [
        "What is the capital of France? Answer in one word.",
        "Translate to Spanish: 'The weather is nice today.'",
    ]

    for p in prompts:
        r = generate(model, tok, p)
        print(f"\n  Prompt: {p}")
        print(f"  Output: {r['text'][:100]}")
        print(f"  Tokens: {r['input_tokens']} in, {r['output_tokens']} out, "
              f"{r['wall_seconds']:.2f}s")

    del model, tok
    torch.cuda.empty_cache()
    print("\nSMOKE PASS")
