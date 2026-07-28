"""Local model inference engine for Atlas R2.

Loads a HuggingFace checkpoint with NF4 quantization,
runs generation with synchronized timing and timeout enforcement,
and returns structured results compatible with the task-record schema.

Protocol: R2.4 (precommit/atlas_r2_protocol_r2_2.md + R2.4 amendment)
"""

import multiprocessing as mp
import queue as queue_mod
import time
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

REPO = Path(__file__).resolve().parent.parent
SYSTEMS_CONFIG = REPO / "configs" / "atlas_r2_systems.yaml"


class _TimeLimit(StoppingCriteria):
    """Stop generation after a wall-clock timeout."""

    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_time is None:
            self.start_time = time.perf_counter()
        return time.perf_counter() - self.start_time > self.max_seconds


def load_systems_config():
    with open(SYSTEMS_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(system_id, device="cuda"):
    """Load a local checkpoint with NF4 quantization per R2.1 protocol.

    All 9 systems use NF4/W4A16 with bfloat16 compute dtype.
    """
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

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=revision,
        quantization_config=bnb_cfg,
        dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok, spec


def generate(model, tok, prompt, max_new_tokens=128, temperature=0.0,
             use_chat=True, timeout_seconds=30):
    """Generate text with synchronized timing and timeout enforcement.

    Uses apply_chat_template(tokenize=True) to avoid double-BOS.
    Returns dict with text, timing, token counts, and status flags.
    """
    if use_chat and hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = tok.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            inputs = tok.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
            )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=2048).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    time_limit = _TimeLimit(timeout_seconds)
    time_limit.start_time = time.perf_counter()
    stopping = StoppingCriteriaList([time_limit])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        if temperature == 0.0:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
                stopping_criteria=stopping,
            )
        else:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=stopping,
            )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - t0

    output_ids = out[0][input_len:]
    text = tok.decode(output_ids, skip_special_tokens=True)

    pad_id = tok.pad_token_id
    n_out = len(output_ids)
    is_empty = n_out == 0 or all(
        oid == pad_id for oid in output_ids.tolist()
    )
    timed_out = (time_limit.start_time is not None
                 and wall_seconds > timeout_seconds)

    eos_reached = (
        n_out > 0
        and tok.eos_token_id is not None
        and int(output_ids[-1]) == tok.eos_token_id
    )
    cap_hit = n_out >= max_new_tokens and not eos_reached

    if is_empty:
        stop_reason = "empty"
    elif timed_out:
        stop_reason = "watchdog_abort"
    elif eos_reached:
        stop_reason = "eos"
    elif cap_hit:
        stop_reason = "cap_hit"
    else:
        stop_reason = "eos"

    return {
        "text": text,
        "input_tokens": input_len,
        "output_tokens": n_out,
        "wall_seconds": round(wall_seconds, 3),
        "is_empty": is_empty,
        "timed_out": timed_out,
        "eos_reached": eos_reached,
        "cap_hit": cap_hit,
        "stop_reason": stop_reason,
    }


def batch_generate(model, tok, prompts, max_new_tokens=128, temperature=0.0):
    results = []
    for prompt in prompts:
        r = generate(model, tok, prompt, max_new_tokens, temperature)
        results.append(r)
    return results


def _worker_main(system_id, req_queue, resp_queue):
    """Child process: load model, process generation requests."""
    try:
        model, tok, spec = load_model(system_id)
        resp_queue.put(("ready", {
            "hf_id": spec["hf_id"],
            "params_b": spec.get("params_b"),
            "hf_revision": spec.get("hf_revision"),
        }))
    except Exception as e:
        resp_queue.put(("error", str(e)))
        return

    while True:
        try:
            msg = req_queue.get(timeout=600)
        except queue_mod.Empty:
            break
        if msg is None:
            break
        prompt, max_new_tokens, temperature, timeout_seconds, use_chat = msg
        try:
            result = generate(model, tok, prompt, max_new_tokens,
                              temperature, use_chat, timeout_seconds)
            resp_queue.put(("result", result))
        except Exception as e:
            resp_queue.put(("error", str(e)))

    del model, tok
    torch.cuda.empty_cache()


class SupervisedWorker:
    """Process-isolated inference with hard watchdog kill."""

    def __init__(self, system_id):
        self.system_id = system_id
        self._ctx = mp.get_context("spawn")
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = None
        self.spec = None

    def start(self, load_timeout=300):
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self.system_id, self._req_q, self._resp_q),
            daemon=True)
        self._proc.start()
        try:
            msg_type, payload = self._resp_q.get(timeout=load_timeout)
        except queue_mod.Empty:
            self._proc.kill()
            self._proc.join(timeout=10)
            raise RuntimeError(
                f"Model load timeout for {self.system_id}")
        if msg_type == "error":
            raise RuntimeError(f"Model load failed: {payload}")
        self.spec = payload

    def generate(self, prompt, max_new_tokens, temperature,
                 watchdog_seconds, use_chat=True):
        if self._proc is None or not self._proc.is_alive():
            raise RuntimeError("Worker not running")
        t0 = time.perf_counter()
        self._req_q.put((prompt, max_new_tokens, temperature,
                         watchdog_seconds, use_chat))
        try:
            msg_type, payload = self._resp_q.get(
                timeout=watchdog_seconds)
        except queue_mod.Empty:
            wall = time.perf_counter() - t0
            self._proc.kill()
            self._proc.join(timeout=10)
            self._proc = None
            return {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "wall_seconds": round(min(wall, watchdog_seconds), 3),
                "is_empty": True,
                "timed_out": True,
                "eos_reached": False,
                "cap_hit": False,
                "stop_reason": "watchdog_abort",
            }
        if msg_type == "error":
            raise RuntimeError(f"Generation error: {payload}")
        result = payload
        result["user_wall_seconds"] = round(
            time.perf_counter() - t0, 3)
        return result

    def respawn(self, load_timeout=300):
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=10)
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self.start(load_timeout)

    def shutdown(self):
        if self._proc is not None and self._proc.is_alive():
            try:
                self._req_q.put(None)
                self._proc.join(timeout=30)
            except Exception:
                self._proc.kill()
                self._proc.join(timeout=10)
        self._proc = None

    @property
    def alive(self):
        return self._proc is not None and self._proc.is_alive()


if __name__ == "__main__":
    import sys

    system_id = sys.argv[1] if len(sys.argv) > 1 else "qwen3_0.6b"
    print(f"Loading {system_id} (NF4/bf16)...")
    model, tok, spec = load_model(system_id)
    print(f"  {spec['hf_id']} ({spec['params_b']}B) on {model.device}")

    prompts = [
        "What is the capital of France? Answer in one word.",
        "Translate to Spanish: 'The weather is nice today.'",
    ]

    for p in prompts:
        r = generate(model, tok, p)
        out = r["text"][:200].encode("ascii", errors="replace").decode()
        print(f"\n  Prompt: {p}")
        print(f"  Output: {out}")
        print(f"  Tokens: {r['input_tokens']} in, {r['output_tokens']} out, "
              f"{r['wall_seconds']:.2f}s, empty={r['is_empty']}")

    del model, tok
    torch.cuda.empty_cache()
    print("\nSMOKE PASS")
