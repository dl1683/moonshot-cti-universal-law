"""API inference client for Atlas R2 Goliath systems.

Wraps OpenAI, Anthropic, and Google Gemini APIs with cost tracking
and structured result format matching the cost ledger.
"""

import os
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SYSTEMS_CONFIG = REPO / "configs" / "atlas_r2_systems.yaml"


def load_api_config():
    with open(SYSTEMS_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("api_systems", {})


def _compute_cost(spec, input_tokens, output_tokens):
    pricing = spec.get("pricing_per_m_tokens", {})
    input_rate = pricing.get("input", 0.0)
    output_rate = pricing.get("output", 0.0)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def call_openai(model_id, prompt, max_tokens=128, temperature=0.0):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    wall = time.perf_counter() - t0
    usage = resp.usage
    return {
        "text": resp.choices[0].message.content or "",
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "wall_seconds": round(wall, 3),
    }


def call_anthropic(model_id, prompt, max_tokens=128, temperature=0.0):
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    wall = time.perf_counter() - t0
    return {
        "text": resp.content[0].text if resp.content else "",
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "wall_seconds": round(wall, 3),
    }


def call_gemini(model_id, prompt, max_tokens=128, temperature=0.0):
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai")

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_id)
    config = genai.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    t0 = time.perf_counter()
    resp = model.generate_content(prompt, generation_config=config)
    wall = time.perf_counter() - t0

    usage = getattr(resp, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

    return {
        "text": resp.text if resp.text else "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "wall_seconds": round(wall, 3),
    }


PROVIDER_DISPATCH = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_gemini,
}


def call_api(system_id, prompt, max_tokens=128, temperature=0.0):
    configs = load_api_config()
    if system_id not in configs:
        raise ValueError(f"Unknown API system: {system_id}. "
                         f"Available: {list(configs.keys())}")

    spec = configs[system_id]
    provider = spec["provider"]
    model_id = spec.get("model_id")
    if model_id is None:
        raise ValueError(f"{system_id}: model_id not pinned yet")

    fn = PROVIDER_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider}")

    result = fn(model_id, prompt, max_tokens, temperature)
    result["api_cost_usd"] = round(
        _compute_cost(spec, result["input_tokens"], result["output_tokens"]),
        6,
    )
    return result


if __name__ == "__main__":
    configs = load_api_config()
    print(f"API systems configured: {len(configs)}")
    for sid, spec in configs.items():
        model_id = spec.get("model_id", "NOT PINNED")
        pricing = spec.get("pricing_per_m_tokens", {})
        print(f"  {sid}: provider={spec['provider']}, "
              f"model={model_id}, "
              f"${pricing.get('input',0)}/{pricing.get('output',0)} per M tokens")
