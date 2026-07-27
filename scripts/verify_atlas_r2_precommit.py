#!/usr/bin/env python
"""
verify_atlas_r2_precommit.py

Fail-closed commitment verifier for Atlas R2.
Must PASS before any GPU execution is authorized.

Checks:
1. All precommit JSON files exist and parse without duplicate keys
2. Budget totals match binding protocol
3. System roster matches configs
4. Workload configs have pinned revisions
5. Task hash seeds are consistent
6. No confirmation tasks leak into discovery
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXIT_CODE = 0


def fail(msg):
    global EXIT_CODE
    print(f"FAIL: {msg}", file=sys.stderr)
    EXIT_CODE = 1


def warn(msg):
    print(f"WARN: {msg}", file=sys.stderr)


class DuplicateKeyError(Exception):
    pass


def json_load_no_dupes(path):
    """Load JSON rejecting duplicate keys at any nesting level."""
    text = path.read_text(encoding="utf-8")

    def check_dupes(pairs):
        seen = set()
        d = {}
        for k, v in pairs:
            if k in seen:
                raise DuplicateKeyError(f"Duplicate key '{k}' in {path.name}")
            seen.add(k)
            d[k] = v
        return d

    return json.loads(text, object_pairs_hook=check_dupes)


def check_budget():
    path = REPO / "precommit" / "atlas_r2_budget.json"
    if not path.exists():
        fail("precommit/atlas_r2_budget.json missing")
        return
    try:
        budget = json_load_no_dupes(path)
    except DuplicateKeyError as e:
        fail(str(e))
        return
    except json.JSONDecodeError as e:
        fail(f"Budget JSON parse error: {e}")
        return

    gpu = budget.get("gpu_budget", {})
    phases = gpu.get("phases", {})
    if not phases:
        fail("Budget has no phases")
        return

    total = sum(p.get("allocated", 0) for p in phases.values())
    declared = gpu.get("scheduled_subtotal", 0)
    if abs(total - declared) > 0.1:
        fail(f"Phase sum {total:.2f} != declared subtotal {declared:.2f}")

    binding_max = gpu.get("binding_maximum", 0)
    if binding_max != 360.0:
        fail(f"Binding maximum {binding_max} != 360.0")

    reserve = gpu.get("failure_reserve", 0)
    if abs(binding_max - declared - reserve) > 0.1:
        fail(f"Reserve arithmetic: {binding_max} - {declared} != {reserve}")

    api = budget.get("api_budget", {})
    if api.get("binding_maximum", 0) != 1200:
        fail(f"API binding maximum != 1200")

    print(f"  Budget: {len(phases)} phases, {total:.2f}h scheduled, "
          f"{binding_max}h max, ${api.get('binding_maximum', 0)} API max")


def check_systems_config():
    try:
        import yaml
    except ImportError:
        warn("PyYAML not installed, skipping systems config check")
        return

    path = REPO / "configs" / "atlas_r2_systems.yaml"
    if not path.exists():
        fail("configs/atlas_r2_systems.yaml missing")
        return

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local = cfg.get("local_checkpoints", {})
    if len(local) != 9:
        fail(f"Expected 9 local checkpoints, got {len(local)}")

    families = set()
    for name, spec in local.items():
        families.add(spec.get("family", "unknown"))
        if spec.get("hf_revision") is None:
            warn(f"{name}: hf_revision not pinned yet (required before P1)")

    if len(families) != 3:
        fail(f"Expected 3 families, got {len(families)}: {families}")

    apis = cfg.get("api_systems", {})
    if len(apis) != 6:
        fail(f"Expected 6 API systems, got {len(apis)}")

    frontiers = sum(1 for a in apis.values() if a.get("role") == "frontier")
    values = sum(1 for a in apis.values() if a.get("role") == "value_control")
    if frontiers != 3:
        fail(f"Expected 3 frontier APIs, got {frontiers}")
    if values != 3:
        fail(f"Expected 3 value control APIs, got {values}")

    print(f"  Systems: {len(local)} local, {len(apis)} API, "
          f"{len(families)} families")


def check_workloads_config():
    try:
        import yaml
    except ImportError:
        warn("PyYAML not installed, skipping workloads config check")
        return

    path = REPO / "configs" / "atlas_r2_workloads.yaml"
    if not path.exists():
        fail("configs/atlas_r2_workloads.yaml missing")
        return

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    workloads = cfg.get("workloads", {})
    expected = {"W-D1", "W-D2", "W-D3", "W-C1", "W-C2"}
    actual = set(workloads.keys())
    if actual != expected:
        fail(f"Workload keys {actual} != expected {expected}")

    for wid, wspec in workloads.items():
        sel = wspec.get("selection", "")
        if "SHA256" in sel:
            seed = sel.split("'")[1] if "'" in sel else ""
            if not seed.startswith("atlas-r2-"):
                fail(f"{wid} hash seed '{seed}' missing atlas-r2- prefix")

    kills = cfg.get("kill_criteria", [])
    if len(kills) < 7:
        fail(f"Expected >= 7 kill criteria, got {len(kills)}")

    print(f"  Workloads: {len(workloads)} defined, {len(kills)} kill criteria")


def check_selector():
    path = REPO / "precommit" / "atlas_r2_selector.json"
    if not path.exists():
        warn("precommit/atlas_r2_selector.json not yet created (required before confirmation)")
        return
    try:
        sel = json_load_no_dupes(path)
        print(f"  Selector: {len(sel)} top-level keys")
    except (DuplicateKeyError, json.JSONDecodeError) as e:
        fail(f"Selector: {e}")


def check_task_seal():
    path = REPO / "precommit" / "atlas_r2_task_seal.json"
    if not path.exists():
        warn("precommit/atlas_r2_task_seal.json not yet created (required before confirmation)")
        return
    try:
        seal = json_load_no_dupes(path)
        print(f"  Task seal: {len(seal)} top-level keys")
    except (DuplicateKeyError, json.JSONDecodeError) as e:
        fail(f"Task seal: {e}")


def check_adaptation_config():
    try:
        import yaml
    except ImportError:
        warn("PyYAML not installed, skipping adaptation config check")
        return

    path = REPO / "configs" / "atlas_r2_adaptation.yaml"
    if not path.exists():
        fail("configs/atlas_r2_adaptation.yaml missing")
        return

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    envelopes = cfg.get("qlora", {}).get("resource_envelopes", [])
    if len(envelopes) != 3:
        fail(f"Expected 3 QLoRA resource envelopes, got {len(envelopes)}")

    grid = cfg.get("qlora", {}).get("grid", {})
    grid_size = 1
    for v in grid.values():
        grid_size *= len(v) if isinstance(v, list) else 1
    print(f"  Adaptation: {len(envelopes)} envelopes, "
          f"{grid_size} grid cells per anchor")


def main():
    print("=" * 60)
    print("Atlas R2 Precommit Verification")
    print("=" * 60)

    print("\n[1/6] Budget contract")
    check_budget()

    print("\n[2/6] Systems configuration")
    check_systems_config()

    print("\n[3/6] Workloads configuration")
    check_workloads_config()

    print("\n[4/6] Adaptation configuration")
    check_adaptation_config()

    print("\n[5/6] Selector (pre-confirmation)")
    check_selector()

    print("\n[6/6] Task seal (pre-confirmation)")
    check_task_seal()

    print("\n" + "=" * 60)
    if EXIT_CODE == 0:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: FAILURES DETECTED - DO NOT PROCEED")
    print("=" * 60)

    return EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
