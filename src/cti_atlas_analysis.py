"""Atlas R2 result analysis and Gate A ranking.

Reads P1 raw JSON results (MKQA, PolicyBench) and applies Gate A:
  - 2 per family: highest quality + cheapest within 10pt of family best
  - Output: 6 anchors for P2/P3
"""

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"


def load_p1_results():
    results = {}
    for name in ["atlas_r2_p1_mkqa_raw.json", "atlas_r2_p1_policybench_raw.json"]:
        path = RESULTS_DIR / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                results[name] = json.load(f)
    return results


def compute_composite_quality(mkqa_results, pb_results):
    systems = {}
    all_ids = set()
    if mkqa_results:
        all_ids |= set(mkqa_results.keys())
    if pb_results:
        all_ids |= set(pb_results.keys())

    for sid in all_ids:
        entry = {"system_id": sid}
        if mkqa_results and sid in mkqa_results:
            m = mkqa_results[sid]
            entry["mkqa_pass_rate"] = m["pass_rate"]
            entry["mkqa_f1"] = m["mean_f1"]
            entry["mkqa_energy_j"] = m["energy"]["energy_joules"]
            entry["params_b"] = m["params_b"]
            entry["family"] = m["family"]
        if pb_results and sid in pb_results:
            p = pb_results[sid]
            entry["pb_pass_rate"] = p["pass_rate"]
            entry["pb_binary_rate"] = p.get("binary_pass_rate", 0)
            entry["pb_dollar_rate"] = p.get("dollar_pass_rate", 0)
            entry["pb_energy_j"] = p["energy"]["energy_joules"]
            if "params_b" not in entry:
                entry["params_b"] = p["params_b"]
                entry["family"] = p["family"]

        mkqa_r = entry.get("mkqa_pass_rate", 0)
        pb_r = entry.get("pb_pass_rate", 0)
        n = sum(1 for k in ["mkqa_pass_rate", "pb_pass_rate"] if k in entry)
        entry["avg_pass_rate"] = (mkqa_r + pb_r) / max(n, 1)
        systems[sid] = entry

    return systems


def apply_gate_a(systems):
    """Gate A: 2 per family (best quality + cheapest within 10pt)."""
    by_family = defaultdict(list)
    for sid, s in systems.items():
        by_family[s["family"]].append(s)

    anchors = {}
    for family, members in sorted(by_family.items()):
        ranked = sorted(members, key=lambda x: x["avg_pass_rate"], reverse=True)
        best = ranked[0]
        anchors[best["system_id"]] = {**best, "gate_a_reason": "best_quality"}

        threshold = best["avg_pass_rate"] - 0.10
        cheapest = sorted(
            [m for m in ranked if m["avg_pass_rate"] >= threshold],
            key=lambda x: x["params_b"],
        )
        if cheapest[0]["system_id"] != best["system_id"]:
            pick = cheapest[0]
        elif len(cheapest) > 1:
            pick = cheapest[1]
        else:
            pick = ranked[1] if len(ranked) > 1 else None

        if pick and pick["system_id"] not in anchors:
            anchors[pick["system_id"]] = {
                **pick, "gate_a_reason": "cheapest_within_10pt",
            }

    return anchors


def print_ranking(systems, anchors=None):
    ranked = sorted(systems.values(),
                    key=lambda x: x["avg_pass_rate"], reverse=True)

    print(f"{'System':<20s} {'Params':>6s} {'Family':<10s} "
          f"{'MKQA':>6s} {'PB':>6s} {'Avg':>6s} {'Anchor':>8s}")
    print("-" * 70)

    for s in ranked:
        anchor = ""
        if anchors and s["system_id"] in anchors:
            anchor = anchors[s["system_id"]]["gate_a_reason"][:8]
        mkqa = f"{s.get('mkqa_pass_rate', 0):.1%}"
        pb = f"{s.get('pb_pass_rate', 0):.1%}"
        avg = f"{s['avg_pass_rate']:.1%}"
        print(f"{s['system_id']:<20s} {s['params_b']:>5.1f}B "
              f"{s.get('family', ''):>10s} {mkqa:>6s} {pb:>6s} "
              f"{avg:>6s} {anchor:>8s}")


if __name__ == "__main__":
    results = load_p1_results()
    mkqa = results.get("atlas_r2_p1_mkqa_raw.json")
    pb = results.get("atlas_r2_p1_policybench_raw.json")

    if not mkqa and not pb:
        print("No P1 results found. Run P1 first.")
        raise SystemExit(1)

    systems = compute_composite_quality(mkqa, pb)
    anchors = apply_gate_a(systems)

    print("=" * 70)
    print("P1 RAW SCREEN - All Systems")
    print("=" * 70)
    print_ranking(systems, anchors)

    print(f"\n{'='*70}")
    print(f"GATE A ANCHORS ({len(anchors)} selected)")
    print("=" * 70)
    for sid, a in sorted(anchors.items(),
                         key=lambda x: x[1]["avg_pass_rate"], reverse=True):
        print(f"  {sid}: {a['avg_pass_rate']:.1%} avg "
              f"({a['gate_a_reason']})")

    out_path = RESULTS_DIR / "atlas_r2_gate_a.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "anchors": {k: {kk: vv for kk, vv in v.items()}
                        for k, v in anchors.items()},
            "all_systems": {k: {kk: vv for kk, vv in v.items()}
                            for k, v in systems.items()},
        }, f, indent=2)
    print(f"\nGate A results written to {out_path}")
