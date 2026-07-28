#!/usr/bin/env python
"""Build sealed R2.2 prevalence and challenge panels for Atlas W-D3.

Protocol: precommit/atlas_r2_protocol_r2_2.md, Section 4.

Outputs:
  data/policybench/r2_2_prevalence.json
  data/policybench/r2_2_challenge.json
  data/policybench/r2_2_field_prior.json
  data/policybench/r2_2_panel_manifest.json
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def canonical_identity(scenario_dict):
    m = scenario_dict["metadata"]
    return (
        scenario_dict["source_dataset"],
        str(m["dataset_year"]),
        str(m["household_id"]),
        str(m["tax_unit_id"]),
        scenario_dict["country"],
    )


def identity_string(id_tuple):
    return "|".join(id_tuple)


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path):
    """Hash file in binary mode (CRLF-preserving)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_rank_challenge(id_tuple):
    return sha256_hex("atlas-r2.2-d3-challenge" + identity_string(id_tuple))


TAX_FIELDS = {
    "federal_income_tax_before_refundable_credits",
    "state_income_tax_before_refundable_credits",
    "local_income_tax",
    "payroll_tax",
    "self_employment_tax",
}
CREDIT_FIELDS = {
    "federal_refundable_credits",
    "state_refundable_credits",
}
BENEFIT_AMOUNT_FIELDS = {"snap", "ssi", "tanf"}


def classify_stratum(refs_dict):
    """Classify a household into one of four strata per protocol Section 4.2."""
    t = any(
        abs(refs_dict.get(f, 0.0)) >= 50.0 for f in TAX_FIELDS
    )
    credit = any(
        abs(refs_dict.get(f, 0.0)) >= 50.0 for f in CREDIT_FIELDS
    )
    b_amount = any(
        abs(refs_dict.get(f, 0.0)) >= 50.0 for f in BENEFIT_AMOUNT_FIELDS
    )
    b_elig = any(
        refs_dict.get(f, 0.0) == 1.0
        for f in refs_dict
        if f.endswith("_eligible")
    )
    b = b_amount or b_elig

    if credit:
        return "REFUNDABLE_CREDIT"
    if t and not b:
        return "TAX_ONLY"
    if b and not t:
        return "BENEFIT_ONLY"
    if t and b:
        return "TAX_AND_BENEFIT"
    return "NONE"


def is_binary_field(field_name):
    return "eligible" in field_name


def all_zero_correct(field_name, ref_value):
    """Whether the all-zero baseline is correct on this field."""
    if is_binary_field(field_name):
        return ref_value == 0.0
    if ref_value == 0.0:
        return True
    return abs(ref_value) < 50.0


def compute_ess(counts):
    """ESS(m) = (sum m_i)^2 / sum(m_i^2). Returns 0 if denominator is 0."""
    s = sum(counts)
    ss = sum(c * c for c in counts)
    if ss == 0:
        return 0.0
    return (s * s) / ss


def ess_preflight(households, panel_label):
    """Compute all ESS checks for a panel. Returns dict of results."""
    global_elig_rescue = []
    global_elig_harm = []
    global_amt_rescue = []
    global_amt_harm = []

    strata_rescue = {}

    for hh in households:
        stratum = hh.get("stratum", "P")
        refs = hh["refs_dict"]

        e_rescue = 0
        e_harm = 0
        a_rescue = 0
        a_harm = 0
        all_rescue = 0

        for fname, ref_val in refs.items():
            correct = all_zero_correct(fname, ref_val)
            if is_binary_field(fname):
                if correct:
                    e_harm += 1
                else:
                    e_rescue += 1
                    all_rescue += 1
            else:
                if correct:
                    a_harm += 1
                else:
                    a_rescue += 1
                    all_rescue += 1

        global_elig_rescue.append(e_rescue)
        global_elig_harm.append(e_harm)
        global_amt_rescue.append(a_rescue)
        global_amt_harm.append(a_harm)

        if stratum != "P":
            if stratum not in strata_rescue:
                strata_rescue[stratum] = []
            strata_rescue[stratum].append(all_rescue)

    result = {
        "panel": panel_label,
        "global_eligibility_rescue_ess": round(compute_ess(global_elig_rescue), 2),
        "global_eligibility_harm_ess": round(compute_ess(global_elig_harm), 2),
        "global_amount_rescue_ess": round(compute_ess(global_amt_rescue), 2),
        "global_amount_harm_ess": round(compute_ess(global_amt_harm), 2),
        "stratum_rescue_ess": {},
    }

    for st, counts in sorted(strata_rescue.items()):
        result["stratum_rescue_ess"][st] = round(compute_ess(counts), 2)

    return result


def build_field_prior(c0_households):
    """Build field-prior baseline from C0 calibration households."""
    from collections import defaultdict

    binary_values = defaultdict(list)
    numeric_values = defaultdict(list)

    for hh in c0_households:
        for fname, ref_val in hh["refs_dict"].items():
            if is_binary_field(fname):
                binary_values[fname].append(int(ref_val))
            else:
                numeric_values[fname].append(ref_val)

    prior_map = {}
    for fname, vals in binary_values.items():
        zeros = vals.count(0)
        ones = vals.count(1)
        prior_map[fname] = 0 if zeros >= ones else 1

    for fname, vals in numeric_values.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        if n == 0:
            prior_map[fname] = 0
        elif n % 2 == 1:
            median = vals_sorted[n // 2]
        else:
            median = (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2.0
        if n > 0:
            rounded = int(
                math.copysign(
                    math.floor(abs(median) + 0.5), median
                )
            ) if median != 0 else 0
            prior_map[fname] = rounded

    return prior_map


def load_c0_household_ids(manifest_path):
    """Extract household_ids from C0 calibration scenarios."""
    ids = set()
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sc = json.loads(row["scenario_json"])
            meta = sc["metadata"]
            ids.add(meta["household_id"])
    return ids


def load_c0_as_households(manifest_path, refs_path):
    """Load C0 as household dicts for field-prior computation."""
    scenarios = {}
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["scenario_id"]
            scenarios[sid] = json.loads(row["scenario_json"])

    refs_by_sid = {}
    with open(refs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["scenario_id"]
            if sid not in refs_by_sid:
                refs_by_sid[sid] = {}
            refs_by_sid[sid][row["variable"]] = float(row["value"])

    households = []
    for sid, sc in scenarios.items():
        if sid in refs_by_sid:
            households.append({
                "scenario_id": sid,
                "refs_dict": refs_by_sid[sid],
            })
    return households


def generate_panel(n, seed, excluded_ids, country, year):
    """Generate scenarios and compute ground truth."""
    from policybench.scenarios import generate_scenarios, scenario_to_dict
    from policybench.ground_truth import calculate_ground_truth

    t0 = time.time()
    scenarios = generate_scenarios(
        n=n, seed=seed, excluded_household_ids=excluded_ids, country=country,
    )
    t_gen = time.time() - t0
    print(f"  Generated {len(scenarios)} scenarios in {t_gen:.1f}s")

    t0 = time.time()
    refs_df = calculate_ground_truth(scenarios, year=year)
    t_ref = time.time() - t0
    print(f"  Computed references in {t_ref:.1f}s")

    refs_by_sid = {}
    for _, row in refs_df.iterrows():
        sid = row["scenario_id"]
        if sid not in refs_by_sid:
            refs_by_sid[sid] = {}
        refs_by_sid[sid][row["variable"]] = float(row["value"])

    households = []
    for sc_obj in scenarios:
        d = scenario_to_dict(sc_obj)
        sid = d["id"]
        refs = refs_by_sid.get(sid, {})
        cid = canonical_identity(d)
        fields_sorted = sorted(refs.keys())

        gold_array = [refs[f] for f in fields_sorted]
        gold_ints = []
        for f, v in zip(fields_sorted, gold_array):
            if is_binary_field(f):
                gold_ints.append(int(v))
            else:
                gold_ints.append(
                    int(math.copysign(math.floor(abs(v) + 0.5), v))
                    if v != 0 else 0
                )

        households.append({
            "scenario_id": sid,
            "canonical_identity": list(cid),
            "identity_string": identity_string(cid),
            "household_id": d["metadata"]["household_id"],
            "scenario_json": json.dumps(d, separators=(",", ":")),
            "fields": fields_sorted,
            "field_order_hash": sha256_hex("|".join(fields_sorted)),
            "n_fields": len(fields_sorted),
            "gold_array": gold_ints,
            "gold_minified": json.dumps(gold_ints, separators=(",", ":")),
            "refs_dict": refs,
        })

    return households


def select_challenge(pool, per_stratum):
    """Classify pool into strata and select per_stratum by hash rank."""
    for hh in pool:
        hh["stratum"] = classify_stratum(hh["refs_dict"])

    strata = {}
    for hh in pool:
        st = hh["stratum"]
        if st == "NONE":
            continue
        if st not in strata:
            strata[st] = []
        strata[st].append(hh)

    required = ["REFUNDABLE_CREDIT", "TAX_ONLY", "BENEFIT_ONLY", "TAX_AND_BENEFIT"]

    for st in required:
        if st not in strata:
            strata[st] = []

    for st in required:
        strata[st].sort(
            key=lambda hh: hash_rank_challenge(tuple(hh["canonical_identity"]))
        )

    selected = []
    insufficient = []
    for st in required:
        available = len(strata[st])
        if available < per_stratum:
            insufficient.append((st, available))
        else:
            selected.extend(strata[st][:per_stratum])

    stratum_counts = {
        st: min(len(strata[st]), per_stratum) for st in required
    }

    return selected, insufficient, stratum_counts


def serialize_household(hh):
    """Serialize a household for output, removing internal refs_dict."""
    out = dict(hh)
    out.pop("refs_dict", None)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build sealed R2.2 panels for Atlas W-D3"
    )
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--calibration-references", required=True)
    parser.add_argument("--prevalence-n", type=int, default=100)
    parser.add_argument("--prevalence-seed", type=int, default=2201)
    parser.add_argument("--challenge-pool-n", type=int, default=2000)
    parser.add_argument("--challenge-fallback-n", type=int, default=4000)
    parser.add_argument("--challenge-seed", type=int, default=2202)
    parser.add_argument("--challenge-per-stratum", type=int, default=50)
    parser.add_argument("--country", default="us")
    parser.add_argument("--program-set", default="headline")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    out_dir = REPO / "data" / "policybench"

    cal_manifest = Path(args.calibration_manifest)
    cal_refs = Path(args.calibration_references)
    if not cal_manifest.exists():
        print(f"FAIL: calibration manifest not found: {cal_manifest}",
              file=sys.stderr)
        sys.exit(1)
    if not cal_refs.exists():
        print(f"FAIL: calibration references not found: {cal_refs}",
              file=sys.stderr)
        sys.exit(1)

    cal_hash = sha256_file(cal_manifest)
    ref_hash = sha256_file(cal_refs)

    EXPECTED_CAL = (
        "f3d5e4d8c80949f50e639e7e80696c6c9c64a2122ab050aca2ee93c21b2747fb"
    )
    EXPECTED_REF = (
        "da51998841f41a7794c23d818a276aee53cd94767fbd08030d9ee293041e7aac"
    )

    if cal_hash != EXPECTED_CAL:
        print(f"FAIL: calibration manifest hash mismatch: {cal_hash}",
              file=sys.stderr)
        sys.exit(1)
    if ref_hash != EXPECTED_REF:
        print(f"FAIL: calibration references hash mismatch: {ref_hash}",
              file=sys.stderr)
        sys.exit(1)
    print("Calibration hashes verified.")

    c0_ids = load_c0_household_ids(cal_manifest)
    print(f"C0 household IDs loaded: {len(c0_ids)}")

    c0_households = load_c0_as_households(cal_manifest, cal_refs)

    print("\n--- Building field prior from C0 ---")
    field_prior = build_field_prior(c0_households)
    prior_json = json.dumps(field_prior, sort_keys=True, separators=(",", ":"))

    prior_path = out_dir / "r2_2_field_prior.json"
    prior_bytes = (prior_json + "\n").encode("utf-8")
    prior_path.write_bytes(prior_bytes)
    prior_hash = hashlib.sha256(prior_bytes).hexdigest()
    print(f"Field prior: {len(field_prior)} fields, hash={prior_hash[:16]}...")
    print(f"Saved: {prior_path}")

    print(f"\n--- Generating prevalence panel (n={args.prevalence_n}, "
          f"seed={args.prevalence_seed}) ---")
    prevalence = generate_panel(
        n=args.prevalence_n, seed=args.prevalence_seed,
        excluded_ids=c0_ids, country=args.country, year=args.year,
    )

    p_ids = {hh["household_id"] for hh in prevalence}
    excluded_for_c = c0_ids | p_ids
    print(f"Prevalence: {len(prevalence)} households generated")
    print(f"Excluding {len(excluded_for_c)} household IDs for challenge pool")

    overlap_c0_p = c0_ids & p_ids
    if overlap_c0_p:
        print(f"FAIL: {len(overlap_c0_p)} C0/P overlaps", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Generating challenge pool (n={args.challenge_pool_n}, "
          f"seed={args.challenge_seed}) ---")
    pool = generate_panel(
        n=args.challenge_pool_n, seed=args.challenge_seed,
        excluded_ids=excluded_for_c, country=args.country, year=args.year,
    )

    pool_hh_ids = {hh["household_id"] for hh in pool}
    overlap_pool = (c0_ids | p_ids) & pool_hh_ids
    if overlap_pool:
        print(f"FAIL: {len(overlap_pool)} pool/C0+P overlaps", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Selecting challenge households ---")
    challenge, insufficient, stratum_counts = select_challenge(
        pool, args.challenge_per_stratum,
    )

    if insufficient:
        print(f"WARNING: Insufficient strata in {args.challenge_pool_n}-pool:")
        for st, count in insufficient:
            print(f"  {st}: {count} < {args.challenge_per_stratum}")

        if args.challenge_fallback_n > args.challenge_pool_n:
            print(f"\n--- Fallback: regenerating at n={args.challenge_fallback_n} ---")
            pool = generate_panel(
                n=args.challenge_fallback_n, seed=args.challenge_seed,
                excluded_ids=excluded_for_c, country=args.country,
                year=args.year,
            )

            pool_hh_ids = {hh["household_id"] for hh in pool}
            overlap_pool = (c0_ids | p_ids) & pool_hh_ids
            if overlap_pool:
                print(f"FAIL: {len(overlap_pool)} fallback pool overlaps",
                      file=sys.stderr)
                sys.exit(1)

            challenge, insufficient, stratum_counts = select_challenge(
                pool, args.challenge_per_stratum,
            )

        if insufficient:
            print("FAIL: W-D3_R2.2_PANEL_INVALID - insufficient strata "
                  "after fallback", file=sys.stderr)
            sys.exit(1)

    print("Challenge stratum counts:")
    for st, count in sorted(stratum_counts.items()):
        print(f"  {st}: {count}")
    print(f"Total challenge: {len(challenge)}")

    c_hh_ids = {hh["household_id"] for hh in challenge}
    overlap_final = (c0_ids | p_ids) & c_hh_ids
    if overlap_final:
        print(f"FAIL: {len(overlap_final)} final C/C0+P overlaps",
              file=sys.stderr)
        sys.exit(1)

    print("\n--- ESS preflight ---")
    ess_c = ess_preflight(challenge, "C")

    print(f"Global eligibility rescue ESS: {ess_c['global_eligibility_rescue_ess']}")
    print(f"Global eligibility harm ESS:   {ess_c['global_eligibility_harm_ess']}")
    print(f"Global amount rescue ESS:      {ess_c['global_amount_rescue_ess']}")
    print(f"Global amount harm ESS:        {ess_c['global_amount_harm_ess']}")

    ess_pass = True
    for key in ["global_eligibility_rescue_ess", "global_eligibility_harm_ess",
                "global_amount_rescue_ess", "global_amount_harm_ess"]:
        if ess_c[key] < 60.0:
            print(f"FAIL: {key} = {ess_c[key]} < 60.0", file=sys.stderr)
            ess_pass = False

    print("\nPer-stratum all-field rescue ESS:")
    for st, val in sorted(ess_c["stratum_rescue_ess"].items()):
        status = "PASS" if val >= 35.0 else "FAIL"
        print(f"  {st}: {val} [{status}]")
        if val < 35.0:
            ess_pass = False

    if not ess_pass:
        print("FAIL: ESS preflight failed", file=sys.stderr)
        sys.exit(1)
    print("ESS preflight: PASS")

    print("\n--- Serializing panels ---")
    p_out = [serialize_household(hh) for hh in prevalence]
    c_out = [serialize_household(hh) for hh in challenge]

    p_json = json.dumps(p_out, separators=(",", ":"))
    c_json = json.dumps(c_out, separators=(",", ":"))

    p_path = out_dir / "r2_2_prevalence.json"
    c_path = out_dir / "r2_2_challenge.json"
    p_bytes = (p_json + "\n").encode("utf-8")
    c_bytes = (c_json + "\n").encode("utf-8")
    p_path.write_bytes(p_bytes)
    c_path.write_bytes(c_bytes)

    p_hash = hashlib.sha256(p_bytes).hexdigest()
    c_hash = hashlib.sha256(c_bytes).hexdigest()

    manifest = {
        "protocol_revision": "r2.2",
        "builder_version": "1.0.0",
        "country": args.country,
        "year": args.year,
        "program_set": args.program_set,
        "calibration": {
            "manifest_path": str(cal_manifest),
            "manifest_hash": cal_hash,
            "references_path": str(cal_refs),
            "references_hash": ref_hash,
            "n_households": len(c0_ids),
        },
        "prevalence": {
            "n": len(prevalence),
            "seed": args.prevalence_seed,
            "file": str(p_path.name),
            "hash": p_hash,
        },
        "challenge": {
            "n": len(challenge),
            "seed": args.challenge_seed,
            "pool_n": args.challenge_pool_n,
            "per_stratum": args.challenge_per_stratum,
            "stratum_counts": stratum_counts,
            "file": str(c_path.name),
            "hash": c_hash,
        },
        "field_prior": {
            "n_fields": len(field_prior),
            "file": str(prior_path.name),
            "hash": prior_hash,
        },
        "ess_preflight": ess_c,
        "disjointness_verified": True,
    }

    manifest_path = out_dir / "r2_2_panel_manifest.json"
    manifest_json = json.dumps(manifest, indent=2, sort_keys=False)
    manifest_path.write_bytes((manifest_json + "\n").encode("utf-8"))

    print(f"\nSaved: {p_path} ({p_hash[:16]}...)")
    print(f"Saved: {c_path} ({c_hash[:16]}...)")
    print(f"Saved: {manifest_path}")

    print("\n--- Summary ---")
    print(f"Prevalence: {len(prevalence)} households")
    print(f"Challenge:  {len(challenge)} households "
          f"({args.challenge_per_stratum}/stratum)")
    print(f"Field prior: {len(field_prior)} fields")
    print(f"ESS preflight: PASS")
    print(f"Disjointness: C0 ({len(c0_ids)}) | P ({len(prevalence)}) "
          f"| C ({len(challenge)}) verified disjoint")
    print("\nPanel builder complete. Seal hashes before smoke.")


if __name__ == "__main__":
    main()
