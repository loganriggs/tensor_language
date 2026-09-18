#!/usr/bin/env python3
# BQGATE: LIBRARY -- machine-readable registry of the readout collection, built from receipts on disk (review-9 collection item).
"""Writes claude_hourly_review/READOUT_COLLECTION.json: one entry per batteried line with family, heads, contrast and the numbers pulled from
its receipts (fresh battery, random-set null, response census, natural FineWeb / Pile). Numbers are read, never typed. Usage:
python ops/collection_registry.py"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "circuits/followups"
OUT = ROOT.parent / "claude_hourly_review/READOUT_COLLECTION.json"

LINES = [
    # family, line, contrast, heads, battery, null, census, natural, pile, scorecard
    ("temporal", "aspectual has/had", ("has", "had"), ["8.1", "9.1", "9.4"], "aspectual_anchor_dod_triple_v8_result.json", "aspectual_anchor_dod_random_head_set_null_v25_result.json", "aspectual_anchor_dod_response_census_v6_result.json", "aspectual_anchor_dod_natural_v20_result.json", "aspectual_anchor_dod_pile_ood_v26_result.json", "ASPECTUAL_DOD_SCORECARD.md"),
    ("temporal", "temporal will/had", ("will", "had"), ["11.3", "9.1", "15.5", "9.4"], "temporal_auxiliary_dod_removal_v28_result.json", "temporal_auxiliary_dod_random_set_null_v31_result.json", "temporal_auxiliary_dod_mlp_response_census_v48_result.json", "temporal_auxiliary_dod_natural_v34_result.json", "temporal_auxiliary_dod_natural_v34_result.json", "TEMPORAL_DOD_SCORECARD.md"),
    ("temporal", "narrative was/is", ("was", "is"), "FROM:narrative_tense_dod_sweep_and_set_v42_result.json", "narrative_tense_dod_confirm_v43_result.json", "narrative_tense_dod_random_set_null_v45_result.json", None, None, None, "NARRATIVE_DOD_SCORECARD.md"),
    ("temporal", "modal would/will", ("would", "will"), ["9.4", "11.3", "9.1", "15.5"], "modal_remoteness_dod_battery_v66_result.json", "modal_remoteness_dod_null_and_templates_v67_result.json", None, None, None, "MODAL_DOD_SCORECARD.md"),
    ("number", "lexical were/was", ("were", "was"), ["11.3", "5.7", "7.8", "9.7"], "number_family_dod_battery_v55_result.json", "number_family_dod_random_set_null_v60_result.json", None, "lexical_number_dod_natural_v138_result.json", "lexical_number_dod_pile_v139_result.json", "NUMBER_DOD_SCORECARD.md"),
    ("number", "perfect have/has", ("have", "has"), ["11.3", "7.8", "5.3", "9.7"], "perfect_number_dod_battery_v97_result.json", "perfect_number_dod_random_set_null_v98_result.json", "perfect_number_dod_response_census_v99_result.json", "perfect_number_dod_natural_v100_result.json", "perfect_number_dod_pile_v101_result.json", "PERFECT_NUMBER_DOD_SCORECARD.md"),
    ("pronoun", "pronoun gender he/she", ("he", "she"), ["10.1", "9.6", "12.4", "15.1"], "pronoun_gender_dod_battery_v71_result.json", "pronoun_gender_dod_random_set_null_v72_result.json", "pronoun_gender_dod_response_census_v74_result.json", "pronoun_gender_dod_natural_v73_result.json", "pronoun_gender_dod_pile_v75_result.json", "PRONOUN_GENDER_DOD_SCORECARD.md"),
    ("pronoun", "pronoun number they/he", ("they", "he"), ["9.6", "12.4", "15.1", "10.5"], "pronoun_number_dod_battery_v76_result.json", "pronoun_number_dod_random_set_null_v79_result.json", "pronoun_number_dod_response_census_v84_result.json", "pronoun_number_dod_natural_v77_result.json", "pronoun_number_dod_pile_v78_result.json", "PRONOUN_NUMBER_DOD_SCORECARD.md"),
    ("selection", "adjective preposition in/of", ("in", "of"), ["8.8", "6.3", "13.8", "7.8"], "selection_dod_battery_v85_result.json", "selection_dod_random_set_null_v87_result.json", "selection_dod_response_census_v88_result.json", "selection_inof_dod_natural_v91_result.json", "selection_inof_dod_pile_v92_result.json", "SELECTION_DOD_SCORECARD.md"),
    ("selection", "verb particle up/down", ("up", "down"), ["13.8", "14.8", "7.8", "8.8"], "selection_particle_dod_battery_v86_result.json", "selection_particle_dod_random_set_null_v89_result.json", "selection_particle_dod_response_census_v90_result.json", "selection_updown_dod_natural_v93_result.json", "selection_updown_dod_pile_v94_result.json", "SELECTION_PARTICLE_DOD_SCORECARD.md"),
    ("correlative", "correlative either/not", ("or", "but"), ["16.8", "14.8", "7.8", "8.1"], "correlative_either_not_dod_battery_v119_result.json", "correlative_either_not_dod_random_set_null_v121_result.json", "correlative_either_not_dod_response_census_v122_result.json", "correlative_either_not_dod_natural_v125_result.json", "correlative_either_not_dod_pile_v126_result.json", "CORRELATIVE_EITHER_NOT_DOD_SCORECARD.md"),
    ("correlative", "correlative both/neither", ("and", "nor"), ["8.1", "7.8", "16.8", "14.8"], "correlative_both_neither_dod_battery_v120_result.json", "correlative_both_neither_dod_random_set_null_v123_result.json", "correlative_both_neither_dod_response_census_v124_result.json", "correlative_both_neither_dod_natural_v127_result.json", "correlative_both_neither_dod_pile_v128_result.json", "CORRELATIVE_BOTH_NEITHER_DOD_SCORECARD.md"),
    ("correlative", "correlative either/neither", ("or", "nor"), ["14.8", "8.1", "16.8", "5.7"], "correlative_either_neither_dod_battery_v147_result.json", None, None, None, None, "CORRELATIVE_EITHER_NEITHER_DOD_SCORECARD.md"),
    ("person", "reflexive person I/you", ("myself", "yourself"), ["8.1", "13.1", "10.5", "15.1"], "person_reflexive_dod_battery_v104_result.json", "person_reflexive_dod_random_set_null_v106_result.json", "person_reflexive_dod_response_census_v107_result.json", "person_reflexive_dod_natural_v110_result.json", "person_reflexive_dod_pile_v111_result.json", "PERSON_REFLEXIVE_DOD_SCORECARD.md"),
    ("person", "object control me/you", ("myself", "yourself"), ["13.1", "8.1", "10.5", "15.1"], "person_object_control_dod_battery_v105_result.json", "person_object_control_dod_random_set_null_v108_result.json", "person_object_control_dod_response_census_v109_result.json", None, None, "PERSON_OBJECT_CONTROL_DOD_SCORECARD.md"),
]


def load(name):
    p = F / name if name else None
    return json.loads(p.read_text()) if p and p.exists() else None


def battery(r):
    if not r: return None
    j = r.get("joint") or r.get("pooled_triple") or {}
    out = {"fraction": j.get("target_damage_fraction"), "positive": j.get("target_damage_positive_fraction"), "null_max": r.get("null_damage_max"), "predictions": r.get("predictions"), "forwards": r.get("forwards")}
    if "keep_only" in r: out["keep_only_retention"] = r["keep_only"].get("retention")
    if "additivity" in r: out["additivity"] = r["additivity"]
    return out


def null(r):
    if not r or "random" not in r or "set" not in r or not isinstance(r["set"], dict): return {"predictions": r.get("predictions")} if r else None
    return {"set_damage": r["set"].get("target_damage_mean"), "random_max": max(x["damage"] for x in r["random"]), "random_live": sum(x["live"] for x in r["random"]), "predictions": r.get("predictions")}


def held(r):
    p = (r or {}).get("predictions") or {}
    return f"{sum(1 for v in p.values() if v is True)}/{sum(1 for v in p.values() if v is not None)}" if p else None


def census(r):
    if not r: return None
    s = r.get("summary", {})
    if "pooled" in s: s = s["pooled"]
    if "direct_share_of_linear" not in s: return {"predictions": r.get("predictions"), "held": held(r), "note": "different census schema (MLP response); see receipt"}
    return {"direct_share": s.get("direct_share_of_linear"), "downstream_net": s.get("downstream_net_mean"), "remainder": s.get("remainder"), "predictions": r.get("predictions")}


def natural(r):
    if not r: return None
    rep = r.get("report", {}).get("congruent") if isinstance(r.get("report"), dict) else None
    if rep: m = rep["removal"]; return {"congruent_damage": m["target_damage_mean"], "congruent_native": m["target_native_mean_margin"], "positive": m["target_damage_positive_fraction"], "null_max": rep["null_damage_max"], "predictions": r.get("predictions"), "held": held(r)}
    return {"predictions": r.get("predictions"), "held": held(r), "note": "earlier natural-row schema (shift-based); see receipt"}


def main():
    entries = []
    for fam, line, contrast, heads, b, n, c, nat, pile, card in LINES:
        rb = load(b)
        if isinstance(heads, str) and heads.startswith("FROM:"):
            src = load(heads[5:]); heads = [h.replace("attn", "").replace("_h", ".") for h in (src or {}).get("set", [])] or None
        entries.append({"family": fam, "line": line, "contrast": list(contrast), "heads": heads or (rb or {}).get("plan", {}).get("heads"), "scorecard": f"claude_hourly_review/{card}",
                        "fresh_battery": {"receipt": b, **(battery(rb) or {})}, "random_set_null": {"receipt": n, **(null(load(n)) or {})}, "response_census": {"receipt": c, **(census(load(c)) or {})},
                        "natural_fineweb": {"receipt": nat, **(natural(load(nat)) or {})}, "natural_pile": {"receipt": pile, **(natural(load(pile)) or {})}})
    OUT.write_text(json.dumps({"schema": "readout_collection_v1", "built_from": "circuits/followups receipts", "families": sorted({e["family"] for e in entries}), "lines": entries}, indent=1, sort_keys=True) + "\n")
    if "--md" in sys.argv:
        import sys as _s
        out = Path(_s.argv[_s.argv.index("--md") + 1]); lines = ["| family | line | heads | fresh fraction | direct share | natural FineWeb / Pile (congruent fraction, or predictions held) |", "|---|---|---|---|---|---|"]
        for e in entries:
            fb, cs, nf, npi = e["fresh_battery"], e["response_census"], e["natural_fineweb"], e["natural_pile"]
            f = lambda x: "—" if x is None else f"{x:.2f}"
            def nat(x):
                if x.get("congruent_damage") is not None: return f"{x['congruent_damage'] / x['congruent_native']:.0%}"
                return f"{x['held']} preds" if x.get("held") else "—"
            lines.append(f"| {e['family']} | {e['line']} | {', '.join(e['heads'] or [])} | {f(fb.get('fraction'))} | {f(cs.get('direct_share')) if cs.get('direct_share') is not None else (cs.get('held') + ' preds' if cs.get('held') else '—')} | {nat(nf)} / {nat(npi)} |")
        out.write_text("\n".join(lines) + "\n"); print("wrote", out)
    for e in entries:
        fb, cs = e["fresh_battery"], e["response_census"]
        print(f"{e['family']:9s} {e['line']:30s} heads {e['heads']} fresh {fb.get('fraction') and round(fb['fraction'],2)} direct {cs.get('direct_share') and round(cs['direct_share'],2)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
