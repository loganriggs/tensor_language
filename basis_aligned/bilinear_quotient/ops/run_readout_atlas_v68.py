#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_all_lines_instrument_replay pred_b_majority_of_capable_lines_have_a_live_null_beating_top4 pred_c_auxiliary_lines_recur_the_two_families pred_d_at_least_one_new_family_appears
"""Readout atlas (v68): the reuse census over every distinct screened behaviour family not yet censused.

Lane: Claude circuit lane (review-5 decision). Library: `dod_reuse_census.run` per line (blind 162-head weight-only readout
sweep on the line's authored A1/A2 contexts, top-4 set + 16 norm-matched nulls + three readers, overlap with the auxiliary
family). Lines: `atlas_lines_v68.json` (one representative per task family; lexical swarms reduced to one each). Each line
writes its own receipt `circuits/followups/atlas_<task>_v68_result.json`; this driver writes the atlas summary. Lines
whose rows fail to build or whose capability is < 0.85 are recorded as such, not skipped silently.

PREDICTIONS (scored as written; failures preserved)
    pred_a_all_lines_instrument_replay          every line's no-edit forward replays producer.native <= 1e-4
    pred_b_majority_of_capable_lines_have_a_live_null_beating_top4   > 50% of capable lines: top-4 fraction >= 0.10, positive
                                                >= 0.75, > max null
    pred_c_auxiliary_lines_recur_the_two_families   every capable line whose vocabulary is an auxiliary pair (was/were, is/was,
                                                has/had, have/has, will/would/had) overlaps the family {8.1, 9.1, 9.4, 11.3, 15.5}
                                                in >= 2 of its top four
    pred_d_at_least_one_new_family_appears      some head set of >= 3 heads recurs as top-4 members in >= 3 non-auxiliary lines

PRICE (registered maximum): per line 181 forwards x batches (rows/32); total registered as the sum over lines (printed at
dry run); the driver refuses to exceed it.
"""
from __future__ import annotations
from datetime import datetime, timezone
import importlib, itertools, json, os, time, collections
from pathlib import Path
import aspectual_dod_lib as L
import dod_reuse_census as R
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/readout_atlas_v68_result.json"
CANDIDATE_ID = "corpus.readout_atlas_v68"
LINES = json.loads((Path(__file__).resolve().parent / "atlas_lines_v68.json").read_text())
AUX = {" was", " were", " is", " has", " had", " have", " will", " would"}


def line_rows(entry):
    m = importlib.import_module(entry["module"])
    pos, neg = entry["vocabulary"]
    rows, pid, nid = R.rows_from_candidate(m, pos, neg, entry["task"].split(".")[0])
    return rows, pid, nid


def main() -> None:
    plans, total = [], 0
    for e in LINES:
        try:
            rows, pid, nid = line_rows(e)
        except Exception as ex:
            plans.append({"task": e["task"], "error": str(ex)[:120]}); continue
        batches = (len(rows) + v1.BATCH - 1) // v1.BATCH
        plans.append({"task": e["task"], "rows": len(rows), "forwards": batches * R.FORWARDS_PER_BATCH}); total += batches * R.FORWARDS_PER_BATCH
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "lines": len(LINES), "buildable": sum(1 for p in plans if "rows" in p), "forwards_max": total, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    summary, forwards = [], 0
    for e, p in zip(LINES, plans):
        if "rows" not in p:
            summary.append({"task": e["task"], "status": "rows_failed", "error": p["error"]}); continue
        rows, pid, nid = line_rows(e)
        out_name = f"atlas_{e['task'].split('.')[0]}_v68_result.json"
        if not (ROOT / "circuits/followups" / out_name).exists():
            R.run(f"{e['task']}.atlas_v68", out_name, rows, pid, nid)
        r = json.loads((ROOT / "circuits/followups" / out_name).read_text())
        forwards += r["forwards"]
        capable = all(v >= 0.85 for v in r["capability"].values())
        summary.append({"task": e["task"], "vocabulary": e["vocabulary"], "status": "ok", "capable": capable, "instrument": r["instrument_max_abs_error"], "top4": r["set"],
                        "top4_damages": [d for _, d in r["sweep_top20"][:4]], "fraction": r["joint"]["target_damage_fraction"], "positive": r["joint"]["target_damage_positive_fraction"],
                        "beats_null": r["joint"]["target_damage_mean"] > r["null_damage_max"], "selective": all(r["gates"].values()), "family_overlap": r["family_overlap"]})
        print(e["task"], "capable", capable, "top4", r["set"], "fraction", round(r["joint"]["target_damage_fraction"], 3), "sel", all(r["gates"].values()))
    ok = [s for s in summary if s["status"] == "ok"]
    capable = [s for s in ok if s["capable"]]
    live = [s for s in capable if s["fraction"] >= 0.10 and s["positive"] >= 0.75 and s["beats_null"]]
    aux = [s for s in capable if set(s["vocabulary"]) <= AUX]
    non_aux = [s for s in capable if not set(s["vocabulary"]) <= AUX]
    counts = collections.Counter(h for s in non_aux for h in s["top4"])
    triples = collections.Counter()
    for s in non_aux:
        for tri in itertools.combinations(sorted(s["top4"]), 3):
            triples[tri] += 1
    new_family = [(list(t), c) for t, c in triples.items() if c >= 3]
    predictions = {"pred_a_all_lines_instrument_replay": all(s["instrument"] <= 1e-4 for s in ok),
                   "pred_b_majority_of_capable_lines_have_a_live_null_beating_top4": bool(capable) and len(live) > len(capable) / 2,
                   "pred_c_auxiliary_lines_recur_the_two_families": bool(aux) and all(len(s["family_overlap"]) >= 2 for s in aux),
                   "pred_d_at_least_one_new_family_appears": bool(new_family)}
    if forwards > total:
        raise SystemExit(f"price exceeded: {forwards} > {total}")
    OUT.write_text(json.dumps({"schema": "readout_atlas_result_v68", "candidate_id": CANDIDATE_ID, "lines": summary, "counts_non_aux_heads": counts.most_common(15), "recurring_triples_non_aux": new_family,
                               "n_lines": len(LINES), "n_ok": len(ok), "n_capable": len(capable), "n_live": len(live), "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "n_lines": len(LINES), "n_capable": len(capable), "n_live": len(live), "top_heads_non_aux": counts.most_common(10), "recurring_triples": new_family, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
