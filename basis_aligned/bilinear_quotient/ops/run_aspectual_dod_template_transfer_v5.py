#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_two_of_three_constructions_capable pred_c_attn9_readout_removal_live_and_beats_null_in_every_capable_construction pred_d_attn9_readout_removal_selective_in_every_capable_construction pred_e_damage_fraction_transfers pred_f_attn5_never_beats_null
"""Aspectual has/had definition-of-done battery, step 5: TEMPLATE-VARYING transfer of the v4 removal.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: `..._dod_readout_removal_v4`
(attention9 H1/H4 weight-only readout removal: 0.70 logits, null max 0.012, selective, on the two
discovery-shaped constructions).

WHY. Every aspectual row so far shares one template shape per construction (cue, `last`, period
noun, `the`, agent). better_circuits §5: "new templates must vary the structural feature under
test"; communicating_results §3 flags any positional finding as confounded until a template-
varying control runs. The v4 removal direction is weight-only, so it can be applied unchanged to
constructions that move the cue, drop `last`, add an article, put the agent first, end on a
comma, or add a lexical tense cue (began/ended). If the component is the head's readout of an
aspect state rather than a template artifact, the removal transfers.

ROWS. 96 fresh-shaped rows (`build_template_rows`): three new constructions x 16 lexical groups
x {present, past}. Lexicon = the v1 fresh lexicon (opened for removal only, never for selection).
Constructions:
    ever_since_by_end        "Ever since the P the A"           / "By the end of the P the A"
    agent_first_comma        "The A, ever since the P,"          / "The A, by the end of the P,"
    clear_that_began_ended   "It is clear that since the P began the A" /
                             "It is clear that by the time the P ended the A"
CAPABILITY is scored per construction (both cue cells >= 0.85). Incapable constructions are
reported and excluded from causal scoring, never retuned.

COMPONENTS: attention9 H1/H4 (the claim), attention11 H3, attention15 H5, attention5 H7/H1/H6/H8
(controls), removed along `O_h^T (u_has - u_had)` at the final input token; null = random 128-d
direction of equal removed norm, 16 seeds. Bars LIVE / NULL / GATE as v2/v4.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   no-edit forward matches producer.native <= 1e-4 (96 rows)
    pred_b_two_of_three_constructions_capable   >= 2 constructions pass capability. Prior: the
                                       comma-final construction is the likely failure.
    pred_c_attn9_readout_removal_live_and_beats_null_in_every_capable_construction
    pred_d_attn9_readout_removal_selective_in_every_capable_construction
    pred_e_damage_fraction_transfers   in every capable construction the attn9 damage fraction is
                                       within [0.5, 1.5] x the v4 fraction (0.339)
    pred_f_attn5_never_beats_null      attention5 readout removal beats its null in NO capable
                                       construction (replicates v4's route reading)

PRICE (registered maximum): 96 rows in 3 batches; native 3 + producer 3 + 4 components x 17 arms
x 3 = 204 -> 210 forwards; 0 backwards; 0 fits. Bar <= 230.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_template_transfer_v5_result.json"
V4 = ROOT / "circuits/followups/aspectual_anchor_dod_readout_removal_v4_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_template_transfer_v5"
EXPECTED_ROWS_SHA256 = "dca4aa137b6add050fd694d496b40414b241ea86fe58086b8007bae3375917d3"
NULL_SEEDS = tuple(range(401, 417))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, CAPABILITY_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.85, 1e-4
TRANSFER_BAND = (0.5, 1.5)
FORWARDS_MAX = 230
COMPONENTS = tuple(c for c in L.COMPONENTS if c.kind == "attn")
TOKENS = {"has": 468, "had": 550}


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "constructions": sorted({r.construction for r in rows}),
            "components": [c.name for c in COMPONENTS], "null_seeds": list(NULL_SEEDS),
            "mode": "project_onto_O_h^T(u_has-u_had)", "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE,
                     "gate_ratio_over_null": GATE_RATIO, "capability_min": CAPABILITY_MIN,
                     "transfer_band": list(TRANSFER_BAND), "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = L.build_template_rows()
    if L.rows_sha256(rows) != EXPECTED_ROWS_SHA256:
        raise SystemExit("template rows changed; refusing to run against an unregistered panel")
    v4 = json.loads(V4.read_text())
    v4_fraction = v4["components"]["attn9_h1_h4_final"]["project"]["target_damage_fraction"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, COMPONENTS, TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument_err = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))

    constructions = sorted({r.construction for r in rows})
    capability = {}
    for c in constructions:
        for present in (True, False):
            cell = [1.0 if e["answer"] > e["foil"] else 0.0 for r, e in zip(rows, native)
                    if r.construction == c and r.present == present]
            capability[f"{c}/{'present' if present else 'past'}"] = sum(cell) / len(cell)
    capable = [c for c in constructions if all(capability[f"{c}/{k}"] >= CAPABILITY_MIN for k in ("present", "past"))]

    arms = {}
    for comp in COMPONENTS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        nulls = []
        for seed in NULL_SEEDS:
            null_arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project_random", seed=seed)
            forwards += n
            nulls.append(null_arm)
        arms[comp.name] = (arm, nulls)

    report = {}
    for c in constructions:
        idx = [i for i, r in enumerate(rows) if r.construction == c]
        sub_rows = [rows[i] for i in idx]
        sub_native = [native[i] for i in idx]
        report[c] = {"capable": c in capable, "components": {}}
        for comp in COMPONENTS:
            arm, nulls = arms[comp.name]
            summary = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])
            null_summaries = [L.summarize(sub_rows, sub_native, [nl[i] for i in idx]) for nl in nulls]
            null_damages = [s["target_damage_mean"] for s in null_summaries]
            null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in null_summaries) / len(null_summaries) for name in L.UNRELATED}
            live = (summary["target_damage_fraction"] >= LIVE_FRACTION and summary["target_damage_positive_fraction"] >= LIVE_POSITIVE)
            beats_null = summary["target_damage_mean"] > max(null_damages)
            gates = {name: summary[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * summary["target_damage_mean"] for name in L.UNRELATED}
            report[c]["components"][comp.name] = {"project": summary, "live": live, "beats_null": beats_null,
                                                  "gates": gates, "selective": all(gates.values()),
                                                  "null_damage_max": max(null_damages),
                                                  "null_damage_median": sorted(null_damages)[len(null_damages) // 2],
                                                  "null_unrelated_abs_move_mean": null_moves}
            print(c, comp.name, "capable", c in capable, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}),
                  "live", live, "beats_null", beats_null, "gates", gates)

    a9 = {c: report[c]["components"]["attn9_h1_h4_final"] for c in capable}
    a5 = {c: report[c]["components"]["attn5_h7_h1_h6_h8_final"] for c in capable}
    predictions = {
        "pred_a_instrument_replays_native": bool(instrument_err <= INSTRUMENT_TOL),
        "pred_b_two_of_three_constructions_capable": len(capable) >= 2,
        "pred_c_attn9_readout_removal_live_and_beats_null_in_every_capable_construction":
            bool(capable) and all(v["live"] and v["beats_null"] for v in a9.values()),
        "pred_d_attn9_readout_removal_selective_in_every_capable_construction":
            bool(capable) and all(v["selective"] for v in a9.values()),
        "pred_e_damage_fraction_transfers": bool(capable) and all(
            TRANSFER_BAND[0] * v4_fraction <= v["project"]["target_damage_fraction"] <= TRANSFER_BAND[1] * v4_fraction
            for v in a9.values()),
        "pred_f_attn5_never_beats_null": bool(capable) and not any(v["beats_null"] for v in a5.values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_template_transfer_result_v5", "candidate_id": CANDIDATE_ID,
              "plan": _plan(rows), "instrument_max_abs_error": instrument_err, "capability": capability,
              "capable_constructions": capable, "v4_attn9_fraction": v4_fraction, "constructions": report,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "capability": capability, "capable": capable,
                      "attn9_fraction": {c: round(v["project"]["target_damage_fraction"], 4) for c, v in a9.items()},
                      "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
