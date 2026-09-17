#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_attn9_readout_removal_is_live_and_beats_null pred_c_attn9_readout_removal_is_selective_over_null pred_d_attn9_readout_direction_aligns_with_paired_delta pred_e_late_heads_beat_null pred_f_attn5_readout_removal_does_not_beat_null pred_g_attn9_readout_removal_carries_half_of_midpoint
"""Aspectual has/had definition-of-done battery, step 4: CUE-INDEPENDENT (weight-derived) removal.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: `..._dod_removal_v2` and
`..._dod_composition_v3` receipts.

WHY. v2 removed each head's write along the exact since/by paired delta; v3 showed that any
random coordinate piece of that delta is "selective" too, so the gate certified the cue-defined
delta rather than the component, and the removal needed the partner row (a port). This run
removes a direction defined by WEIGHTS ALONE: for head h, `v_h = O_h^T (u_has - u_had)`, the
pre-c_proj direction that the head's own output projection maps onto the has/had unembedding
contrast. No partner row, no fit, no activation enters the direction. Removal subtracts the
projection `(w . v_h) v_h` at the final query; the null subtracts a random 128-d direction of the
same removed norm (16 seeds). A fold-type cosine between `v_h` and the mean oriented paired delta
reports how much of the cue-carrying write lies on the readout direction.

Scope caveat registered now: `v_h` is the head's DIRECT readout direction. A head whose aspect
write is consumed by later modules (attention5's transport) need not write along it; failing
this removal is then evidence about the route, not evidence against the component.

ROWS / READERS: as v1-v3 (64 rows, sha 5ec7d32f..., opened by v1-v3 for removal only).
COMPONENTS: the four attention components (mlp4 has no per-head readout direction; excluded).

REGISTERED BARS: LIVE (>= 0.10 fraction, >= 0.75 positive), NULL (mean damage > max of 16 null
mean damages), GATE (unrelated mean|move| <= null mean|move| + 0.25 x damage) exactly as v2.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native            no-edit forward matches producer.native <= 1e-4
    pred_b_attn9_readout_removal_is_live_and_beats_null   prior: yes (v2 midpoint 0.72, 35%)
    pred_c_attn9_readout_removal_is_selective_over_null   prior: yes
    pred_d_attn9_readout_direction_aligns_with_paired_delta
                                                cosine(v_h, mean oriented delta) >= 0.30 for BOTH
                                                heads 1 and 4 (fold). Prior: uncertain.
    pred_e_late_heads_beat_null                 attn11 H3 and attn15 H5 readout removals each beat
                                                their null (they sit two and six blocks before the
                                                unembedding). Prior: yes.
    pred_f_attn5_readout_removal_does_not_beat_null
                                                attn5 H7/H1/H6/H8 readout removal does NOT beat its
                                                null (transport heads write for later readers, not
                                                the unembedding). Prior: yes; a pass here would
                                                mean attention5 also writes the answer directly.
    pred_g_attn9_readout_removal_carries_half_of_midpoint
                                                attn9 readout damage >= 0.50 x its v2 midpoint
                                                damage (0.7238). Prior: unsure.

PRICE (registered maximum): capture 2 + native 2 + producer 2 + 4 x (1 + 16) x 2 = 142
forwards; 0 backwards; 0 fits. Bar <= 160.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_readout_removal_v4_result.json"
V2 = ROOT / "circuits/followups/aspectual_anchor_dod_removal_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_readout_removal_v4"
EXPECTED_ROWS_SHA256 = v1.EXPECTED_ROWS_SHA256
NULL_SEEDS = tuple(range(301, 317))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, COSINE_MIN, HALF_RATIO, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.30, 0.50, 1e-4
FORWARDS_MAX = 160
COMPONENTS = tuple(c for c in L.COMPONENTS if c.kind == "attn")
TOKENS = {"has": 468, "had": 550}


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "components": [c.name for c in COMPONENTS], "null_seeds": list(NULL_SEEDS),
            "mode": "project_onto_O_h^T(u_has-u_had)", "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE,
                     "gate_ratio_over_null": GATE_RATIO, "cosine_min": COSINE_MIN,
                     "half_ratio": HALF_RATIO, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != EXPECTED_ROWS_SHA256:
        raise SystemExit("fresh rows changed; refusing to run against an unregistered panel")
    v2 = json.loads(V2.read_text())
    midpoint_attn9 = v2["components"]["attn9_h1_h4_final"]["midpoint"]["target_damage_mean"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    model = backend.model
    if int(L.ENCODING.encode(" has")[0]) != TOKENS["has"] or int(L.ENCODING.encode(" had")[0]) != TOKENS["had"]:
        raise SystemExit("token ids changed")
    fw.directions = L.readout_directions(model, COMPONENTS, TOKENS["has"], TOKENS["had"])

    store = {}
    for chunk in v1._batches(rows):
        store.update(fw.capture(chunk, COMPONENTS)); forwards += 1
    deltas = L.paired_deltas(store, rows, COMPONENTS)
    cosines = {}
    for c in COMPONENTS:
        for head in c.heads:
            m = L.mean_oriented_delta(deltas, rows, c, head)
            v = fw.directions[(c.name, head)]
            cosines[f"{c.name}:head{head}"] = float((m @ v) / (m.norm() * v.norm()))

    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument_err = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))

    report = {}
    for comp in COMPONENTS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        summary = L.summarize(rows, native, arm)
        nulls = []
        for seed in NULL_SEEDS:
            null_arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project_random", seed=seed)
            forwards += n
            nulls.append(L.summarize(rows, native, null_arm))
        null_damages = [s["target_damage_mean"] for s in nulls]
        null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
        live = (summary["target_damage_fraction"] >= LIVE_FRACTION and summary["target_damage_positive_fraction"] >= LIVE_POSITIVE)
        beats_null = summary["target_damage_mean"] > max(null_damages)
        gates = {name: summary[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * summary["target_damage_mean"]
                 for name in L.UNRELATED}
        report[comp.name] = {"project": summary, "live": live, "beats_null": beats_null, "gates": gates,
                             "selective": all(gates.values()), "null_damage_means": null_damages,
                             "null_damage_max": max(null_damages),
                             "null_damage_median": sorted(null_damages)[len(null_damages) // 2],
                             "null_unrelated_abs_move_mean": null_moves}
        print(comp.name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}),
              "live", live, "beats_null", beats_null, "gates", gates)
    a9, a11, a15, a5 = (report[k] for k in ("attn9_h1_h4_final", "attn11_h3_final", "attn15_h5_final", "attn5_h7_h1_h6_h8_final"))
    predictions = {
        "pred_a_instrument_replays_native": bool(instrument_err <= INSTRUMENT_TOL),
        "pred_b_attn9_readout_removal_is_live_and_beats_null": bool(a9["live"] and a9["beats_null"]),
        "pred_c_attn9_readout_removal_is_selective_over_null": bool(a9["selective"]),
        "pred_d_attn9_readout_direction_aligns_with_paired_delta": bool(
            cosines["attn9_h1_h4_final:head1"] >= COSINE_MIN and cosines["attn9_h1_h4_final:head4"] >= COSINE_MIN),
        "pred_e_late_heads_beat_null": bool(a11["beats_null"] and a15["beats_null"]),
        "pred_f_attn5_readout_removal_does_not_beat_null": bool(not a5["beats_null"]),
        "pred_g_attn9_readout_removal_carries_half_of_midpoint": bool(
            a9["project"]["target_damage_mean"] >= HALF_RATIO * midpoint_attn9),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_readout_removal_result_v4", "candidate_id": CANDIDATE_ID,
              "plan": _plan(rows), "instrument_max_abs_error": instrument_err, "cosines": cosines,
              "v2_midpoint_attn9_damage": midpoint_attn9, "components": report,
              "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "cosines": cosines, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
