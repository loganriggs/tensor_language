#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_shared_axis_removal_live_and_beats_null_in_every_line pred_c_shared_axis_carries_half_of_the_line_specific_removal pred_d_shared_axis_removal_selective_in_every_line
"""One temporal axis for four decisions (v70): remove, at heads {9.1, 9.4, 15.5, 11.3}, the weight-only SHARED axis — the
first left singular vector of the four contrast directions [has−had, will−had, is−was, will−would] mapped by O_h^T — on the
fresh rows of the aspectual (v1 lexicon), temporal (v28), narrative (v43) and modal (v66) lines, and compare with each
line's own contrast removal. No data enters the axis (weights only, v69). Null: 16 random directions of the removed norm.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native                      <= 1e-4 on every line
    pred_b_shared_axis_removal_live_and_beats_null_in_every_line   fraction >= 0.10, positive >= 0.75, > max null, per line
    pred_c_shared_axis_carries_half_of_the_line_specific_removal   shared-axis damage >= 0.50 x the line's own-contrast removal
                                                          damage (same four heads), per line. Prior: unsure for aspectual (its
                                                          own set includes 8.1, excluded here).
    pred_d_shared_axis_removal_selective_in_every_line    three reader gates over the null, per line (was−were, who−which, night−day)

PRICE (registered maximum): four lines x 2 batches x (native + producer + own-contrast set + shared-axis set + 16 nulls) = 4 x 40
= 160 forwards; 0 backwards; 0 fits. Bar <= 176.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28
import run_narrative_dod_confirm_v43 as v43
import run_modal_dod_battery_v66 as v66

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_axis_v70_result.json"
CANDIDATE_ID = "corpus.temporal_axis_v70"
NULL_SEEDS = tuple(range(3101, 3117))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, HALF, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.50, 1e-4
FORWARDS_MAX = 176
HEADS = (L.Component("axis_attn9_1_4", 9, "attn", (1, 4), "final"), L.Component("axis_attn11_3", 11, "attn", (3,), "final"), L.Component("axis_attn15_5", 15, "attn", (5,), "final"))
CONTRASTS = ((" has", " had"), (" will", " had"), (" is", " was"), (" will", " would"))


def lines():
    asp = L.build_rows()
    return {"aspectual": (asp, L._single(" has"), L._single(" had")), "temporal": (v28.build(), v28.WILL, v28.HAD), "narrative": (v43.build(), L._single(" was"), L._single(" is")), "modal": (v66.build(), v66.WOULD, v66.WILL)}


def main() -> None:
    data = lines()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "lines": {k: len(v[0]) for k, v in data.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    fw = L.ManualForward(backend)
    # shared axis per head: first left singular vector of the 4 mapped contrasts
    axis = {}
    for comp in HEADS:
        for head in comp.heads:
            cols = []
            for a, b in CONTRASTS:
                d = L.readout_directions(model, (comp,), L._single(a), L._single(b))[(comp.name, head)].float()
                cols.append(d / d.norm())
            M = torch.stack(cols, dim=1)
            U, S, _ = torch.linalg.svd(M, full_matrices=False)
            axis[(comp.name, head)] = U[:, 0]
            print(comp.name, head, "singular values", [round(float(s), 3) for s in S])
    forwards, report = 0, {}
    for name, (rows, pos, neg) in data.items():
        native, n = v1._run_arm(fw, rows); forwards += n
        ref, n = v1._producer_native(backend, rows); forwards += n
        instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
        fw.directions = L.readout_directions(model, HEADS, pos, neg)
        own, n = v1._run_arm(fw, rows, components=HEADS, mode="project"); forwards += n
        own_s = L.summarize(rows, native, own)
        fw.directions = dict(axis)
        shared, n = v1._run_arm(fw, rows, components=HEADS, mode="project"); forwards += n
        sh = L.summarize(rows, native, shared)
        nulls = []
        for seed in NULL_SEEDS:
            arm, n = v1._run_arm(fw, rows, components=HEADS, mode="project_random", seed=seed); forwards += n
            nulls.append(L.summarize(rows, native, arm))
        null_max = max(s["target_damage_mean"] for s in nulls); null_moves = {rn: sum(s[f"{rn}_abs_move_mean"] for s in nulls) / len(nulls) for rn in L.UNRELATED}
        gates = {rn: sh[f"{rn}_abs_move_mean"] <= null_moves[rn] + GATE_RATIO * sh["target_damage_mean"] for rn in L.UNRELATED}
        report[name] = {"instrument": instrument, "own_damage": own_s["target_damage_mean"], "own_fraction": own_s["target_damage_fraction"], "shared": sh, "null_damage_max": null_max, "gates": gates,
                        "live": sh["target_damage_fraction"] >= LIVE_FRACTION and sh["target_damage_positive_fraction"] >= LIVE_POSITIVE and sh["target_damage_mean"] > null_max,
                        "half": sh["target_damage_mean"] >= HALF * own_s["target_damage_mean"], "selective": all(gates.values())}
        print(name, "own", round(own_s["target_damage_mean"], 3), "shared", round(sh["target_damage_mean"], 3), "fraction", round(sh["target_damage_fraction"], 3), "pos", sh["target_damage_positive_fraction"], "null_max", round(null_max, 3), "gates", gates)
    predictions = {"pred_a_instrument_replays_native": all(r["instrument"] <= INSTRUMENT_TOL for r in report.values()), "pred_b_shared_axis_removal_live_and_beats_null_in_every_line": all(r["live"] for r in report.values()),
                   "pred_c_shared_axis_carries_half_of_the_line_specific_removal": all(r["half"] for r in report.values()), "pred_d_shared_axis_removal_selective_in_every_line": all(r["selective"] for r in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "temporal_axis_result_v70", "candidate_id": CANDIDATE_ID, "lines": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
