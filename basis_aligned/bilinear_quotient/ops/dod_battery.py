#!/usr/bin/env python3
# BQGATE: LIBRARY -- generic fresh-row set battery for one readout component (review-4/5 efficiency item).
"""One `LineSpec` -> the standard fresh-row battery a readout component gets after its sweep:
native + producer replay (instrument), capability per (construction x side), set removal along weight-only readout
directions with 16 norm-matched nulls and three readers, singles + additivity, zero / keep-only / 16 random keeps, and
an optional frozen-fraction band. Replaces the near-identical bodies of run_temporal_dod_removal_v28 /
run_narrative_dod_confirm_v43 / run_number_dod_battery_v55 / run_modal_dod_battery_v66 for future lines.

A LineSpec declares: rows (list[L.Row]), positive/negative token ids, the head set as (layer, head) pairs, reader names
(must exist in L.READERS at row-build time), an optional frozen fraction, and bars. PRICE is registered per 32-row batch:
41 forwards (native, producer, set, 16 nulls, 4 singles [or |set|], zero, keep, 16 random keeps); the runner refuses to
write above batches x (2 + 1 + 16 + n_singles + 2 + 16).

GATE NOTE: `ops/gate.py` discovers prediction keys from literal dict keys in the RUNNER file, so a LineSpec runner must
carry a literal `PREDICTIONS = {"pred_a_instrument_replays_native": ..., ...}` dict mirroring the keys scored here
(see run_pronoun_gender_dod_battery_v71.py); the header line lists the same keys."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LineSpec:
    candidate_id: str
    out_name: str
    rows: list
    positive_id: int
    negative_id: int
    heads: tuple                      # ((layer, head), ...)
    frozen_fraction: float | None = None
    band: float = 0.15
    null_seeds: tuple = tuple(range(3001, 3017))
    keep_seeds: tuple = tuple(range(3051, 3067))
    bars: dict = field(default_factory=lambda: {"capability_min": 0.85, "live_fraction": 0.10, "live_positive": 0.75, "gate_ratio_over_null": 0.25, "add_ratio": 0.25, "retain_min": 0.70, "random_max": 0.30, "instrument_tol": 1e-4})

    def singles(self):
        return tuple(L.Component(f"attn{l}_h{h}_final", l, "attn", (h,), "final") for l, h in self.heads)

    def set_components(self):
        by_layer = {}
        for l, h in self.heads:
            by_layer.setdefault(l, []).append(h)
        return tuple(L.Component(f"set_attn{l}_" + "_".join(map(str, hs)), l, "attn", tuple(hs), "final") for l, hs in sorted(by_layer.items()))


def run(spec: LineSpec) -> None:
    rows, B = spec.rows, spec.bars
    batches = (len(rows) + v1.BATCH - 1) // v1.BATCH
    forwards_max = batches * (2 + 1 + len(spec.null_seeds) + len(spec.heads) + 2 + len(spec.keep_seeds))
    out = ROOT / f"circuits/followups/{spec.out_name}"
    plan = {"candidate_id": spec.candidate_id, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(spec.heads), "frozen_fraction": spec.frozen_fraction, "band": spec.band,
            "forwards_max": forwards_max, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": B}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    singles_c, set_c = spec.singles(), spec.set_components()
    fw.directions = L.readout_directions(backend.model, singles_c + set_c, spec.positive_id, spec.negative_id)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    constructions = sorted({r.construction for r in rows})
    capability = {}
    for c in constructions:
        for side in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c and r.present == side]
            capability[f"{c}/{'positive' if side else 'negative'}"] = sum(cell) / len(cell) if cell else None
    joint, n = v1._run_arm(fw, rows, components=set_c, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    per_c = {c: L.summarize([r for r in rows if r.construction == c], [native[i] for i, r in enumerate(rows) if r.construction == c], [joint[i] for i, r in enumerate(rows) if r.construction == c])["target_damage_fraction"] for c in constructions}
    nulls = []
    for seed in spec.null_seeds:
        arm, n = v1._run_arm(fw, rows, components=set_c, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls); null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + B["gate_ratio_over_null"] * js["target_damage_mean"] for name in L.UNRELATED}
    singles = {}
    for comp in singles_c:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        singles[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    gap = abs(js["target_damage_mean"] - sum(singles.values())); bar = B["add_ratio"] * min(singles.values())
    zero, n = v1._run_arm(fw, rows, components=set_c, mode="zero"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    keep, n = v1._run_arm(fw, rows, components=set_c, mode="keep_only"); forwards += n
    retention = 1.0 - L.summarize(rows, native, keep)["target_damage_mean"] / zd if zd else None
    random_ret = []
    for seed in spec.keep_seeds:
        arm, n = v1._run_arm(fw, rows, components=set_c, mode="keep_only_random", seed=seed); forwards += n
        random_ret.append(1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd if zd else None)
    print("capability", capability, "fractions", {k: round(v, 3) for k, v in per_c.items()}, "joint", round(js["target_damage_mean"], 3), "pos", js["target_damage_positive_fraction"], "null_max", round(null_max, 4), "gates", gates)
    print("singles", {k: round(v, 3) for k, v in singles.items()}, "gap/bar", round(gap, 4), round(bar, 4), "keep", retention, "random keep max", max(random_ret))
    predictions = {"pred_a_instrument_replays_native": instrument <= B["instrument_tol"], "pred_b_native_capability": all(v is not None and v >= B["capability_min"] for v in capability.values()),
                   "pred_c_set_live_and_beats_null": js["target_damage_fraction"] >= B["live_fraction"] and js["target_damage_positive_fraction"] >= B["live_positive"] and js["target_damage_mean"] > null_max,
                   "pred_d_set_selective": all(gates.values()), "pred_e_set_is_additive": gap <= bar,
                   "pred_f_keep_only_retains_most": retention is not None and retention >= B["retain_min"] and all(r is not None and r <= B["random_max"] for r in random_ret)}
    if spec.frozen_fraction is not None:
        predictions["pred_g_set_fraction_within_band_of_frozen"] = all(abs(v - spec.frozen_fraction) <= spec.band for v in per_c.values())
    if forwards > forwards_max:
        raise SystemExit(f"price exceeded: {forwards} > {forwards_max}")
    out.write_text(json.dumps({"schema": "dod_battery_result_v1", "candidate_id": spec.candidate_id, "plan": plan, "instrument_max_abs_error": instrument, "capability": capability, "joint": js, "fractions": per_c,
                               "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates, "singles": singles, "additivity": {"gap": gap, "bar": bar},
                               "keep_only": {"zero_damage": zd, "retention": retention, "random_retention": random_ret}, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))
