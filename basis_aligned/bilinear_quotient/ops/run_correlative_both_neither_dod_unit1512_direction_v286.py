#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v262 pred_b_direction_removal_live pred_c_beats_random_units pred_d_direction_removal_selective pred_e_direction_carries_half_of_whole_unit
"""Correlative both/neither DoD (v286): is the and / nor part of MLP-8 unit 1512 SEPARABLE along 16.8's reader direction? v262 / v264b: 1512 is a neither-context
detector at the final whose whole write costs 1.8% of the and - nor margin (panel) and 1.9% (natural) but moves tense / animacy / canonical readers 40x their null --
not selective. For head components the fix was a direction-restricted removal (v208: a shared head's number axis was selective where its whole slice was not).
Here only the component of 1512's write along r = the weight-only reader direction of 16.8 (unit-normalised, in residual space) is removed at the FINAL token: the
MLP-8 output is edited by  y <- y - (h_1512 (Down[:,1512] . r_hat)) r_hat. Null: the same direction-restricted removal for 16 seeded random MLP-8 units. Readers
will-would / who-which / night-day; gate move <= null mean + 0.25 x damage. Margin = and - nor for every row (panel rows v120).
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_replays_v262        the native pooled and - nor margin matches v262 (696.61) within relative 1e-3
    pred_b_direction_removal_live       the direction-restricted removal removes >= 0.005 of the pooled margin
    pred_c_beats_random_units           it removes more than every one of the 16 random units' direction-restricted removals
    pred_d_direction_removal_selective  each unrelated reader |move| <= null mean |move| + 0.25 x damage (the property the whole-unit edit failed)
    pred_e_direction_carries_half_of_whole_unit  it removes >= 0.50 x the whole-unit zeroing measured in the same run
PRICE (registered maximum): 3 batches x (native + whole-unit + direction + 16 random) = 57 forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_correlative_both_neither_dod_battery_v120 as g
import dod_battery, dod_units

L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")
ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/correlative_both_neither_dod_unit1512_direction_v286_result.json"
CANDIDATE_ID = "correlative_both_neither.and_vs_nor.dod_unit1512_direction_v286"
LAYER, UNIT, HEAD, V262_MARGIN, LIVE_MIN, GATE_RATIO, HALF, N_NULL, SEED, BATCH = 8, 1512, (16, 8), 696.61, 0.005, 0.25, 0.50, 16, 286, 32
FORWARDS_MAX = 60
PREDICTIONS = {"pred_a_baseline_replays_v262": "<= 1e-3", "pred_b_direction_removal_live": ">= 0.005", "pred_c_beats_random_units": "> 16 random", "pred_d_direction_removal_selective": "three gates", "pred_e_direction_carries_half_of_whole_unit": ">= 0.50 x whole"}


def main() -> None:
    rows, he, she, *_ = g.build()            # he = and-token (positive), she = nor-token (negative)
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}; readers["target"] = (he, she)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "reader_head": f"{HEAD[0]}.{HEAD[1]}", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"live_min": LIVE_MIN, "gate_ratio": GATE_RATIO, "half": HALF, "n_null": N_NULL, "seed": SEED}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); fw.backend = backend; blocks = model.transformer.h; mlp = blocks[LAYER].mlp
    comp = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, (HEAD,)).set_components())
    fw.directions = L.readout_directions(model, (comp,), he, she); r = L.reader_directions(model, comp, fw.directions)[HEAD[1]].float(); r_hat = (r / r.norm())
    Down = mlp.Down.weight.detach().float()
    import random
    rng = random.Random(SEED); pool = [j for j in range(Down.shape[1]) if j != UNIT]; null_units = [rng.choice(pool) for _ in range(N_NULL)]

    def run(chunk, mode, unit):
        """mode None = native; 'zero' = zero unit's hidden value at the final; 'dir' = remove the unit's write along r_hat at the final."""
        tokens = fw._tokens(chunk); fin = [rw.final for rw in chunk]; idx = torch.arange(len(chunk))
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == LAYER and mode is not None:
                    h = dod_units.hidden(model, mlp, xin); y = mlp.Down(h) + mlp.Down_bias
                    if mode == "zero":
                        for i, p in enumerate(fin): y[i, p] = y[i, p] - h[i, p, unit] * Down[:, unit].to(y.dtype)
                    else:
                        coef = float(Down[:, unit] @ r_hat.to(Down.device)); rh = r_hat.to(y.device, y.dtype)
                        for i, p in enumerate(fin): y[i, p] = y[i, p] - (h[i, p, unit] * coef) * rh
                    x = x + y
                else:
                    x = x + block.mlp(xin)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30); lg = logits[idx, torch.tensor(fin)].float()
        return [{name: float(lg[i, a] - lg[i, b]) for name, (a, b) in readers.items()} for i in range(len(chunk))]

    conds = [("native", None, UNIT), ("zero", "zero", UNIT), ("dir", "dir", UNIT)] + [(f"null{k}", "dir", u) for k, u in enumerate(null_units)]
    outs = {name: [] for name, _, _ in conds}; forwards = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, mode, unit in conds:
            outs[name].extend(run(chunk, mode, unit)); forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    pooled = lambda name: sum(outs[name][i]["target"] - outs[name][j]["target"] for i, j in pairs)
    native = pooled("native"); damage = {name: (native - pooled(name)) / native for name in outs if name != "native"}
    unrel = [k for k in readers if k != "target"]
    moves = {name: {k: sum(abs(a[k] - n[k]) for n, a in zip(outs["native"], outs[name])) / len(rows) for k in unrel} for name in outs if name != "native"}
    null_moves = {k: sum(moves[f"null{q}"][k] for q in range(N_NULL)) / N_NULL for k in unrel}; null_max = max(damage[f"null{q}"] for q in range(N_NULL))
    dmg_row = damage["dir"] * native / len(pairs); gates = {k: moves["dir"][k] <= null_moves[k] + GATE_RATIO * dmg_row for k in unrel}
    report = {"native_pooled": native, "damage_whole_unit": damage["zero"], "damage_direction": damage["dir"], "null_direction_max": null_max, "moves_direction": moves["dir"], "moves_whole_unit": moves["zero"], "null_moves": null_moves, "gates": gates, "null_units": null_units}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k not in ("moves_direction", "moves_whole_unit", "null_moves", "null_units")}); print("moves dir", {k: round(v, 4) for k, v in moves["dir"].items()}, "whole", {k: round(v, 4) for k, v in moves["zero"].items()}, "null", {k: round(v, 4) for k, v in null_moves.items()})
    predictions = {"pred_a_baseline_replays_v262": abs(native - V262_MARGIN) / V262_MARGIN <= 1e-3, "pred_b_direction_removal_live": damage["dir"] >= LIVE_MIN, "pred_c_beats_random_units": damage["dir"] > null_max, "pred_d_direction_removal_selective": all(gates.values()), "pred_e_direction_carries_half_of_whole_unit": damage["dir"] >= HALF * damage["zero"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "correlative_both_neither_dod_unit1512_direction_result_v286", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
