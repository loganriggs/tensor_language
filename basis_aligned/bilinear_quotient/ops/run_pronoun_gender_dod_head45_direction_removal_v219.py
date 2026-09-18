#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_whole_slice_damages pred_b_direction_removal_damages pred_c_direction_beats_random_directions pred_d_direction_removal_selective pred_e_direction_carries_half_of_full_zeroing
"""Pronoun gender he/she DoD (v219): does the GENDER ride a separable direction of head 4.5's copy, as number does (v208 / v209: -4.3% of the margin along
one 128-d direction, selective, 5x a norm-matched random direction; v218: 4.5 carries 77% of block 4 into the male detector at the verb)? Remove only the
component of 4.5's verb slice along d = O_{4.5}^T r, r = the male detector 3152's product gradient at the verb (per row, weights + native trace, scaled to block 8);
compare with zeroing the whole slice at the verb; null = 16 random 128-d directions each removing exactly the norm the gender direction removed on that row.
Readers will-would / who-which / night-day (gender-free); gate move <= null mean + 0.25 x damage. Margin = he - she for every row.
PREDICTIONS (scored as written; failures preserved; priors from v206-v209 on the number line)
    pred_a_whole_slice_damages                zeroing 4.5's whole slice at the verb removes >= 0.02 of the pooled he - she margin (number: 0.077)
    pred_b_direction_removal_damages          the gender-direction removal removes >= 0.01 (number: 0.043)
    pred_c_direction_beats_random_directions  its damage exceeds every one of the 16 norm-matched random directions
    pred_d_direction_removal_selective        each unrelated reader |move| <= matched-null mean |move| + 0.25 x damage
    pred_e_direction_carries_half_of_full_zeroing  its damage >= 0.50 x the whole-slice damage
PRICE (registered maximum): 3 batches x (native + whole slice + direction + 16 random + 1 trace) = 60 forwards; 0 backwards; 0 fits. Bar <= 63.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_head45_direction_removal_v219_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_head45_direction_removal_v219"
BLOCK, HEAD, U829, WHOLE_MIN, DROP_MIN, GATE_RATIO, HALF, N_NULL, SEED, BATCH = 4, 5, 3152, 0.02, 0.01, 0.25, 0.50, 16, 219, 32
FORWARDS_MAX = 63
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
PREDICTIONS = {"pred_a_whole_slice_damages": ">= 0.02", "pred_b_direction_removal_damages": ">= 0.02", "pred_c_direction_beats_random_directions": "> 16 random", "pred_d_direction_removal_selective": "three gates", "pred_e_direction_carries_half_of_full_zeroing": ">= 0.50 x full"}
L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns); verb_of = lambda row: noun_of(row) + 1
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "block": BLOCK, "head": HEAD, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"drop_min": DROP_MIN, "gate_ratio": GATE_RATIO, "half": HALF, "n_null": N_NULL, "seed": SEED}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); fw.backend = backend; blocks = model.transformer.h
    readers = {name: (L._single(a_), L._single(b_)) for name, (a_, b_) in L.READERS.items()}; readers["target"] = (he, she)
    mlp8 = blocks[8].mlp; Lrow, Rrow = mlp8.Left.weight.detach().float()[U829], mlp8.Right.weight.detach().float()[U829]
    Wo = blocks[BLOCK].attn.c_proj.weight.detach().float()[:, HEAD * L.HEAD_DIM:(HEAD + 1) * L.HEAD_DIM]
    scale = 1.0
    for l in range(BLOCK + 1, 9): scale *= float(blocks[l].lambdas[0])
    gen = torch.Generator().manual_seed(SEED); rand_dirs = [torch.randn(L.HEAD_DIM, generator=gen) for _ in range(N_NULL)]
    forwards = 0

    def run(chunk, mode, direction_of=None):
        """mode: None (native), 'zero' (whole slice at the verb), or 'project' (remove the component along direction_of(i) at the verb)."""
        nonlocal forwards
        pos = [verb_of(r) for r in chunk]; handle = None; removed = []
        if mode is not None:
            def pre(_m, args):
                z = args[0].clone(); s_, e_ = HEAD * L.HEAD_DIM, (HEAD + 1) * L.HEAD_DIM
                for i, p in enumerate(pos):
                    sl = z[i, p, s_:e_]
                    if mode == "zero": z[i, p, s_:e_] = 0
                    elif mode == "matched":
                        d = direction_of(i).to(device=sl.device, dtype=sl.dtype); d = d / d.norm(); comp = match_norm[start_of[0] + i] * d; removed.append(float(comp.norm())); z[i, p, s_:e_] = sl - comp
                    else:
                        d = direction_of(i).to(device=sl.device, dtype=sl.dtype); d = d / d.norm(); comp = (sl @ d) * d; removed.append(float(comp.norm())); z[i, p, s_:e_] = sl - comp
                return (z,)
            handle = blocks[BLOCK].attn.c_proj.register_forward_pre_hook(pre)
        try:
            out = dod_units.forward_margins(backend, fw, chunk, 8, None, readers); forwards += 1
        finally:
            if handle is not None: handle.remove()
        return out, removed

    match_norm, start_of = [], [0]
    per_row = {"native": [], "zero": [], "number": []}; per_row.update({f"rand{k}": [] for k in range(N_NULL)}); removed_norms = {"number": [], **{f"rand{k}": [] for k in range(N_NULL)}}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        traces = L.forward_trace_positions(fw, chunk, lambda rw: [verb_of(rw)], upto_layer=9, head_write_layers=(8,)); forwards += 1
        dirs = []
        for row, tr in zip(chunk, traces):
            C = v16.writers_at_block8_input(tr, verb_of(row)); x = sum(C.values()).to(Lrow.device); rms2 = float(x.pow(2).mean())
            r = (float(Rrow @ x) * Lrow + float(Lrow @ x) * Rrow) / rms2; dirs.append(scale * (Wo.T @ r))
        o, _ = run(chunk, None); per_row["native"].extend(o)
        o, _ = run(chunk, "zero"); per_row["zero"].extend(o)
        start_of[0] = start
        o, rm = run(chunk, "project", lambda i: dirs[i]); per_row["number"].extend(o); removed_norms["number"].extend(rm); match_norm.extend(rm)
        for k, rd in enumerate(rand_dirs):
            # norm-matched: remove n_i d_k, n_i = the number removal's norm on row i
            def direction_of(i, rd=rd): return rd
            o, rm = run(chunk, "matched", direction_of); per_row[f"rand{k}"].extend(o); removed_norms[f"rand{k}"].extend(rm)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    pooled = lambda name: sum(per_row[name][i]["target"] - per_row[name][j]["target"] for i, j in pairs)
    native = pooled("native"); damage = {name: (native - pooled(name)) / native for name in per_row if name != "native"}
    unrel = [r_ for r_ in readers if r_ != "target"]
    moves = {name: {r_: sum(abs(a[r_] - n[r_]) for n, a in zip(per_row["native"], per_row[name])) / len(rows) for r_ in unrel} for name in per_row if name != "native"}
    null_moves = {r_: sum(moves[f"rand{k}"][r_] for k in range(N_NULL)) / N_NULL for r_ in unrel}; dmg_row = damage["number"] * native / len(pairs)
    gates = {r_: moves["number"][r_] <= null_moves[r_] + GATE_RATIO * dmg_row for r_ in unrel}
    norm_ratio = {k: (sum(removed_norms[k]) / max(sum(removed_norms["number"]), 1e-9)) for k in removed_norms}
    report = {"native_pooled": native, "damage": damage, "moves": moves, "null_moves": null_moves, "gates": gates, "removed_norm_ratio_vs_number": norm_ratio, "null_damage_max": max(damage[f"rand{k}"] for k in range(N_NULL))}
    print("native", round(native, 2), "damage zero", round(damage["zero"], 4), "number direction", round(damage["number"], 4), "random max", round(report["null_damage_max"], 4), "random norm ratios", [round(norm_ratio[f"rand{k}"], 2) for k in range(4)])
    print("moves number", {r_: round(v, 4) for r_, v in moves["number"].items()}, "null", {r_: round(v, 4) for r_, v in null_moves.items()}, "gates", gates)
    predictions = {"pred_a_whole_slice_damages": damage["zero"] >= WHOLE_MIN, "pred_b_direction_removal_damages": damage["number"] >= DROP_MIN, "pred_c_direction_beats_random_directions": damage["number"] > report["null_damage_max"],
                   "pred_d_direction_removal_selective": all(gates.values()), "pred_e_direction_carries_half_of_full_zeroing": damage["number"] >= HALF * damage["zero"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head45_direction_removal_result_v219", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
