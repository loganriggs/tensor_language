#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v206 pred_b_aligned_directions_cohere pred_c_fixed_direction_damages_like_per_row pred_d_fixed_direction_beats_matched_random pred_e_fixed_direction_selective
"""Pronoun number they/he DoD (v213): the SIGN-ALIGNED fixed direction. v210: the per-row number directions d_i = O_{4.5}^T r_i lie (algebraically) in the 2-D span
of O^T L_829 and O^T R_829 with row-dependent signs, so their plain mean (cosine 0.03 pairwise) is dominated by the plural rows -- and on natural text it acted
as a plural/they direction (v212). Here each d_i is sign-aligned to the first row's direction before averaging (d_i <- sign(d_i . d_1) d_i), the mean
|cosine| is reported, and the aligned mean d_al is removed from 4.5's verb slice on every row (per-row removal repeated in the same run), against 16
norm-matched random directions and the three unrelated readers. If alignment recovers the per-row effect, the component is one fixed direction up to sign.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_replays_v206         native pooled they - he margin = 196.62 within relative 1e-3
    pred_b_aligned_directions_cohere     mean pairwise |cosine| of the normalized d_i >= 0.80
    pred_c_fixed_direction_damages_like_per_row  the aligned-mean removal damage >= 0.80 x the per-row-direction damage
    pred_d_fixed_direction_beats_matched_random  it exceeds every one of the 16 norm-matched random directions
    pred_e_fixed_direction_selective     each unrelated reader |move| <= matched-null mean |move| + 0.25 x damage
PRICE (registered maximum): 3 batches x (native + per-row + fixed + 16 random + 1 trace) = 60 forwards; 0 backwards; 0 fits. Bar <= 63.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_head45_aligned_direction_v213_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_head45_aligned_direction_v213"
BLOCK, HEAD, U829, V206_MARGIN, DROP_MIN, GATE_RATIO, HALF, N_NULL, SEED, BATCH = 4, 5, 829, 196.62, 0.02, 0.25, 0.80, 16, 213, 32
FORWARDS_MAX = 63
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
PREDICTIONS = {"pred_a_baseline_replays_v206": "<= 1e-3", "pred_b_aligned_directions_cohere": ">= 0.80 cosine", "pred_c_fixed_direction_damages_like_per_row": ">= 0.80 x per-row", "pred_d_fixed_direction_beats_matched_random": "> 16 matched random", "pred_e_fixed_direction_selective": "three gates"}
L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
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
    per_row = {"native": [], "number": [], "fixed": []}; per_row.update({f"rand{k}": [] for k in range(N_NULL)}); removed_norms = {"number": [], "fixed": [], **{f"rand{k}": [] for k in range(N_NULL)}}
    all_dirs = []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        traces = L.forward_trace_positions(fw, chunk, lambda rw: [verb_of(rw)], upto_layer=9, head_write_layers=(8,)); forwards += 1
        for row, tr in zip(chunk, traces):
            C = v16.writers_at_block8_input(tr, verb_of(row)); x = sum(C.values()).to(Lrow.device); rms2 = float(x.pow(2).mean())
            r = (float(Rrow @ x) * Lrow + float(Lrow @ x) * Rrow) / rms2; all_dirs.append((scale * (Wo.T @ r)).cpu())
    unit_dirs = torch.stack([d / d.norm() for d in all_dirs]); n_ = len(rows)
    signs = torch.sign(unit_dirs @ unit_dirs[0]); signs[signs == 0] = 1; aligned = unit_dirs * signs[:, None]
    G = unit_dirs @ unit_dirs.T; mean_cos = float((G.abs().sum() - G.abs().diagonal().sum()) / (n_ * (n_ - 1)))
    d_bar = aligned.mean(0); d_bar = d_bar / d_bar.norm()
    print("mean pairwise |cosine| of per-row directions", round(mean_cos, 4), "| plural rows flipped:", int((signs[[i for i, r_ in enumerate(rows) if r_.present]] < 0).sum()), "singular rows flipped:", int((signs[[i for i, r_ in enumerate(rows) if not r_.present]] < 0).sum()), "| cosine with the aligned mean: min", round(float((aligned @ d_bar).min()), 3))
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]; start_of[0] = start
        o, _ = run(chunk, None); per_row["native"].extend(o)
        o, rm = run(chunk, "project", lambda i: all_dirs[start + i]); per_row["number"].extend(o); removed_norms["number"].extend(rm); match_norm.extend(rm)
        o, rm = run(chunk, "project", lambda i: d_bar); per_row["fixed"].extend(o); removed_norms["fixed"].extend(rm)
        for k, rd in enumerate(rand_dirs):
            def direction_of(i, rd=rd): return rd
            o, rm = run(chunk, "matched", direction_of); per_row[f"rand{k}"].extend(o); removed_norms[f"rand{k}"].extend(rm)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    pooled = lambda name: sum(per_row[name][i]["target"] - per_row[name][j]["target"] for i, j in pairs)
    native = pooled("native"); damage = {name: (native - pooled(name)) / native for name in per_row if name != "native"}
    unrel = [r_ for r_ in readers if r_ != "target"]
    moves = {name: {r_: sum(abs(a[r_] - n[r_]) for n, a in zip(per_row["native"], per_row[name])) / len(rows) for r_ in unrel} for name in per_row if name != "native"}
    null_moves = {r_: sum(moves[f"rand{k}"][r_] for k in range(N_NULL)) / N_NULL for r_ in unrel}; dmg_row = damage["number"] * native / len(pairs)
    dmg_row = damage["fixed"] * native / len(pairs); gates = {r_: moves["fixed"][r_] <= null_moves[r_] + GATE_RATIO * dmg_row for r_ in unrel}
    norm_ratio = {k: (sum(removed_norms[k]) / max(sum(removed_norms["number"]), 1e-9)) for k in removed_norms}
    report = {"native_pooled": native, "mean_pairwise_abs_cosine": mean_cos, "plural_rows_flipped": int((signs[[i for i, r_ in enumerate(rows) if r_.present]] < 0).sum()), "singular_rows_flipped": int((signs[[i for i, r_ in enumerate(rows) if not r_.present]] < 0).sum()), "damage": damage, "moves": moves, "null_moves": null_moves, "gates": gates, "removed_norm_ratio_vs_number": norm_ratio, "null_damage_max": max(damage[f"rand{k}"] for k in range(N_NULL))}
    print("native", round(native, 2), "damage per-row", round(damage["number"], 4), "fixed", round(damage["fixed"], 4), "random max", round(report["null_damage_max"], 4))
    print("moves fixed", {r_: round(v, 4) for r_, v in moves["fixed"].items()}, "null", {r_: round(v, 4) for r_, v in null_moves.items()}, "gates", gates)
    predictions = {"pred_a_baseline_replays_v206": abs(native - V206_MARGIN) / V206_MARGIN <= 1e-3, "pred_b_aligned_directions_cohere": mean_cos >= 0.80, "pred_c_fixed_direction_damages_like_per_row": damage["fixed"] >= HALF * damage["number"],
                   "pred_d_fixed_direction_beats_matched_random": damage["fixed"] > report["null_damage_max"], "pred_e_fixed_direction_selective": all(gates.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head45_aligned_direction_result_v213", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
