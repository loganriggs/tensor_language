#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_damage_live pred_c_beats_matched_random_directions pred_d_selective pred_e_incongruent_shifts_toward_label
"""Pronoun number they/he DoD (v212): head 4.5's fixed number direction OUT OF THE PANEL. d_bar = the normalized mean over the v76 rows of O_{4.5}^T r_i
(r_i = the plural detector 829's product gradient at the verb; v210: -3.6% of the panel margin when removed at the verb, selective, 4x the matched null).
Here d_bar is recomputed from the panel (3 traces) and its component removed from head 4.5's slice at ALL positions of the 128 natural rows (FineWeb v77 +
Pile v78; the verb position is not annotated there, and v211 showed all-positions removal is 1.15x the verb-only one). Damage = oriented drop of the label
margin on congruent rows; incongruent rows should shift toward the label if the direction carries the noun's number and nothing else. Null: 16 random
directions, norm-matched per row and position. Readers will-would / who-which / night-day (number-free), gate move <= null mean + 0.25 x damage.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_instrument_replays_native      unedited forward = producer native <= 1e-4
    pred_b_congruent_damage_live          congruent rows: damage fraction >= 0.02 and positive >= 0.60 (panel: 0.041 at all positions)
    pred_c_beats_matched_random_directions  congruent damage > every one of the 16 norm-matched random directions
    pred_d_selective                      each unrelated reader |move| <= matched-null mean |move| + 0.25 x damage
    pred_e_incongruent_shifts_toward_label  incongruent rows: mean oriented damage < 0
PRICE (registered maximum): 3 panel traces + 4 natural batches x (native + edit + 16 nulls) + producer replay 4 = 79 forwards; 0 backwards; 0 fits. Bar <= 84.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_natural_line as N, dod_units

L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_head45_direction_natural_v212_result.json"
SOURCES = {"fineweb": ROOT / "circuits/followups/pronoun_number_dod_natural_rows_v77.json", "pile": ROOT / "circuits/followups/pronoun_number_dod_pile_rows_v78.json"}
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_head45_direction_natural_v212"
BLOCK, HEAD, U829, CONGRUENT = 4, 5, 829, ("natural_plural_they", "natural_singular_he")
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, INSTRUMENT_TOL, N_NULL, SEED, BATCH = 0.02, 0.60, 0.25, 1e-4, 16, 212, 32
FORWARDS_MAX = 84
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_damage_live": ">= 0.02, positive >= 0.60", "pred_c_beats_matched_random_directions": "> 16 matched", "pred_d_selective": "three gates", "pred_e_incongruent_shifts_toward_label": "< 0"}


def main() -> None:
    prows, they, he_, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    verb_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns) + 1
    rows, shas = [], {}
    for name, path in SOURCES.items():
        rs, sha, _t, _h = N.rows_from_receipt(path, "they", " they", " he", lambda r: f"natural_{r['number']}_{r['label']}"); rows.extend(rs); shas[name] = sha
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": shas, "panel_rows_sha256": L.rows_sha256(prows), "block": BLOCK, "head": HEAD, "congruent": list(CONGRUENT), "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "gate_ratio": GATE_RATIO, "instrument_tol": INSTRUMENT_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    fw = L.ManualForward(backend); fw.backend = backend; blocks = model.transformer.h
    mlp8 = blocks[8].mlp; Lrow, Rrow = mlp8.Left.weight.detach().float()[U829], mlp8.Right.weight.detach().float()[U829]
    Wo = blocks[BLOCK].attn.c_proj.weight.detach().float()[:, HEAD * L.HEAD_DIM:(HEAD + 1) * L.HEAD_DIM]
    forwards, dirs = 0, []
    for start in range(0, len(prows), BATCH):
        chunk = prows[start:start + BATCH]
        traces = L.forward_trace_positions(fw, chunk, lambda rw: [verb_of(rw)], upto_layer=9, head_write_layers=(8,)); forwards += 1
        for row, tr in zip(chunk, traces):
            C = v16.writers_at_block8_input(tr, verb_of(row)); x = sum(C.values()).to(Lrow.device); rms2 = float(x.pow(2).mean())
            r = (float(Rrow @ x) * Lrow + float(Lrow @ x) * Rrow) / rms2; d = Wo.T @ r; dirs.append((d / d.norm()).cpu())
    d_bar = torch.stack(dirs).mean(0); d_bar = d_bar / d_bar.norm()
    gen = torch.Generator().manual_seed(SEED); rand_dirs = [torch.randn(L.HEAD_DIM, generator=gen) for _ in range(N_NULL)]
    match = {}

    def run(chunk, mode, direction=None, offset=0):
        nonlocal forwards; handle = None
        if mode is not None:
            def pre(_m, args):
                z = args[0].clone(); s_, e_ = HEAD * L.HEAD_DIM, (HEAD + 1) * L.HEAD_DIM
                d = direction.to(device=z.device, dtype=z.dtype); d = d / d.norm()
                for i in range(z.shape[0]):
                    for q in range(z.shape[1]):
                        sl = z[i, q, s_:e_]
                        if mode == "project": comp = (sl @ d) * d; match[(offset + i, q)] = float(comp.norm())
                        else: comp = match[(offset + i, q)] * d
                        z[i, q, s_:e_] = sl - comp
                return (z,)
            handle = blocks[BLOCK].attn.c_proj.register_forward_pre_hook(pre)
        try:
            out = dod_units.forward_margins(backend, fw, chunk, 8, None, readers); forwards += 1
        finally:
            if handle is not None: handle.remove()
        return out

    native, edit, nulls = [], [], [[] for _ in range(N_NULL)]
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        native.extend(run(chunk, None)); edit.extend(run(chunk, "project", d_bar, start))
        for k, rd in enumerate(rand_dirs): nulls[k].extend(run(chunk, "matched", rd, start))
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    cong = [i for i, r in enumerate(rows) if r.construction in CONGRUENT]; incong = [i for i, r in enumerate(rows) if r.construction not in CONGRUENT]
    def summarize(arm, idx):
        d = [(native[i]["answer"] - native[i]["foil"]) - (arm[i]["answer"] - arm[i]["foil"]) for i in idx]
        m = sum(d) / len(d); nm = sum(native[i]["answer"] - native[i]["foil"] for i in idx) / len(idx)
        return {"damage": m, "native_margin": nm, "fraction": m / nm, "positive": sum(1 for x in d if x > 0) / len(d), **{f"{k}_abs_move": sum(abs(arm[i][k] - native[i][k]) for i in idx) / len(idx) for k in readers}}
    arms = {"congruent": summarize(edit, cong), "incongruent": summarize(edit, incong), "per_cell": {c: summarize(edit, [i for i, r in enumerate(rows) if r.construction == c]) for c in sorted({r.construction for r in rows})}}
    null_s = [summarize(nl, cong) for nl in nulls]; null_max = max(s["damage"] for s in null_s); null_moves = {k: sum(s[f"{k}_abs_move"] for s in null_s) / N_NULL for k in readers}
    two = arms["congruent"]; gates = {k: two[f"{k}_abs_move"] <= null_moves[k] + GATE_RATIO * two["damage"] for k in readers}
    print("instrument", instrument, "congruent", {k: round(v, 4) for k, v in two.items()}, "incongruent", round(arms["incongruent"]["damage"], 4), round(arms["incongruent"]["fraction"], 4), "null max", round(null_max, 4), "gates", gates)
    print("per cell", {c: (round(s["damage"], 3), round(s["fraction"], 3), s["positive"]) for c, s in arms["per_cell"].items()})
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_congruent_damage_live": two["fraction"] >= LIVE_FRACTION and two["positive"] >= LIVE_POSITIVE, "pred_c_beats_matched_random_directions": two["damage"] > null_max,
                   "pred_d_selective": all(gates.values()), "pred_e_incongruent_shifts_toward_label": arms["incongruent"]["damage"] < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head45_direction_natural_result_v212", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": arms, "null_damage_max": null_max, "null_moves": null_moves, "nulls": null_s, "gates": gates,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
