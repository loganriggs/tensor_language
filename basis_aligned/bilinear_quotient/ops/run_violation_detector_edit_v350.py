#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_frames_replay pred_b_zeroing_3465_moves_the_violation_cells_more pred_c_beats_random_units pred_d_direction_toward_repair pred_e_493_effect_on_its_own_cell
"""Does the violation detector matter to the next prediction? (v350). v347-v349: MLP-3 unit 3465 fires at the verb of "The X are" when X is singular (and at
"The X is" when X is plural); 493 fires for plural + is. Edit on the four cells of the "The X is / are" frames (256 pairs): zero 3465, zero 493, or zero
both at the verb position; read the next-token distribution at the verb: KL(native || edited) per cell, and the change in the log-odds of a plural-agreeing
continuation set (" they", " them", " these", " those", " all", " both") against a singular one (" it", " him", " her", " this", " that", " one"). Null: 8 seeded
random MLP-3 units, each zeroed alone. Registered reading: the violation cells move more than the grammatical cells, and beyond the null.
PREDICTIONS (scored as written; failures preserved; priors unsure -- a detector's downstream use is unknown)
    pred_a_baseline_frames_replay                  the native pass reproduces v349's 3465 medians (are|s -164.6, is|s -4.0) within 5%
    pred_b_zeroing_3465_moves_the_violation_cells_more  mean KL under zero-3465 is larger on the two violation cells than on the two grammatical cells
    pred_c_beats_random_units                      mean KL under zero-3465 on the violation cells exceeds 3x the largest mean KL among the 8 random units on those cells
    pred_d_direction_toward_repair                 on the "singular X + are" cell, zeroing 3465 raises the plural-vs-singular continuation log-odds (the model reads the plural verb at face value without the violation flag). Prior: unsure.
    pred_e_493_effect_on_its_own_cell              zero-493's mean KL is largest on the "plural X + is" cell among the four
PRICE (registered maximum): 4 cells x (1 native + 3 edits + 8 nulls) = 48 forwards (full depth, 256 rows each); 0 backwards; 0 fits. Bar <= 52.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/violation_detector_edit_v350_result.json"
CANDIDATE_ID = "pronoun_number.violation_detector_edit_v350"
LAYER, UNITS, N_NULL, SEED = 3, {"3465": (3465,), "493": (493,), "both": (3465, 493)}, 8, 350
PLURAL = (" they", " them", " these", " those", " all", " both"); SINGULAR = (" it", " him", " her", " this", " that", " one")
V349 = {"are|s": -164.6, "is|s": -4.0}; REPLAY, NULL_FACTOR = 0.05, 3.0
FORWARDS_MAX = 52
PREDICTIONS = {"pred_a_baseline_frames_replay": "within 5%", "pred_b_zeroing_3465_moves_the_violation_cells_more": "violation KL > grammatical KL", "pred_c_beats_random_units": "> 3x null", "pred_d_direction_toward_repair": "log-odds rise", "pred_e_493_effect_on_its_own_cell": "largest on plural + is"}


def run(backend, ids, units):
    """full forward with the given MLP-3 units zeroed at the last position; returns log-probs at the last position and unit 3465 / 493 activations there."""
    torch, F, model = backend.torch, backend.F, backend.model
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None; acts = None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == LAYER:
                h = dod_units.hidden(model, block.mlp, xin); acts = h[:, -1, [3465, 493]].float().cpu()
                if units: h[:, -1, torch.tensor(list(units), device=h.device)] = 0
                m = block.mlp.Down(h) + block.mlp.Down_bias
            else: m = block.mlp(xin)
            x = x + m
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    return torch.log_softmax(logits[:, -1].float(), -1).cpu(), acts


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    the, is_, are = L._single("The"), L._single(" is"), L._single(" are"); pl = [L._single(t) for t in PLURAL]; sg = [L._single(t) for t in SINGULAR]
    cells = {"is|s": [[the, a, is_] for a, _ in pairs], "is|p": [[the, b, is_] for _, b in pairs], "are|s": [[the, a, are] for a, _ in pairs], "are|p": [[the, b, are] for _, b in pairs]}
    rng = random.Random(SEED); pool = [j for j in range(4608) if j not in (3465, 493)]; nulls = [(rng.choice(pool),) for _ in range(N_NULL)]
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "cells": list(cells), "units": {k: list(v) for k, v in UNITS.items()}, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay": REPLAY, "null_factor": NULL_FACTOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; forwards = 0
    conds = [("native", ())] + [(k, v) for k, v in UNITS.items()] + [(f"null{i}", n) for i, n in enumerate(nulls)]
    lp, act = {}, {}
    for cname, rows in cells.items():
        ids = torch.tensor(rows, device="cuda")
        for name, units in conds:
            l_, a_ = run(backend, ids, units); forwards += 1; lp[(cname, name)] = l_
            if name == "native": act[cname] = a_
    med3465 = {c: float(act[c][:, 0].median()) for c in cells}; replay = all(abs(med3465[c] - V349[c]) <= REPLAY * abs(V349[c]) for c in V349)
    def kl(c, name): P, Q = lp[(c, "native")], lp[(c, name)]; return float((P.exp() * (P - Q)).sum(-1).mean())
    def logodds(c, name): Q = lp[(c, name)]; return float((torch.logsumexp(Q[:, pl], -1) - torch.logsumexp(Q[:, sg], -1)).mean())
    KL = {name: {c: kl(c, name) for c in cells} for name, _ in conds if name != "native"}; LO = {name: {c: logodds(c, name) for c in cells} for name, _ in conds}
    viol = ("are|s", "is|p"); gram = ("is|s", "are|p")
    kl3_v = sum(KL["3465"][c] for c in viol) / 2; kl3_g = sum(KL["3465"][c] for c in gram) / 2; null_v = max(sum(KL[f"null{i}"][c] for c in viol) / 2 for i in range(N_NULL))
    d_lo = LO["3465"]["are|s"] - LO["native"]["are|s"]
    report = {"native_3465_medians": med3465, "kl": KL, "logodds_plural_vs_singular": LO, "kl_3465_violation_mean": kl3_v, "kl_3465_grammatical_mean": kl3_g, "null_violation_kl_max": null_v, "delta_logodds_3465_on_are_s": d_lo,
              "kl_493_argmax_cell": max(cells, key=lambda c: KL["493"][c])}
    print(json.dumps({k: v for k, v in report.items() if k not in ("kl", "logodds_plural_vs_singular")}, indent=1)); print(json.dumps({k: {c: round(v, 4) for c, v in d.items()} for k, d in KL.items() if not k.startswith("null")}, indent=1))
    predictions = {"pred_a_baseline_frames_replay": replay, "pred_b_zeroing_3465_moves_the_violation_cells_more": kl3_v > kl3_g, "pred_c_beats_random_units": kl3_v > NULL_FACTOR * null_v, "pred_d_direction_toward_repair": d_lo > 0, "pred_e_493_effect_on_its_own_cell": report["kl_493_argmax_cell"] == "is|p"}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "violation_detector_edit_result_v350", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
