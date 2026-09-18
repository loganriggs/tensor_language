#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_zeroing_3230_shrinks_the_male_detector pred_c_zeroing_3230_beats_random_mlp6_units pred_d_margin_drop_beats_random_units
"""Pronoun gender he/she DoD (v177): EDIT the MLP-6 -> MLP-8 link. Zero MLP-6 unit 3230 at the gendered-noun position and measure (i) the male-noun
detector's activation (MLP-8 unit 3152's (Lx)(Rx) at the noun, oriented male - female) and (ii) the he/she margin at the final query; null = 16 random
single MLP-6 units zeroed the same way. Folds (v173 / v175) nominated 3230 as ~82% of MLP 6's input to 3152 and MLP 6 as 0.17-0.28 of 3152's factors;
the edit sizes the link. Hooked plain forward (v165's), instrument-checked against the producer.
PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native            <= 1e-4
    pred_b_zeroing_3230_shrinks_the_male_detector  3152's male - female activation contrast at the noun drops by >= 0.15 of its native value (prior: ~0.2)
    pred_c_zeroing_3230_beats_random_mlp6_units  that drop exceeds the max over 16 random single units
    pred_d_margin_drop_beats_random_units       the he/she margin drop exceeds the max over the 16 random units (prior: the drop itself will be tiny, ~0.01 logits)
PRICE (registered maximum): 2 batches x (native + producer + 1 arm + 16 nulls) = 38 forwards; 0 backwards; 0 fits. Bar <= 40.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_unit3230_edit_v177_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_unit3230_edit_v177"
SRC_LAYER, SRC_UNIT, DST_LAYER, DST_UNIT = 6, 3230, 8, 3152
NULL_SEEDS = tuple(range(7701, 7717))
DROP_MIN, INSTRUMENT_TOL = 0.15, 1e-4
FORWARDS_MAX = 40
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_zeroing_3230_shrinks_the_male_detector": ">= 0.15 of native", "pred_c_zeroing_3230_beats_random_mlp6_units": "> random max", "pred_d_margin_drop_beats_random_units": "> random max"}


def forward(backend, fw, rows, noun_of, zero_unit):
    """Plain forward; optionally zero MLP-6 hidden unit `zero_unit` at the noun. Returns per-row (margin, 3152 activation at the noun)."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens = fw._tokens(rows); m6, m8 = model.transformer.h[SRC_LAYER].mlp, model.transformer.h[DST_LAYER].mlp
    act = [None] * len(rows)
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == SRC_LAYER and zero_unit is not None:
                h = m6.Left(xin) * m6.Right(xin)
                for i, row in enumerate(rows): h[i, noun_of(row), zero_unit] = 0
                x = x + m6.Down(h) + m6.Down_bias
            else:
                if l == DST_LAYER:
                    h8 = m8.Left(xin) * m8.Right(xin)
                    for i, row in enumerate(rows): act[i] = float(h8[i, noun_of(row), DST_UNIT])
                x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    return [{"answer": float(logits[i, r.final, r.answer_id]), "foil": float(logits[i, r.final, r.foil_id]), "act": act[i]} for i, r in enumerate(rows)]


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "edit": f"zero MLP{SRC_LAYER} unit {SRC_UNIT} at the noun", "measure": f"MLP{DST_LAYER} unit {DST_UNIT} activation at the noun + he/she margin",
            "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"drop_min": DROP_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fw = L.ManualForward(backend); forwards = 0
    def run(zero_unit):
        nonlocal forwards; out = []
        for start in range(0, len(rows), v1.BATCH):
            out.extend(forward(backend, fw, rows[start:start + v1.BATCH], noun_of, zero_unit)); forwards += 1
        return out
    native = run(None)
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    def contrast(arm):
        return sum(arm[i]["act"] - arm[partner[(r.construction, r.group, False)]]["act"] for i, r in enumerate(rows) if r.present) / sum(1 for r in rows if r.present)
    def margin_drop(arm):
        return sum((nat["answer"] - nat["foil"]) - (a["answer"] - a["foil"]) for nat, a in zip(native, arm)) / len(rows)
    nat_c = contrast(native)
    edit = run(SRC_UNIT); edit_drop = (nat_c - contrast(edit)) / nat_c; edit_margin = margin_drop(edit)
    nulls = []
    for seed in NULL_SEEDS:
        gen = torch.Generator(device="cpu").manual_seed(seed); u = int(torch.randint(0, 4608, (1,), generator=gen))
        arm = run(u); nulls.append({"unit": u, "act_drop": (nat_c - contrast(arm)) / nat_c, "margin_drop": margin_drop(arm)})
    print("native 3152 contrast", round(nat_c, 2), "edit: act drop", round(edit_drop, 3), "margin drop", round(edit_margin, 4), "| null max act drop", round(max(x["act_drop"] for x in nulls), 3), "null max margin drop", round(max(x["margin_drop"] for x in nulls), 4))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_zeroing_3230_shrinks_the_male_detector": edit_drop >= DROP_MIN,
                   "pred_c_zeroing_3230_beats_random_mlp6_units": edit_drop > max(x["act_drop"] for x in nulls), "pred_d_margin_drop_beats_random_units": edit_margin > max(x["margin_drop"] for x in nulls)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_unit3230_edit_result_v177", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "native_3152_contrast": nat_c, "edit": {"act_drop": edit_drop, "margin_drop": edit_margin},
                               "nulls": nulls, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
