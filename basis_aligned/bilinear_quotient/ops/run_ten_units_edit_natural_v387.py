#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_margin_positive pred_b_ten_lower_the_margin_on_text pred_c_beats_random_sets_on_text pred_d_ten_exceed_six_on_text pred_e_within_2x_of_panel
"""The ten named units (noun chain + post-noun site) edited on natural text (v387). v386 (panel): {3465, 493, 1036, 2483, 1779, 829} + {3661, 715, 69, 834} cost
20.2% of the margin; v371 (text): the six cost 8.8%. Here the ten on the 128 natural rows (labelled he / they), the they - he logit difference at the final token
oriented by the label; 12 layer-matched random 10-sets as null.
PREDICTIONS (scored as written; failures preserved; priors from v371 / v386)
    pred_a_native_margin_positive     the native oriented margin on the natural rows is positive
    pred_b_ten_lower_the_margin_on_text  zeroing the ten lowers the oriented margin by >= 0.08 of its native value
    pred_c_beats_random_sets_on_text  |change| exceeds 3x the largest |change| among the 12 random 10-sets
    pred_d_ten_exceed_six_on_text     |ten change| > the six's 0.210 (v371)
    pred_e_within_2x_of_panel         the relative change is within a factor 2 of the panel's -0.202 (-0.10 to -0.40). Prior: unsure.
PRICE (registered maximum): 2 natural batches x (1 native + 1 edit + 12 null) = 28 forwards; 0 backwards; 0 fits. Bar <= 30.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/ten_units_edit_natural_v387_result.json"
CANDIDATE_ID = "pronoun_number.ten_units_edit_natural_v387"
CHAIN = {3: (3465, 493), 4: (3661,), 5: (1036, 715), 6: (2483, 69), 7: (1779, 834), 8: (829,)}; N_NULL, SEED, BATCH = 12, 387, 32
MOVE_MIN, NULL_FACTOR, PANEL_REL, SIX_TEXT = 0.08, 3.0, -0.2018, 0.2103
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 32
PREDICTIONS = {"pred_a_native_margin_positive": "> 0", "pred_b_ten_lower_the_margin_on_text": "<= -0.08 of native", "pred_c_beats_random_sets_on_text": "> 3x null", "pred_d_ten_exceed_six_on_text": "> 0.210", "pred_e_within_2x_of_panel": "-0.10 to -0.40"}


def margins_multi(backend, tokens, finals, edits, readers):
    """they - he margins at `finals` with `edits` = {layer: units} zeroed at every position (several layers in one forward)."""
    torch, F, model = backend.torch, backend.F, backend.model
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if edits and l in edits:
                h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0; x = x + block.mlp.Down(h) + block.mlp.Down_bias
            else: x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    return [{name: float(logits[i, finals[i], a] - logits[i, finals[i], b]) for name, (a, b) in readers.items()} for i in range(tokens.shape[0])]


def main() -> None:
    recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    rng = random.Random(SEED); layers = [3, 3, 4, 5, 5, 6, 6, 7, 7, 8]; named = {u for v in CHAIN.values() for u in v}
    null_sets = []
    for _ in range(N_NULL):
        d = {}
        for l in layers: d.setdefault(l, []).append(rng.choice([j for j in range(4608) if j not in named]))
        null_sets.append({l: tuple(v) for l, v in d.items()})
    plan = {"candidate_id": CANDIDATE_ID, "natural_rows": len(recs), "chain": {str(k): list(v) for k, v in CHAIN.items()}, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"move_min": MOVE_MIN, "null_factor": NULL_FACTOR, "panel_rel": PANEL_REL, "six_text": SIX_TEXT}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; forwards = 0
    nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda"); finals = [len(r_["ids"]) - 1 for r_ in recs]; sign = [1.0 if r_["label"] == "they" else -1.0 for r_ in recs]
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    conditions = [("baseline", None), ("chain", CHAIN)] + [(f"null{k}", s_) for k, s_ in enumerate(null_sets)]
    for name, edits in conditions:
        out = []
        for start in range(0, len(recs), 64):
            out += margins_multi(backend, nat[start:start + 64], finals[start:start + 64], edits, readers); forwards += 1
        margins[name] = out
    def oriented(name): return sum(m["they_he"] * sg for m, sg in zip(margins[name], sign)) / len(recs)
    base = oriented("baseline"); d_chain = oriented("chain") - base; nulls = [oriented(f"null{k}") - base for k in range(N_NULL)]
    report = {"native_margin": base, "change_ten": d_chain, "null_changes": nulls, "null_max_abs": max(abs(v) for v in nulls), "chain_rel": d_chain / base if base else None, "labels": {"they": sign.count(1.0), "he": sign.count(-1.0)}}
    print(json.dumps(report, indent=1))
    rel = report["chain_rel"] if report["chain_rel"] is not None else 0.0
    predictions = {"pred_a_native_margin_positive": base > 0, "pred_b_ten_lower_the_margin_on_text": d_chain <= -MOVE_MIN * abs(base), "pred_c_beats_random_sets_on_text": abs(d_chain) > NULL_FACTOR * report["null_max_abs"], "pred_d_ten_exceed_six_on_text": abs(d_chain) > SIX_TEXT,
                   "pred_e_within_2x_of_panel": PANEL_REL * 2 <= rel <= PANEL_REL / 2}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "ten_units_edit_natural_result_v387", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
