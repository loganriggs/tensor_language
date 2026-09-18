#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v204 pred_b_69_contrast_drops_010 pred_c_829_verb_contrast_drops_003 pred_d_edit_beats_random_sets pred_e_both_drop_not_grow
"""Pronoun number they/he DoD (v246): EDIT of the verb-site MLP-5 units {1036, 715} (v245: 52% + 30% of MLP 5's carriage into MLP-6 unit 69 at the verb) at the VERB only,
in a plain forward: the pooled plural - singular contrast of u_69 at the verb (block 6), of u_829 at the verb (block 8), and the they - he margin. Null: 16 seeded random
2-unit sets of MLP 5 zeroed at the verb. Sizing prior (v195 / v225 calibration: edits run at or below the in-place leave-out; first-order carriage of MLP 5 into 69 is 36%
(v242) x 82% = 30%): the verb-site chain is one of two sites (v215: verb 5.9% of the margin), so margin effects should be small.
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_replays_v204       the unedited pooled contrast of u_829 at the verb matches v204 (3611.8) within relative 1e-3
    pred_b_69_contrast_drops_010       the edit removes >= 0.10 of |u_69's verb contrast|
    pred_c_829_verb_contrast_drops_003 the edit removes >= 0.03 of |u_829's verb contrast|
    pred_d_edit_beats_random_sets      the edit's |fractional change of u_69| exceeds 3x the largest among the 16 random sets
    pred_e_both_drop_not_grow          u_69 and u_829 verb contrasts shrink in magnitude
PRICE (registered maximum): 3 batches x (1 baseline + 1 edit + 16 null) = 54 forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_verb_mlp5_units_edit_v246_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_verb_mlp5_units_edit_v246"
UNITS, LAYER, U2483, U829, V183_CONTRAST, DROP_2483_MIN, DROP_829_MIN, NULL_FACTOR, N_NULL, SEED, BATCH = (1036, 715), 5, 69, 829, 3611.8, 0.10, 0.03, 3.0, 16, 246, 32
FORWARDS_MAX = 60
PREDICTIONS = {"pred_a_baseline_replays_v204": "<= 1e-3", "pred_b_69_contrast_drops_010": ">= 0.10", "pred_c_829_verb_contrast_drops_003": ">= 0.03", "pred_d_edit_beats_random_sets": "> 3x null max", "pred_e_both_drop_not_grow": "both shrink"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns) + 1     # the VERB position
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": UNITS, "layer": LAYER, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"drop_2483_min": DROP_2483_MIN, "drop_829_min": DROP_829_MIN, "null_factor": NULL_FACTOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h
    import random
    rng = random.Random(SEED); pool = [j for j in range(blocks[LAYER].mlp.Down.weight.shape[1]) if j not in UNITS]
    null_sets = [tuple(sorted(rng.sample(pool, len(UNITS)))) for _ in range(N_NULL)]
    conditions = [("baseline", None), ("edit", UNITS)] + [(f"null{k}", s) for k, s in enumerate(null_sets)]
    forwards, per_row = 0, {name: [] for name, _ in conditions}

    def run(chunk, units):
        tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk)); out = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                h = dod_units.hidden(model, block.mlp, xin)
                if l == LAYER and units is not None:
                    h = h.clone()
                    for i, p in enumerate(pos): h[i, p, list(units)] = 0
                if l == 6: out["u2483"] = h[idx, pos, U2483].float().cpu()
                if l == 8: out["u829"] = h[idx, pos, U829].float().cpu()
                x = x + block.mlp.Down(h) + block.mlp.Down_bias
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            fin = torch.tensor([r.final for r in chunk]); lg = logits[idx, fin].float()
            out["margin"] = (lg[:, he] - lg[:, she]).cpu()
        return out

    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, units in conditions:
            o = run(chunk, units); forwards += 1
            per_row[name].extend({"u2483": float(o["u2483"][i]), "u829": float(o["u829"][i]), "margin": float(o["margin"][i])} for i in range(len(chunk)))
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    def pooled(name, key):
        return sum(per_row[name][i][key] - per_row[name][partner[(row.construction, row.group, False)]][key] for i, row in enumerate(rows) if row.present)
    base = {k: pooled("baseline", k) for k in ("u2483", "u829", "margin")}; edit = {k: pooled("edit", k) for k in base}
    frac = {k: (abs(edit[k]) - abs(base[k])) / abs(base[k]) for k in base}          # negative = shrink
    nulls = [{k: (abs(pooled(f"null{j}", k)) - abs(base[k])) / abs(base[k]) for k in base} for j in range(N_NULL)]
    null_max = {k: max(abs(n[k]) for n in nulls) for k in base}
    report = {"baseline": base, "edit": edit, "fractional_change": frac, "null_max_abs": null_max, "null_sets": null_sets, "nulls": nulls}
    print("baseline", {k: round(v, 2) for k, v in base.items()}, "edit", {k: round(v, 2) for k, v in edit.items()}); print("fractional change", {k: round(v, 4) for k, v in frac.items()}, "null max", {k: round(v, 4) for k, v in null_max.items()})
    predictions = {"pred_a_baseline_replays_v204": abs(base["u829"] - V183_CONTRAST) / abs(V183_CONTRAST) <= 1e-3, "pred_b_69_contrast_drops_010": -frac["u2483"] >= DROP_2483_MIN, "pred_c_829_verb_contrast_drops_003": -frac["u829"] >= DROP_829_MIN,
                   "pred_d_edit_beats_random_sets": abs(frac["u2483"]) > NULL_FACTOR * null_max["u2483"], "pred_e_both_drop_not_grow": frac["u2483"] < 0 and frac["u829"] < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_verb_mlp5_units_edit_result_v246", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
