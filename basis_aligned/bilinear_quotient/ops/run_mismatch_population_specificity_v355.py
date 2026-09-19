#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_rows_load pred_b_leaders_quiet_at_grammatical_verbs pred_c_leaders_loud_on_the_violation_frames pred_d_ratio_at_least_5x pred_e_no_number_separation_at_natural_verbs
"""Specificity of the mismatch populations on text (v355). v351 / v354: MLP-3 units 3040 and 114 lead the populations that flag a noun-verb number mismatch
and are used by the model. On grammatical text they should be quiet at the verb. The 128 natural rows' verbs (miner's second_offset; past-tense, number-
neutral) give a grammatical baseline: |h| of 3040 / 114 / 565 / 3465 at those verbs vs their |h| on the violation cells of the "The X is / are" frames
(v349 medians: 3040 -1390 on singular + are; 114 +1098 on plural + is), and the plural-cue vs singular-cue separation at the natural verbs.
PREDICTIONS (scored as written; failures preserved; priors from v349 / v352)
    pred_a_rows_load                          128 rows with valid verb offsets (instrument)
    pred_b_leaders_quiet_at_grammatical_verbs the median |h| of 3040 and 114 at the natural verbs is <= 0.2x their violation-cell magnitude
    pred_c_leaders_loud_on_the_violation_frames  re-measured here, 3040's singular + are median and 114's plural + is median replay v349 / v352 within 10%
    pred_d_ratio_at_least_5x                  violation-cell |median| / natural-verb median |h| >= 5 for both leaders
    pred_e_no_number_separation_at_natural_verbs  at the natural verbs, 3040 and 114 separate plural-cue from singular-cue rows by < 0.5 pooled std (past-tense verbs carry no number to check)
PRICE (registered maximum): 2 natural batches + 2 frame cells = 4 forwards (blocks 0-3); 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mismatch_population_specificity_v355_result.json"
CANDIDATE_ID = "pronoun_number.mismatch_population_specificity_v355"
UNITS = {3: (3040, 114, 565, 3465)}; BATCH = 64
V352 = {"3040": -1390.0, "114": 1098.0}
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
QUIET, REPLAY, RATIO, SEP_MAX = 0.2, 0.10, 5.0, 0.5
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_rows_load": "128 rows", "pred_b_leaders_quiet_at_grammatical_verbs": "<= 0.2x", "pred_c_leaders_loud_on_the_violation_frames": "within 10%", "pred_d_ratio_at_least_5x": ">= 5", "pred_e_no_number_separation_at_natural_verbs": "< 0.5 std"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    cue = [int(r["cue_offset"]) for r in recs]; verb = [int(r["second_offset"]) for r in recs]; fin = [len(r["ids"]) - 1 for r in recs]; plural = [i for i, r in enumerate(recs) if r["cue"] == "plural"]; sing = [i for i, r in enumerate(recs) if r["cue"] != "plural"]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "plural_rows": len(plural), "singular_rows": len(sing), "units": {str(k): list(v) for k, v in UNITS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"quiet": QUIET, "replay": REPLAY, "ratio": RATIO, "sep_max": SEP_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); H = {u: [] for u in UNITS[3]}
    def blocks03(ids, pos):
        out = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None; idx = torch.arange(ids.shape[0])
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == 3:
                    hh = dod_units.hidden(model, block.mlp, xin).float().cpu()
                    for u in UNITS[3]: out[u] = hh[idx, pos, u]
                    return out
                x = x + block.mlp(xin)
    for s0 in range(0, len(recs), BATCH):
        chunk = nat[s0:s0 + BATCH]; o = blocks03(chunk, torch.tensor(verb[s0:s0 + BATCH])); forwards += 1
        for u in UNITS[3]: H[u].append(o[u])
    H = {u: torch.cat(v) for u, v in H.items()}
    import run_mlp1_token_table_scaling_v287 as v287
    pairs = [(a_, b_) for _, a_, b_ in v287.SPEC["noun_pairs_vocab"]][:256]; the, is_, are = L._single("The"), L._single(" is"), L._single(" are")
    cellA = blocks03(torch.tensor([[the, a_, are] for a_, _ in pairs], device="cuda"), torch.full((256,), 2)); forwards += 1     # singular + are
    cellB = blocks03(torch.tensor([[the, b_, is_] for _, b_ in pairs], device="cuda"), torch.full((256,), 2)); forwards += 1     # plural + is
    viol = {"3040": float(cellA[3040].median()), "114": float(cellB[114].median())}
    per = {}
    for u in UNITS[3]:
        h = H[u]; p_, s_ = h[plural], h[sing]; std = float(torch.cat([p_ - p_.mean(), s_ - s_.mean()]).std())
        per[str(u)] = {"natural_verb_abs_median": float(h.abs().median()), "natural_verb_median": float(h.median()), "sep_over_std_at_verb": float((p_.median() - s_.median()) / std)}
    ratio = {u: abs(viol[u]) / max(per[u]["natural_verb_abs_median"], 1e-6) for u in viol}
    report = {"violation_cell_medians": viol, "per_unit_at_natural_verbs": per, "violation_over_natural_ratio": ratio}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_rows_load": len(recs) == 128 and all(0 <= v_ < len(recs[i]["ids"]) for i, v_ in enumerate(verb)), "pred_b_leaders_quiet_at_grammatical_verbs": all(per[u]["natural_verb_abs_median"] <= QUIET * abs(viol[u]) for u in viol),
                   "pred_c_leaders_loud_on_the_violation_frames": all(abs(viol[u] - V352[u]) <= REPLAY * abs(V352[u]) for u in viol), "pred_d_ratio_at_least_5x": all(r_ >= RATIO for r_ in ratio.values()),
                   "pred_e_no_number_separation_at_natural_verbs": all(abs(per[u]["sep_over_std_at_verb"]) < SEP_MAX for u in ("3040", "114"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mismatch_population_specificity_result_v355", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
