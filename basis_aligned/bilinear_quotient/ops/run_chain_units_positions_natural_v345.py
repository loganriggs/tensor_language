#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_rows_load pred_b_cue_separation_replays_v344 pred_c_no_unit_separates_at_the_final pred_d_verb_position_weaker_than_cue pred_e_829_or_1036_carry_to_the_verb
"""The agreement units at the cue noun, the verb and the final position of natural sentences (v345). v334 (panel): the chain's units separate plural from
singular at the noun and not at the answer position. The 128 natural verb rows carry a verb between the cue noun and the pronoun (miner's second_offset).
Read h at the cue (cue_offset), at the verb (second_offset) and at the final token for 3465 / 493 / 1036 / 829 / 953 / 1030; separation of plural-cue from
singular-cue rows in pooled std at each position.
PREDICTIONS (scored as written; failures preserved; priors from v334 / v344)
    pred_a_rows_load                      128 rows with valid cue and verb offsets (instrument)
    pred_b_cue_separation_replays_v344    at the cue, 3465 / 493 / 1036 / 829 replay v344's separations within 0.1 std
    pred_c_no_unit_separates_at_the_final at the final token no unit separates by >= 1 pooled std (as on the panel)
    pred_d_verb_position_weaker_than_cue  at the verb every unit's |separation| is below its cue value
    pred_e_829_or_1036_carry_to_the_verb  at the verb, 829 or 1036 still separates by >= 0.5 pooled std (a trace of the number state at the agreement site). Prior: unsure.
PRICE (registered maximum): 2 natural batches (blocks 0-8) = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/chain_units_positions_natural_v345_result.json"
CANDIDATE_ID = "pronoun_number.chain_units_positions_natural_v345"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; BATCH = 64
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
V344 = {3465: -2.19, 493: 1.27, 1036: 0.98, 829: 1.65}; REPLAY_TOL, FINAL_MAX, VERB_MIN = 0.1, 1.0, 0.5
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_rows_load": "128 rows", "pred_b_cue_separation_replays_v344": "within 0.1 std x 4", "pred_c_no_unit_separates_at_the_final": "< 1 std x 6", "pred_d_verb_position_weaker_than_cue": "|verb| < |cue| x 6", "pred_e_829_or_1036_carry_to_the_verb": ">= 0.5 std"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    cue = [int(r["cue_offset"]) for r in recs]; verb = [int(r["second_offset"]) for r in recs]; fin = [len(r["ids"]) - 1 for r in recs]; plural = [i for i, r in enumerate(recs) if r["cue"] == "plural"]; sing = [i for i, r in enumerate(recs) if r["cue"] != "plural"]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "plural_rows": len(plural), "singular_rows": len(sing), "units": {str(k): list(v) for k, v in UNITS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "final_max": FINAL_MAX, "verb_min": VERB_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); POS = {"cue": cue, "verb": verb, "final": fin}; H = {(p_, u): [] for p_ in POS for l in UNITS for u in UNITS[l]}
    with torch.no_grad():
        for s0 in range(0, len(recs), BATCH):
            chunk = nat[s0:s0 + BATCH]; idx = torch.arange(chunk.shape[0])
            x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in UNITS:
                    hh = dod_units.hidden(model, block.mlp, xin).float().cpu()
                    for p_, offs in POS.items():
                        pn = torch.tensor(offs[s0:s0 + BATCH])
                        for u in UNITS[l]: H[(p_, u)].append(hh[idx, pn, u])
                x = x + block.mlp(xin)
                if l == 8: break
            forwards += 1
    seps = {}
    for (p_, u), v in H.items():
        h = torch.cat(v); a_, b_ = h[plural], h[sing]; std = float(torch.cat([a_ - a_.mean(), b_ - b_.mean()]).std()); seps.setdefault(p_, {})[str(u)] = float((a_.median() - b_.median()) / std)
    report = {"separation_over_std": seps}
    print(json.dumps(report, indent=1))
    units = [u for l in UNITS for u in UNITS[l]]
    predictions = {"pred_a_rows_load": len(recs) == 128 and all(0 <= v_ < len(recs[i]["ids"]) for i, v_ in enumerate(verb)), "pred_b_cue_separation_replays_v344": all(abs(seps["cue"][str(u)] - V344[u]) <= REPLAY_TOL for u in V344),
                   "pred_c_no_unit_separates_at_the_final": all(abs(seps["final"][str(u)]) < FINAL_MAX for u in units), "pred_d_verb_position_weaker_than_cue": all(abs(seps["verb"][str(u)]) < abs(seps["cue"][str(u)]) for u in units),
                   "pred_e_829_or_1036_carry_to_the_verb": max(abs(seps["verb"]["829"]), abs(seps["verb"]["1036"])) >= VERB_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "chain_units_positions_natural_result_v345", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
