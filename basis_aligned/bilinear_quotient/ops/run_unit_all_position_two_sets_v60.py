"""v60: v56's all-position removal + per-token collateral for the two v57 sets (verb_preposition, verb_complementizer),
so that all six behaviours on the terminal table carry the same rows. Instrument imported from v56 unchanged; the
final-only arm is controlled against v57's recorded final-position damage (same intervention, different code path).

REGISTERED BEFORE THE RUN
    pred_a_instrument      no-hook forward = g.forward_units to 1e-4; final-only own damage within 0.02 nat of v57 (0.201 / 0.583). Worked: 0.200 True.
    pred_b_final_carrier   all-position / final-only own answer-CE in [0.8, 1.5] for both sets. Worked: 1.1 True; 2.5 False.
    pred_c_offtarget_C     per-token CE increase on the C sentences UB <= 0.01 nat for both. Worked: 0.004 True.
    pred_d_cross_tokens    per-token CE on the other five sets' A1 sentences UB <= 0.10 x own answer damage on all 10 pairs.
                           Worked: 0.010 vs 0.20 True; 0.03 vs 0.20 False.
    pred_e_random          random rank-1, all positions, own per-token CE UB <= 0.01 nat. Worked: 0.002 True.
    Reading rule as v56: d False -> the relative bar and the absolute bar disagree for a weak set; report both, move neither.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23
import run_unit_all_position_removal_v56 as v56

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_all_position_two_sets_v60_result.json"
V57 = ROOT / "circuits/followups/unit_terminal_two_sets_v57_result.json"
NEW = {k: (v15.SETS[k][0], list(v15.SETS[k][1])) for k in ("verb_preposition", "verb_complementizer")}
ALL = {**NEW, **{k: (v[0], list(v[1])) for k, v in v23.SETS.items()}}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 80, 4000


def _plan():
    return {"candidate_id": "corpus.unit_all_position_two_sets_v60", "sets": {k: v[1] for k, v in NEW.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    t0 = time.perf_counter()
    v57 = json.loads(V57.read_text())["behaviours"]
    preps = {n: g.prepare(backend, g.rows_of(m, "A1")) for n, (m, _) in ALL.items()}
    c_prep = g.prepare(backend, g.rows_of(v23.SETS["polarity_licensing"][0], "C"))

    def natives(prep):
        return {side: v56.forward_all(backend, batch) for side, batch in (("base", prep.base_batch), ("donor", prep.donor_batch))}
    nat = {n: natives(p) for n, p in preps.items()}
    nat["C"] = natives(c_prep)

    def stat(x):
        p, lb, ub = v56.v50._boot(torch, x)
        return {"point": p, "lb975": lb, "ub975": ub, "documents": len(x)}

    report = {}
    for n, (m, units) in NEW.items():
        prep = preps[n]
        mine, _, lengths = v56.forward_all(backend, prep.base_batch)
        i = torch.arange(len(lengths), device=mine.device); last = torch.tensor([l - 1 for l in lengths], device=mine.device)
        af_mine = torch.stack([mine[i, last, torch.tensor(prep.base_batch.answer_ids, device=mine.device)],
                               mine[i, last, torch.tensor(prep.base_batch.foil_ids, device=mine.device)]], 1)
        instr = float((af_mine - g.forward_units(backend, prep.base_batch).float()).abs().max())
        spans = v56.spans_of(units)
        q = g.block_diff_in_means(backend, prep, units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        c_final, c_all = v56.coords(backend, prep, spans, q)
        cr_final, cr_all = v56.coords(backend, prep, spans, q_rand)

        def run(prep_, native, mode, qq=q, cf=c_final, ca=c_all):
            ans_d, tok_d = [], []
            for side, batch in (("base", prep_.base_batch), ("donor", prep_.donor_batch)):
                l0, tokens, lengths = native[side]
                l1, _, _ = v56.forward_all(backend, batch, v56.removal_hooks(torch, spans, qq, cf, ca, lengths, mode))
                i = torch.arange(len(lengths), device=l0.device); last = torch.tensor([l - 1 for l in lengths], device=l0.device)
                ans = torch.tensor(batch.answer_ids, device=l0.device)
                ans_d += (F.log_softmax(l0[i, last], -1)[i, ans] - F.log_softmax(l1[i, last], -1)[i, ans]).tolist()
                a, b = v56.per_token_ce(F, l0, tokens, lengths), v56.per_token_ce(F, l1, tokens, lengths)
                tok_d += [y - x for x, y in zip(a, b)]
            return {"answer_ce": stat(ans_d), "token_ce": stat(tok_d)}
        r = {"units": units, "instrument_max_abs_err": instr, "v57_final_only_ce": v57[n]["target_A1"]["ce_damage"],
             "own_final_only": run(prep, nat[n], "final"), "own_all": run(prep, nat[n], "all"), "C_all": run(c_prep, nat["C"], "all"),
             "own_random_all": run(prep, nat[n], "all", q_rand, cr_final, cr_all),
             "cross_all": {t: run(preps[t], nat[t], "all") for t in ALL if t != n}}
        r["carrier_ratio"] = r["own_all"]["answer_ce"]["point"] / r["own_final_only"]["answer_ce"]["point"]
        report[n] = r
        print(n, json.dumps({"instr": instr, "final_only": round(r["own_final_only"]["answer_ce"]["point"], 3), "v57": round(r["v57_final_only_ce"], 3),
                             "all": round(r["own_all"]["answer_ce"]["point"], 3), "ratio": round(r["carrier_ratio"], 2), "own_tok": round(r["own_all"]["token_ce"]["point"], 4),
                             "C_tok_ub": round(r["C_all"]["token_ce"]["ub975"], 4), "C_ans": round(r["C_all"]["answer_ce"]["point"], 3),
                             "rand_tok_ub": round(r["own_random_all"]["token_ce"]["ub975"], 4),
                             "cross_tok_ub": {t: round(c["token_ce"]["ub975"], 4) for t, c in r["cross_all"].items()}}), flush=True)
    R = report.values()
    predictions = {
        'pred_a_instrument': all(r["instrument_max_abs_err"] <= v56.INSTR_TOL and abs(r["own_final_only"]["answer_ce"]["point"] - r["v57_final_only_ce"]) <= v56.V51_TOL for r in R),
        'pred_b_final_carrier': all(v56.CARRIER_BAND[0] <= r["carrier_ratio"] <= v56.CARRIER_BAND[1] for r in R),
        'pred_c_offtarget_C': all(r["C_all"]["token_ce"]["ub975"] <= v56.OFF_UB for r in R),
        'pred_d_cross_tokens': all(c["token_ce"]["ub975"] <= v56.CROSS_FRAC * r["own_all"]["answer_ce"]["point"] for r in R for c in r["cross_all"].values()),
        'pred_e_random': all(r["own_random_all"]["token_ce"]["ub975"] <= v56.RAND_UB for r in R),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_all_position_removal_result_v1", "candidate_id": "corpus.unit_all_position_two_sets_v60",
              "bars": {"instr_tol": v56.INSTR_TOL, "v57_tol": v56.V51_TOL, "carrier_band": v56.CARRIER_BAND, "off_ub": v56.OFF_UB, "cross_frac": v56.CROSS_FRAC, "rand_ub": v56.RAND_UB},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
