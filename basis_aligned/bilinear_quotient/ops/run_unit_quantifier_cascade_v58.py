"""v58: the quantifier gate's 15 -> 16/17 cascade -- close the inclusion-exclusion over which downstream MLPs are nonlinear.

v43 (quantifier_number, set {07:08, 11:03}, stack mlp 11-14, downstream 15-17): all-linearised floor 0.09 of b; keep-ONE
layer nonlinear 15: 0.71, 16: 0.51, 17: -0.39 (floor included); all three 1.00. Singles sum to 0.84 (floor-corrected 0.56 vs
0.91), so the layers do not add: a product formed at 15 must feed products at 16/17 (a cascade), or 17's opposing term
must cancel differently in company. Here the three PAIR arms are measured on the same rows, and the pair excesses
    X(l,m) = S(l,m) - S(l) - S(m) + floor
and the three-way term  W = S(15,16,17) - sum pairs + sum singles - floor  close the decomposition exactly.
Four-run design as in v42-v55 (B, D, V, DV; I = rec(DV) - rec(D) - rec(V) + rec(B)); nonlinearity kept only in the named
layers, every other downstream MLP replaced by its first-order expansion around the base input (v43._linearised).

REGISTERED BEFORE THE RUN (shares of the unmasked I on the same 32 A1 rows)
    pred_a_control       keep-all-three nonlinear reproduces the unhooked I within 1e-3 of |I| and the all-linearised floor
                         matches v43's 0.093 within 0.02. Worked: 1e-6, 0.091 True.
    pred_b_cascade_15_16 X(15,16) >= +0.20 (a genuine 15 -> 16 cascade). Worked: 0.31 True; 0.05 False.
    pred_c_no_late_pair  |X(16,17)| <= 0.10 (no cascade between the two late layers without 15). Worked: 0.04 True; 0.25 False.
    pred_d_17_opposes    S(15,17) < S(15) (17's opposing nonlinearity persists when 15 is live). Worked: 0.60 < 0.71 True; 0.80 False.
    pred_e_three_way     |W| <= 0.10 (pairs suffice). Worked: 0.03 True; 0.30 False.
    Reading rule. b True and e True: the gate is "product at 15, re-multiplied at 16" -- a two-stage bilinear cascade, the
    Tier-4 statement for quantifier. b False: 16's excess share is its own product of stack signals (parallel, not cascade)
    and the v43 non-additivity is 17's cancellation -- report as parallel formation. e False: the gate is irreducibly
    three-layer; no smaller statement.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_downstream_linearisation_v43 as v43
import run_unit_gate_hidden_units_v55 as v55

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_quantifier_cascade_v58_result.json"
V43 = ROOT / "circuits/followups/unit_downstream_linearisation_v43_result.json"
NAME = "quantifier_number"
CTRL_TOL, FLOOR_TOL, CASCADE_MIN, LATE_MAX, THREE_MAX = 1e-3, 0.02, 0.20, 0.10, 0.10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 2000


def _plan():
    return {"candidate_id": "corpus.unit_quantifier_cascade_v58", "set": v35.SETS[NAME][1], "stack": v35.STACK[NAME],
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    t0 = time.perf_counter()
    module, units = v35.SETS[NAME]
    stack_mlps = [m for m in v35.STACK[NAME] if m.startswith("mlp:")]
    prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
    S = v55.setup(backend, prep, units, stack_mlps)
    down = S["down"]

    def four(keep):
        """Nonlinearity kept in `keep`; other downstream MLPs linearised. keep=None -> no hooks."""
        recs = {}
        for k, (w_, vec) in S["RUNS"].items():
            lin = [l for l in down if keep is not None and l not in keep]
            with ExitStack() as es:
                if lin:
                    es.enter_context(v43._linearised(torch, model, lin, S["positions"], backend.device, S["u_base"], S["out_base"]))
                rec, _, _ = v30._capture(backend, prep, down, **S["cfg"](w_, vec))
            recs[k] = rec
        return recs["DV"] - recs["D"] - recs["V"] + recs["B"]

    I_full = four(None)
    arms = {"all_three": four(set(down)), "floor": four(set())}
    for r in (1, 2):
        for keep in itertools.combinations(down, r):
            arms["keep_" + "_".join(str(l) for l in keep)] = four(set(keep))
    sh = {k: v_ / I_full for k, v_ in arms.items()}
    Sg = lambda *ls: sh["keep_" + "_".join(str(l) for l in ls)]
    floor = sh["floor"]
    X = {f"{l}_{m}": Sg(l, m) - Sg(l) - Sg(m) + floor for l, m in itertools.combinations(down, 2)}
    W = sh["all_three"] - sum(Sg(l, m) for l, m in itertools.combinations(down, 2)) + sum(Sg(l) for l in down) - floor
    v43_floor = json.loads(V43.read_text())["behaviours"][NAME]["share_all_linearised"]
    a, b, c = down
    predictions = {
        'pred_a_control': abs(arms["all_three"] - I_full) <= CTRL_TOL * abs(I_full) and abs(floor - v43_floor) <= FLOOR_TOL,
        'pred_b_cascade_15_16': X[f"{a}_{b}"] >= CASCADE_MIN,
        'pred_c_no_late_pair': abs(X[f"{b}_{c}"]) <= LATE_MAX,
        'pred_d_17_opposes': Sg(a, c) < Sg(a),
        'pred_e_three_way': abs(W) <= THREE_MAX,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_quantifier_cascade_result_v1", "candidate_id": "corpus.unit_quantifier_cascade_v58",
              "set": list(units), "stack_mlps": stack_mlps, "downstream": down, "rows": len(prep.rows), "I_full": I_full,
              "bars": {"ctrl_tol": CTRL_TOL, "floor_tol": FLOOR_TOL, "cascade_min": CASCADE_MIN, "late_max": LATE_MAX, "three_max": THREE_MAX},
              "interactions": arms, "shares": sh, "pair_excess": X, "three_way": W, "v43_floor": v43_floor,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "I_full": round(I_full, 5), "shares": {k: round(v_, 3) for k, v_ in sh.items()},
                      "pair_excess": {k: round(v_, 3) for k, v_ in X.items()}, "three_way": round(W, 3), "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
