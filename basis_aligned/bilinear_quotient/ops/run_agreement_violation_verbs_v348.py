#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_3465_fires_for_singular_then_plural_verb pred_c_3465_silent_for_the_other_three_cells pred_d_493_fires_for_plural_then_singular_verb pred_e_violation_reading_generalises_across_verbs
"""Agreement-violation detectors across verbs (v348). v347 (is / are): at the verb, unit 3465 fires ~ -300 only for "singular noun + are" and 493 +46 only for
"plural noun + is". Generalisation over four more verb pairs -- plural-agreeing " were", " have", " do", " go" vs singular-agreeing " was", " has", " does",
" goes" -- on the 256 vocabulary pairs (X singular / plural), read at the verb: per verb pair and unit, the four cell medians; the violation cells are
(singular X, plural verb) for 3465's reading and (plural X, singular verb) for 493's.
PREDICTIONS (scored as written; failures preserved; priors from v347)
    pred_a_closure                                    the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_3465_fires_for_singular_then_plural_verb   for every plural-agreeing verb, 3465's (singular X, plural verb) cell is below its (singular X, singular verb) cell by >= 1 pooled std
    pred_c_3465_silent_for_the_other_three_cells      for every verb pair, 3465's other three cells lie within 0.5 pooled std of each other
    pred_d_493_fires_for_plural_then_singular_verb    for every singular-agreeing verb, 493's (plural X, singular verb) cell is above its (plural X, plural verb) cell by >= 0.5 pooled std
    pred_e_violation_reading_generalises_across_verbs the sign pattern of v347 (3465 violation cell lowest; 493 violation cell highest) holds for >= 3 of the 4 verb pairs on both units
PRICE (registered maximum): 8 verbs x 512 rows / 256 = 16 forwards (blocks 0-8); 0 backwards; 0 fits. Bar <= 18.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/agreement_violation_verbs_v348_result.json"
CANDIDATE_ID = "pronoun_number.agreement_violation_verbs_v348"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, GAP_3465, NEAR, GAP_493 = 1e-4, 1.0, 0.5, 0.5
PAIRS_V = ((" was", " were"), (" has", " have"), (" does", " do"), (" goes", " go"))
VERBS = tuple(v for pr in PAIRS_V for v in pr)
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 18
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_3465_fires_for_singular_then_plural_verb": ">= 1 std x 4", "pred_c_3465_silent_for_the_other_three_cells": "within 0.5 std x 4", "pred_d_493_fires_for_plural_then_singular_verb": ">= 0.5 std x 4", "pred_e_violation_reading_generalises_across_verbs": ">= 3 of 4, both units"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gap_3465": GAP_3465, "near": NEAR, "gap_493": GAP_493}, "verb_pairs": [list(p) for p in PAIRS_V]}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    cells = {}; closure = 0.0
    for vb in VERBS:
        vid = L._single(vb); H, H2 = {}, {}
        with torch.no_grad():
            for s0 in range(0, len(tokens), 256):
                ids = torch.tensor([[t, vid] for t in tokens[s0:s0 + 256]], device="cuda")
                x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if l in UNITS:
                        hh = dod_units.hidden(model, block.mlp, xin)[:, 1].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 1].float().cpu()
                        for u in UNITS[l]: H.setdefault((l, u), []).append(hh[:, u]); H2.setdefault((l, u), []).append(hh2[:, u])
                    if l == TOP: break
                    x = x + block.mlp(xin)
                forwards += 1
        H = {kk: torch.cat(v) for kk, v in H.items()}; H2 = {kk: torch.cat(v) for kk, v in H2.items()}
        closure = max(closure, max(float(((H2[kk] - H[kk]).abs() / H[kk].abs().clamp_min(1e-6)).max()) for kk in H)); tindex = {t: i for i, t in enumerate(tokens)}
        for (l, u), h in H.items():
            cells[(f"mlp{l}.{u}", vb, "s")] = torch.tensor([float(h[tindex[a_]]) for a_, _ in pairs]); cells[(f"mlp{l}.{u}", vb, "p")] = torch.tensor([float(h[tindex[b_]]) for _, b_ in pairs])
    per = {}
    for key in sorted({k[0] for k in cells}):
        per[key] = {}
        for sg, pl in PAIRS_V:
            c = {(vb, n): cells[(key, vb, n)] for vb in (sg, pl) for n in ("s", "p")}; std = float(torch.cat([v - v.mean() for v in c.values()]).std())
            med = {f"{vb.strip()}|{n}": float(v.median()) for (vb, n), v in c.items()}
            per[key][f"{sg.strip()}/{pl.strip()}"] = {"medians": med, "pooled_std": std, "viol_3465_gap_std": (med[f"{pl.strip()}|s"] - med[f"{sg.strip()}|s"]) / std, "viol_493_gap_std": (med[f"{sg.strip()}|p"] - med[f"{pl.strip()}|p"]) / std,
                                                    "other_three_spread_std": (max(med[f"{sg.strip()}|s"], med[f"{sg.strip()}|p"], med[f"{pl.strip()}|p"]) - min(med[f"{sg.strip()}|s"], med[f"{sg.strip()}|p"], med[f"{pl.strip()}|p"])) / std}
    report = {"closure_max": closure, "per_unit": per}
    print(json.dumps({k: {pr: {a_: round(b_, 2) for a_, b_ in v.items() if a_ != "medians"} for pr, v in d.items()} for k, d in per.items() if k in ("mlp3.3465", "mlp3.493")}, indent=1))
    u3, u4 = per["mlp3.3465"], per["mlp3.493"]
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_3465_fires_for_singular_then_plural_verb": all(v["viol_3465_gap_std"] <= -GAP_3465 for v in u3.values()), "pred_c_3465_silent_for_the_other_three_cells": all(v["other_three_spread_std"] <= NEAR for v in u3.values()),
                   "pred_d_493_fires_for_plural_then_singular_verb": all(v["viol_493_gap_std"] >= GAP_493 for v in u4.values()), "pred_e_violation_reading_generalises_across_verbs": sum(v["viol_3465_gap_std"] < 0 for v in u3.values()) >= 3 and sum(v["viol_493_gap_std"] > 0 for v in u4.values()) >= 3}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "agreement_violation_verbs_result_v348", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
