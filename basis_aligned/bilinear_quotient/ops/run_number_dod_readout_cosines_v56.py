#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_number_heads_overlap pred_b_temporal_heads_orthogonal pred_c_unembedding_overlap_smaller
"""Number family DoD, step 2 (v56, weights only, CPU): why does the number readout removal move the has−had reader?

For each head, the per-head readout directions `O_h^T(u_were − u_was)` and `O_h^T(u_has − u_had)` are weight objects;
their cosine says whether the two contrasts share the head's output subspace. Tag: fold (no activations).

PREDICTIONS (scored as written)
    pred_a_number_heads_overlap        |cos| >= 0.40 for 11.3, 7.8, 9.7 (the number heads that also read tense)
    pred_b_temporal_heads_orthogonal   |cos| <= 0.15 for 9.1, 9.4, 15.5
    pred_c_unembedding_overlap_smaller |cos| at the unembedding itself is smaller than at every number head above
PRICE: 0 forwards; weights only.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os
from pathlib import Path
import torch
import aspectual_dod_lib as L

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_readout_cosines_v56_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_readout_cosines_v56"
NUMBER, TEMPORAL = ((11, 3), (7, 8), (9, 7), (5, 7)), ((9, 1), (9, 4), (15, 5))


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "cpu_lane"}, indent=2)); return
    import fastload
    m = fastload.load_model_fast().eval()
    W = m.lm_head.weight.detach().float()
    u_num = W[L._single(" were")] - W[L._single(" was")]; u_tense = W[L._single(" has")] - W[L._single(" had")]
    cos = lambda a, b: float((a @ b) / (a.norm() * b.norm()))
    out = {"unembedding": cos(u_num, u_tense), "heads": {}}
    for layer, head in NUMBER + TEMPORAL:
        O = m.transformer.h[layer].attn.c_proj.weight.detach().float()[:, head * 128:(head + 1) * 128]
        out["heads"][f"{layer}.{head}"] = {"cos_were_was__has_had": cos(O.T @ u_num, O.T @ u_tense), "norm_ratio": float((O.T @ u_num).norm() / (O.T @ u_tense).norm())}
    print(json.dumps(out, indent=1))
    h = out["heads"]
    predictions = {"pred_a_number_heads_overlap": all(abs(h[k]["cos_were_was__has_had"]) >= 0.40 for k in ("11.3", "7.8", "9.7")),
                   "pred_b_temporal_heads_orthogonal": all(abs(h[k]["cos_were_was__has_had"]) <= 0.15 for k in ("9.1", "9.4", "15.5")),
                   "pred_c_unembedding_overlap_smaller": all(abs(out["unembedding"]) < abs(h[k]["cos_were_was__has_had"]) for k in ("11.3", "7.8", "9.7"))}
    OUT.write_text(json.dumps({"schema": "number_family_dod_readout_cosines_result_v56", "candidate_id": CANDIDATE_ID, **out, "predictions": predictions, "forwards": 0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
