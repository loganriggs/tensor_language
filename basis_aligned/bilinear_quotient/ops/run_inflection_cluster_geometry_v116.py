#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_number_contrasts_share_an_axis_at_cluster_heads pred_b_finiteness_and_mood_orthogonal_to_number_at_cluster_heads pred_c_finiteness_and_mood_share_an_axis pred_d_cross_type_unembedding_cosines_small
"""Readout-direction geometry of the atlas residue cluster {11.3, 7.8, 17.4} (v116, weights only, CPU; v69's construction).

Seven live atlas lines outside the five families share 11.3 + 7.8 + 17.4 but split by contrast type: NUMBER (partitive has-have, do-does,
lifts-lift), FINITENESS (adjective eager-sure -> to-that, finiteness selection) and MOOD (mandative be-was, requested/adjective subjunctive).
Is this one family (one axis per head) or several decisions sharing heads (as temporal and number share 11.3, v69)? d_h(c) = O_h^T (u_a - u_b)
for c in {has-have, do-does, to-that, be-was} plus the number family's were-was and the temporal has-had as references, at 11.3, 7.8, 17.4,
13.1, 6.3 and the number core 5.7 / 9.7. Tag: fold (weights only), 0 forwards.

PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_number_contrasts_share_an_axis_at_cluster_heads     |cos(has-have, do-does)| >= 0.30 at each of 11.3, 7.8, 17.4
    pred_b_finiteness_and_mood_orthogonal_to_number_at_cluster_heads   |cos(to-that, has-have)|, |cos(be-was, has-have)| <= 0.25 at 11.3, 7.8, 17.4
    pred_c_finiteness_and_mood_share_an_axis                   |cos(to-that, be-was)| >= 0.30 at 11.3, 7.8, 17.4
    pred_d_cross_type_unembedding_cosines_small                every cross-type |cos| at the unembedding <= 0.25
PRICE: 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import itertools, json, os
from pathlib import Path
import aspectual_dod_lib as L

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/inflection_cluster_geometry_v116_result.json"
CANDIDATE_ID = "corpus.inflection_cluster_geometry_v116"
CONTRASTS = {"has-have": (" has", " have"), "do-does": (" do", " does"), "to-that": (" to", " that"), "be-was": (" be", " was"), "were-was": (" were", " was"), "has-had": (" has", " had")}
TYPE = {"has-have": "number", "do-does": "number", "were-was": "number", "to-that": "finiteness", "be-was": "mood", "has-had": "temporal"}
HEADS = ((11, 3), (7, 8), (17, 4), (13, 1), (6, 3), (5, 7), (9, 7))
CLUSTER = ("11.3", "7.8", "17.4")
PREDICTIONS = {"pred_a_number_contrasts_share_an_axis_at_cluster_heads": ">= 0.30 x 3", "pred_b_finiteness_and_mood_orthogonal_to_number_at_cluster_heads": "<= 0.25",
               "pred_c_finiteness_and_mood_share_an_axis": ">= 0.30 x 3", "pred_d_cross_type_unembedding_cosines_small": "<= 0.25"}


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "cpu_lane"}, indent=2)); return
    import torch, fastload
    m = fastload.load_model_fast().eval(); W = m.lm_head.weight.detach().float()
    u = {k: W[L._single(a)] - W[L._single(b)] for k, (a, b) in CONTRASTS.items()}
    cos = lambda a, b: float((a @ b) / (a.norm() * b.norm()))
    pairs = list(itertools.combinations(CONTRASTS, 2))
    out = {"unembedding": {f"{a}|{b}": cos(u[a], u[b]) for a, b in pairs}, "heads": {}}
    for layer, head in HEADS:
        O = m.transformer.h[layer].attn.c_proj.weight.detach().float()[:, head * 128:(head + 1) * 128]
        d = {k: O.T @ v for k, v in u.items()}
        out["heads"][f"{layer}.{head}"] = {f"{a}|{b}": cos(d[a], d[b]) for a, b in pairs}
        print(f"{layer}.{head}", {k: round(v, 2) for k, v in out["heads"][f"{layer}.{head}"].items()})
    def c(h, a, b):
        t = out["heads"][h]; return abs(t.get(f"{a}|{b}", t.get(f"{b}|{a}")))
    cross = max(abs(v) for k, v in out["unembedding"].items() if TYPE[k.split("|")[0]] != TYPE[k.split("|")[1]])
    predictions = {"pred_a_number_contrasts_share_an_axis_at_cluster_heads": all(c(h, "has-have", "do-does") >= 0.30 for h in CLUSTER),
                   "pred_b_finiteness_and_mood_orthogonal_to_number_at_cluster_heads": all(c(h, x, "has-have") <= 0.25 for h in CLUSTER for x in ("to-that", "be-was")),
                   "pred_c_finiteness_and_mood_share_an_axis": all(c(h, "to-that", "be-was") >= 0.30 for h in CLUSTER),
                   "pred_d_cross_type_unembedding_cosines_small": cross <= 0.25}
    print("unembedding", {k: round(v, 2) for k, v in out["unembedding"].items()}, "cross-type max", round(cross, 3))
    OUT.write_text(json.dumps({"schema": "inflection_cluster_geometry_result_v116", "candidate_id": CANDIDATE_ID, **out, "cross_type_unembedding_max_abs_cos": cross, "predictions": predictions, "forwards": 0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
