#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_selection_contrasts_share_an_axis_at_13_8_and_8_8 pred_b_pronoun_contrasts_share_an_axis_at_core_heads pred_c_number_and_selection_orthogonal_at_7_8 pred_d_pronoun_and_temporal_orthogonal_at_15_1 pred_e_cross_family_unembedding_cosines_small
"""Readout-direction geometry across the four families at the heads they share (v95, weights only, CPU; v69's construction).

For each head h and contrast c, d_h(c) = O_h^T (u_a - u_b). Contrasts: pronoun {he-she, they-he}, selection {in-of, up-down},
number {were-was}, temporal {has-had, will-had}. Heads: the pronoun core {9.6, 12.4, 15.1} (+10.1, 10.5), the selection core {13.8,
7.8, 8.8} (+6.3, 14.8), and the number/temporal shared heads {11.3, 5.7, 9.7}. Head 7.8 sits in the number AND selection cores;
15.1 in the pronoun core and many temporal lines. Tag: fold (weights only), 0 forwards.

PREDICTIONS (scored as written; failures preserved; priors unsure throughout)
    pred_a_selection_contrasts_share_an_axis_at_13_8_and_8_8   |cos(in-of, up-down)| >= 0.30 at 13.8 and at 8.8
    pred_b_pronoun_contrasts_share_an_axis_at_core_heads        |cos(he-she, they-he)| >= 0.30 at each of 9.6, 12.4, 15.1
    pred_c_number_and_selection_orthogonal_at_7_8               |cos(were-was, in-of)| and |cos(were-was, up-down)| <= 0.25 at 7.8
    pred_d_pronoun_and_temporal_orthogonal_at_15_1              |cos(he-she, has-had)|, |cos(he-she, will-had)|, |cos(they-he, has-had)|,
                                                                |cos(they-he, will-had)| <= 0.25 at 15.1
    pred_e_cross_family_unembedding_cosines_small               every cross-family |cos| at the unembedding itself <= 0.25 (else the
                                                                head-level cosines are inherited from the vocabulary, not the head)
PRICE: 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import itertools, json, os
from pathlib import Path
import aspectual_dod_lib as L

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/family_geometry_v95_result.json"
CANDIDATE_ID = "corpus.family_geometry_v95"
CONTRASTS = {"he-she": (" he", " she"), "they-he": (" they", " he"), "in-of": (" in", " of"), "up-down": (" up", " down"), "were-was": (" were", " was"), "has-had": (" has", " had"), "will-had": (" will", " had")}
FAMILY = {"he-she": "pronoun", "they-he": "pronoun", "in-of": "selection", "up-down": "selection", "were-was": "number", "has-had": "temporal", "will-had": "temporal"}
HEADS = ((9, 6), (12, 4), (15, 1), (10, 1), (10, 5), (13, 8), (7, 8), (8, 8), (6, 3), (14, 8), (11, 3), (5, 7), (9, 7))
PREDICTIONS = {"pred_a_selection_contrasts_share_an_axis_at_13_8_and_8_8": ">= 0.30", "pred_b_pronoun_contrasts_share_an_axis_at_core_heads": ">= 0.30 x 3",
               "pred_c_number_and_selection_orthogonal_at_7_8": "<= 0.25", "pred_d_pronoun_and_temporal_orthogonal_at_15_1": "<= 0.25 x 4", "pred_e_cross_family_unembedding_cosines_small": "<= 0.25"}


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
    cross_unemb = max(abs(v) for k, v in out["unembedding"].items() if FAMILY[k.split("|")[0]] != FAMILY[k.split("|")[1]])
    predictions = {"pred_a_selection_contrasts_share_an_axis_at_13_8_and_8_8": all(c(h, "in-of", "up-down") >= 0.30 for h in ("13.8", "8.8")),
                   "pred_b_pronoun_contrasts_share_an_axis_at_core_heads": all(c(h, "he-she", "they-he") >= 0.30 for h in ("9.6", "12.4", "15.1")),
                   "pred_c_number_and_selection_orthogonal_at_7_8": all(c("7.8", "were-was", s) <= 0.25 for s in ("in-of", "up-down")),
                   "pred_d_pronoun_and_temporal_orthogonal_at_15_1": all(c("15.1", p, t) <= 0.25 for p in ("he-she", "they-he") for t in ("has-had", "will-had")),
                   "pred_e_cross_family_unembedding_cosines_small": cross_unemb <= 0.25}
    print("unembedding", {k: round(v, 2) for k, v in out["unembedding"].items()}, "cross-family max", round(cross_unemb, 3))
    OUT.write_text(json.dumps({"schema": "family_geometry_result_v95", "candidate_id": CANDIDATE_ID, **out, "cross_family_unembedding_max_abs_cos": cross_unemb, "predictions": predictions, "forwards": 0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
