#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_temporal_contrasts_share_an_axis_at_family_heads pred_b_number_contrast_orthogonal_to_temporal_axis_at_temporal_heads pred_c_shared_head_11_3_carries_both_axes
"""Readout-direction geometry at the shared heads (v69, weights only, CPU): are the four temporal/mood/aspect readout contrasts
one axis at {9.1, 9.4, 15.5} and how does the number contrast sit at 11.3?

For each head h and contrast c in {has−had, will−had, was−is, would−will, were−was}, d_h(c) = O_h^T (u_a − u_b). Report the
pairwise cosine matrix per head. Tag: fold (weights only), no activations.

PREDICTIONS (scored as written)
    pred_a_temporal_contrasts_share_an_axis_at_family_heads   at each of 9.1, 9.4, 15.5 the mean |cos| among the four temporal
                                                contrasts (oriented so that the past/remote member is second) is >= 0.40
    pred_b_number_contrast_orthogonal_to_temporal_axis_at_temporal_heads   at 9.1, 9.4, 15.5: |cos(were−was, c)| <= 0.25 for every
                                                temporal contrast c
    pred_c_shared_head_11_3_carries_both_axes   at 11.3 the number contrast has |cos| >= 0.30 with at least one temporal contrast
                                                and the temporal contrasts' mean |cos| >= 0.40
PRICE: 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import itertools, json, os
from pathlib import Path
import aspectual_dod_lib as L

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/readout_geometry_v69_result.json"
CANDIDATE_ID = "corpus.readout_geometry_v69"
CONTRASTS = {"has-had": (" has", " had"), "will-had": (" will", " had"), "is-was": (" is", " was"), "will-would": (" will", " would"), "were-was": (" were", " was")}
TEMPORAL = ("has-had", "will-had", "is-was", "will-would")
HEADS = ((9, 1), (9, 4), (15, 5), (11, 3), (8, 1), (5, 7), (7, 8), (9, 7))


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "cpu_lane"}, indent=2)); return
    import torch, fastload
    m = fastload.load_model_fast().eval(); W = m.lm_head.weight.detach().float()
    u = {k: W[L._single(a)] - W[L._single(b)] for k, (a, b) in CONTRASTS.items()}
    cos = lambda a, b: float((a @ b) / (a.norm() * b.norm()))
    out = {"unembedding": {f"{a}|{b}": cos(u[a], u[b]) for a, b in itertools.combinations(CONTRASTS, 2)}, "heads": {}}
    for layer, head in HEADS:
        O = m.transformer.h[layer].attn.c_proj.weight.detach().float()[:, head * 128:(head + 1) * 128]
        d = {k: O.T @ v for k, v in u.items()}
        out["heads"][f"{layer}.{head}"] = {f"{a}|{b}": cos(d[a], d[b]) for a, b in itertools.combinations(CONTRASTS, 2)}
    def mean_abs_temporal(h):
        vals = [abs(out["heads"][h][f"{a}|{b}"]) for a, b in itertools.combinations(TEMPORAL, 2)]; return sum(vals) / len(vals)
    def max_abs_number(h):
        return max(abs(out["heads"][h][f"{a}|were-was"]) if f"{a}|were-was" in out["heads"][h] else abs(out["heads"][h][f"were-was|{a}"]) for a in TEMPORAL)
    for h in out["heads"]:
        print(h, "temporal mean|cos|", round(mean_abs_temporal(h), 3), "number vs temporal max|cos|", round(max_abs_number(h), 3), {k: round(v, 2) for k, v in out["heads"][h].items()})
    predictions = {"pred_a_temporal_contrasts_share_an_axis_at_family_heads": all(mean_abs_temporal(h) >= 0.40 for h in ("9.1", "9.4", "15.5")),
                   "pred_b_number_contrast_orthogonal_to_temporal_axis_at_temporal_heads": all(max_abs_number(h) <= 0.25 for h in ("9.1", "9.4", "15.5")),
                   "pred_c_shared_head_11_3_carries_both_axes": max_abs_number("11.3") >= 0.30 and mean_abs_temporal("11.3") >= 0.40}
    OUT.write_text(json.dumps({"schema": "readout_geometry_result_v69", "candidate_id": CANDIDATE_ID, **out, "predictions": predictions, "forwards": 0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
