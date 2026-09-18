#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_down_column_feeds_unit_3152 pred_b_unit_is_a_one_sided_gender_detector pred_c_one_factor_reads_the_embedding_gender_axis
"""What does MLP-6 unit 3230 compute? (v174, weights + token embeddings, CPU.) v173: it carries ~82% of MLP 6's input to the male-noun detector 3152.
(a) Its Down column should project onto 3152's Left / Right rows (that is how it feeds the detector); (b) on raw noun embeddings its bilinear value
should separate male from female nouns one-sidedly (like 3152 / 3943); (c) one of its factors should read the embedding gender axis g.
PREDICTIONS (scored as written; failures preserved; priors unsure): pred_a |cos(Down6[:, 3230], L_3152)| >= 0.30 or |cos(Down6[:, 3230], R_3152)| >= 0.30;
pred_b sign(u(male) - u(female)) consistent on >= 85% of single-token pairs; pred_c max(|cos(L_3230, g)|, |cos(R_3230, g)|) >= 0.20.
PRICE: 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os
from pathlib import Path
import aspectual_dod_lib as L
import run_pronoun_gender_dod_battery_v71 as g71
import circuit_fast_screen_candidate_pronoun_gender as pg

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/mlp6_unit3230_weights_v174_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.mlp6_unit3230_weights_v174"
UNIT, TARGET = 3230, 3152
PREDICTIONS = {"pred_a_down_column_feeds_unit_3152": ">= 0.30", "pred_b_unit_is_a_one_sided_gender_detector": ">= 85% of pairs", "pred_c_one_factor_reads_the_embedding_gender_axis": ">= 0.20"}


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "cpu_lane"}, indent=2)); return
    import torch, fastload
    F = torch.nn.functional
    m = fastload.load_model_fast().eval(); E = m.transformer.wte.weight.detach().float()
    m6, m8 = m.transformer.h[6].mlp, m.transformer.h[8].mlp
    L6, R6, D6 = m6.Left.weight.detach().float()[UNIT], m6.Right.weight.detach().float()[UNIT], m6.Down.weight.detach().float()[:, UNIT]
    L8, R8 = m8.Left.weight.detach().float()[TARGET], m8.Right.weight.detach().float()[TARGET]
    cos = lambda a, b: float((a @ b) / (a.norm() * b.norm()))
    def single(w):
        try: L._single(" " + w); return True
        except L.RowError: return False
    pairs = [(a, b) for a, b in list(pg.GENDER) + list(g71.PAIRS) if single(a) and single(b)]
    emb = lambda w: F.rms_norm(E[L._single(" " + w)], (E.shape[1],))
    gdir = torch.stack([emb(a) - emb(b) for a, b in pairs]).mean(0)
    u = lambda x: float((L6 @ x) * (R6 @ x))
    signs = [u(emb(a)) - u(emb(b)) for a, b in pairs]
    report = {"cos_down_L3152": cos(D6, L8), "cos_down_R3152": cos(D6, R8), "cos_L_g": cos(L6, gdir), "cos_R_g": cos(R6, gdir), "cos_L_R": cos(L6, R6),
              "male_minus_female_positive": sum(1 for s in signs if s > 0), "male_minus_female_negative": sum(1 for s in signs if s < 0), "n_pairs": len(pairs),
              "unit_values_examples": {f"{a}/{b}": (round(u(emb(a)), 3), round(u(emb(b)), 3)) for a, b in pairs[:8]}}
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items() if k != "unit_values_examples"}); print(report["unit_values_examples"])
    predictions = {"pred_a_down_column_feeds_unit_3152": max(abs(report["cos_down_L3152"]), abs(report["cos_down_R3152"])) >= 0.30,
                   "pred_b_unit_is_a_one_sided_gender_detector": max(report["male_minus_female_positive"], report["male_minus_female_negative"]) >= 0.85 * len(pairs),
                   "pred_c_one_factor_reads_the_embedding_gender_axis": max(abs(report["cos_L_g"]), abs(report["cos_R_g"])) >= 0.20}
    OUT.write_text(json.dumps({"schema": "mlp6_unit3230_weights_result_v174", "candidate_id": CANDIDATE_ID, "unit": report, "predictions": predictions, "forwards": 0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
