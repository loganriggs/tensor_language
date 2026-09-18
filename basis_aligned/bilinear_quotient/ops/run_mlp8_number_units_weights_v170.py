#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_down_columns_align_with_reader pred_b_one_factor_reads_the_embedding_gender_axis pred_c_units_are_gender_specific_on_the_vocabulary
"""What do MLP-8 units 829, 953 and 1030 compute? (v170, weights + token embeddings, CPU, 0 row forwards.) v167's construction on the NUMBER line: v168 nominated them (65% of MLP 8's number write on 9.6's they-he direction); pairs are the v76 / v158 / v104-style agent nouns in plural vs singular form; g = mean rms(E[plural]) - rms(E[singular]); "male" below means PLURAL and "female" SINGULAR (names kept from v167).

v164 nominated them (76% of MLP 8's gender write on 9.6's reader direction r at the noun); v165 sized them (3% of the behaviour). A bilinear unit is
u_j(x) = (L_j . x)(R_j . x), written back as Down[:, j] u_j(x). Weights only: (a) Down[:, j] should point along r (that is why the units carry r's
contrast); (b) one of L_j, R_j should read a gender axis of the residual -- here tested against the block-0 embedding gender axis
g = mean over the 26 gender pairs of rms(E[male]) - rms(E[female]) (the noun position's residual at block 8 contains the embedding directly,
v82: 33%); (c) over the vocabulary, u_j(rms(E[token])) should separate male from female kin/title nouns: for the 26 pairs, sign(u_j(male) -
u_j(female)) consistent for >= 85% of the single-token pairs for at least one of the two units.
PREDICTIONS (scored as written; failures preserved; priors unsure): pred_a |cos(Down[:, j], r)| >= 0.30 for both units; pred_b max(|cos(L_j, g)|,
|cos(R_j, g)|) >= 0.20 for both units; pred_c the sign test above.
PRICE: 0 forwards.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os
from pathlib import Path
import aspectual_dod_lib as L
import dod_lexicon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/mlp8_number_units_weights_v170_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.mlp8_number_units_weights_v170"
UNITS = (829, 953, 1030)
PREDICTIONS = {"pred_a_down_columns_align_with_reader": ">= 0.30", "pred_b_one_factor_reads_the_embedding_gender_axis": ">= 0.20", "pred_c_units_are_gender_specific_on_the_vocabulary": ">= 85% of pairs"}


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "cpu_lane"}, indent=2)); return
    import torch, fastload
    F = torch.nn.functional
    m = fastload.load_model_fast().eval()
    W = m.lm_head.weight.detach().float(); E = m.transformer.wte.weight.detach().float()
    mlp = m.transformer.h[8].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float(), mlp.Right.weight.detach().float(), mlp.Down.weight.detach().float()
    # 9.6's reader direction r = V^T v_hat with v_hat = O^T (u_he - u_she) normalized, from weights
    attn9 = m.transformer.h[9].attn; O = attn9.c_proj.weight.detach().float()[:, 6 * 128:7 * 128]; V = attn9.c_v.weight.detach().float()[6 * 128:7 * 128, :]
    v_hat = O.T @ (W[L._single(" they")] - W[L._single(" he")]); v_hat = v_hat / v_hat.norm(); r = V.T @ v_hat
    cos = lambda a, b: float((a @ b) / (a.norm() * b.norm()))
    def single(w):
        try: L._single(" " + w); return True
        except L.RowError: return False
    pairs = [(a + "s", a) for a in dod_lexicon.AGENT_POOL if single(a) and single(a + "s")][:40]   # (plural, singular) agent nouns from the screened pool
    emb = lambda w: F.rms_norm(E[L._single(" " + w)], (E.shape[1],))
    gdir = torch.stack([emb(a) - emb(b) for a, b in pairs]).mean(0)
    report = {}
    for j in UNITS:
        u = lambda x: float((Lw[j] @ x) * (Rw[j] @ x))
        signs = [u(emb(a)) - u(emb(b)) for a, b in pairs]
        report[str(j)] = {"cos_down_r": cos(Dw[:, j], r), "cos_L_g": cos(Lw[j], gdir), "cos_R_g": cos(Rw[j], gdir), "cos_L_R": cos(Lw[j], Rw[j]),
                          "male_minus_female_positive": sum(1 for s in signs if s > 0), "male_minus_female_negative": sum(1 for s in signs if s < 0), "n_pairs": len(pairs),
                          "unit_values_examples": {f"{a}/{b}": (round(u(emb(a)), 3), round(u(emb(b)), 3)) for a, b in pairs[:6]}}
        print(j, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report[str(j)].items() if k != "unit_values_examples"}, report[str(j)]["unit_values_examples"])
    predictions = {"pred_a_down_columns_align_with_reader": all(abs(report[str(j)]["cos_down_r"]) >= 0.30 for j in UNITS),
                   "pred_b_one_factor_reads_the_embedding_gender_axis": all(max(abs(report[str(j)]["cos_L_g"]), abs(report[str(j)]["cos_R_g"])) >= 0.20 for j in UNITS),
                   "pred_c_units_are_gender_specific_on_the_vocabulary": any(max(report[str(j)]["male_minus_female_positive"], report[str(j)]["male_minus_female_negative"]) >= 0.85 * len(pairs) for j in UNITS)}
    OUT.write_text(json.dumps({"schema": "mlp8_number_units_weights_result_v170", "candidate_id": CANDIDATE_ID, "units": report, "predictions": predictions, "forwards": 0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
