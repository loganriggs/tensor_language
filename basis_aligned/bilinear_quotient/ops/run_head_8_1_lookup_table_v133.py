#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_lookup_sign_matches_every_registered_decision pred_b_registered_pairs_beat_random_token_pairs pred_c_lookup_is_contrast_specific
"""Head 8.1 as ONE token lookup serving four families (v133, weights + single-token block-0 values, CPU, 0 model forwards on rows).

The temporal line (v12 / v15), the person lines (v113 / v117) and the correlative lines (v131 / v132) each closed head 8.1 to the token-only
term p x lamb x v1(cue): its service is a lookup on the cue token's block-0 value, read out through O_{8.1} along a contrast-specific direction.
This run asks whether that is one table: for each family's registered cue pair (a, b) and its decision contrast (u_pos - u_neg), the score
S = (u_pos - u_neg) . O_{8.1} (v1(a) - v1(b)) must have the decision's sign; its magnitude is compared with the same projection of 16 random
vocabulary token pairs (the null); and the cross-talk of each cue pair onto the OTHER families' contrasts is reported. v1(token) is the block-0
value of a one-token input (position 0, attention to itself), so this touches no rows. Pairs: since/by -> has-had (aspectual), tomorrow/earlier
-> will-had (temporal), I/you and me/you -> myself-yourself (person), either/not -> or-but, both/neither -> and-nor (correlative); king/queen ->
he-she is reported as exploratory (8.1 was fifth in that sweep) and not scored.

PREDICTIONS (scored as written; failures preserved)
    pred_a_lookup_sign_matches_every_registered_decision   S > 0 for all six registered pairs
    pred_b_registered_pairs_beat_random_token_pairs        |S| of each registered pair > max |S| over 16 random token pairs on the same contrast
    pred_c_lookup_is_contrast_specific                     for each registered pair, |S| on every other family's contrast <= 0.25 x its own |S|
                                                           (prior: unsure -- the temporal and number contrasts are entangled at 11.3, maybe here too)
PRICE: 0 forwards on rows; one block-0 attention call per token on CPU.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random
from pathlib import Path
import aspectual_dod_lib as L

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/head_8_1_lookup_table_v133_result.json"
CANDIDATE_ID = "corpus.head_8_1_lookup_table_v133"
LAYER, HEAD, D = 8, 1, 128
PAIRS = {"aspectual since/by": (" since", " by", " has", " had"), "temporal tomorrow/earlier": (" tomorrow", " earlier", " will", " had"),
         "person I/you": (" I", " you", " myself", " yourself"), "person me/you": (" me", " you", " myself", " yourself"),
         "correlative either/not": (" either", " not", " or", " but"), "correlative both/neither": (" both", " neither", " and", " nor")}
EXPLORATORY = {"gender king/queen": (" king", " queen", " he", " she")}
CROSS_MAX, N_RANDOM = 0.25, 16
PREDICTIONS = {"pred_a_lookup_sign_matches_every_registered_decision": "> 0 x 6", "pred_b_registered_pairs_beat_random_token_pairs": "> random max", "pred_c_lookup_is_contrast_specific": "<= 0.25 x own"}


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "cpu_lane"}, indent=2)); return
    import torch, fastload
    m = fastload.load_model_fast().eval(); F = torch.nn.functional
    W = m.lm_head.weight.detach().float(); O = m.transformer.h[LAYER].attn.c_proj.weight.detach().float()[:, HEAD * D:(HEAD + 1) * D]
    block0 = m.transformer.h[0]

    def v1(text):
        tid = L._single(text)
        x = F.rms_norm(m.transformer.wte(torch.tensor([[tid]])), (m.config.n_embd,))
        with torch.no_grad():
            _, v = block0.attn(x, None)
        return v[0, 0, HEAD].float()   # (128,)

    def score(a, b, pos, neg):
        return float((W[L._single(pos)] - W[L._single(neg)]) @ (O @ (v1(a) - v1(b))))
    rng = random.Random(2026_09_18_133)
    vocab = [t for t in range(200, 20000) if 0 < len(m.transformer.wte.weight) and True]
    def random_pairs():
        out = []
        while len(out) < N_RANDOM:
            a, b = rng.sample(vocab, 2)
            out.append((a, b))
        return out
    def v1_id(tid):
        x = F.rms_norm(m.transformer.wte(torch.tensor([[tid]])), (m.config.n_embd,))
        with torch.no_grad():
            _, v = block0.attn(x, None)
        return v[0, 0, HEAD].float()
    report = {}
    for name, (a, b, pos, neg) in {**PAIRS, **EXPLORATORY}.items():
        S = score(a, b, pos, neg)
        u = W[L._single(pos)] - W[L._single(neg)]
        nulls = [float(u @ (O @ (v1_id(x) - v1_id(y)))) for x, y in random_pairs()]
        cross = {other: score(a, b, opos, oneg) for other, (_, _, opos, oneg) in PAIRS.items() if other != name and (opos, oneg) != (pos, neg)}
        report[name] = {"score": S, "random_abs_max": max(abs(v) for v in nulls), "random_abs_median": sorted(abs(v) for v in nulls)[N_RANDOM // 2], "cross": cross,
                        "cross_max_ratio": max((abs(v) for v in cross.values()), default=0.0) / max(abs(S), 1e-9)}
        print(name, "S", round(S, 3), "random max", round(report[name]["random_abs_max"], 3), "cross", {k: round(v, 3) for k, v in cross.items()})
    reg = {k: report[k] for k in PAIRS}
    predictions = {"pred_a_lookup_sign_matches_every_registered_decision": all(r["score"] > 0 for r in reg.values()),
                   "pred_b_registered_pairs_beat_random_token_pairs": all(abs(r["score"]) > r["random_abs_max"] for r in reg.values()),
                   "pred_c_lookup_is_contrast_specific": all(r["cross_max_ratio"] <= CROSS_MAX for r in reg.values())}
    OUT.write_text(json.dumps({"schema": "head_8_1_lookup_table_result_v133", "candidate_id": CANDIDATE_ID, "pairs": report, "exploratory": list(EXPLORATORY), "predictions": predictions, "forwards": 0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
