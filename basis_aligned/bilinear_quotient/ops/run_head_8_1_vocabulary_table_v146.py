#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_registered_cues_rank_in_top_twenty pred_b_registered_cue_signs_match pred_c_top_tokens_are_family_cue_words
"""Head 8.1's lookup table over the WHOLE vocabulary (v146, weights + single-token block-0 values, CPU): along each family's contrast direction, which
tokens does O_{8.1} v1(token) score highest?

v133 showed the registered cue pairs have the right sign and beat random pairs; v145 that the same copy feeds a later reader. This run ranks every
vocabulary token t by S_c(t) = (u_pos - u_neg) . O_{8.1} v1(t) for c in {has-had, will-had, myself-yourself, or-but, and-nor, he-she} and asks whether the
families' own cue words sit at the top. v1(t) is the block-0 value of a one-token input. Scored on the first 32000 vocabulary ids (single tokens with a
leading space and alphabetic content; the rest are reported but not scored).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_registered_cues_rank_in_top_twenty   for each contrast, each registered cue token of its family is within the top 20 by |S| (positive
                                                cue among the top-20 positive, negative cue among the top-20 negative) among alphabetic tokens
    pred_b_registered_cue_signs_match           S(positive cue) > 0 > S(negative cue) for every contrast (v133 replay at the token level)
    pred_c_top_tokens_are_family_cue_words      for each contrast, at least 5 of the top-10 positive tokens are of the family's cue class (temporal:
                                                since / until / already / now / recently ...; person: I / me / my / we / us; correlative: either /
                                                both / neither / whether ...; gender: male / female kin and title nouns) -- judged by a fixed
                                                word list in the file, not after seeing the ranking
PRICE: 0 forwards on rows; block-0 attention on single tokens, CPU.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os
from pathlib import Path
import aspectual_dod_lib as L

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/head_8_1_vocabulary_table_v146_result.json"
CANDIDATE_ID = "corpus.head_8_1_vocabulary_table_v146"
LAYER, HEAD, D, N_VOCAB = 8, 1, 128, 32000
CONTRASTS = {"has-had": ((" has", " had"), (" since", " by")), "will-had": ((" will", " had"), (" tomorrow", " earlier")), "myself-yourself": ((" myself", " yourself"), (" I", " you")),
             "or-but": ((" or", " but"), (" either", " not")), "and-nor": ((" and", " nor"), (" both", " neither")), "he-she": ((" he", " she"), (" king", " queen"))}
CUE_CLASS = {"has-had": {"since", "until", "already", "now", "recently", "lately", "ever", "just", "yet", "still", "so", "far", "before", "previously", "once", "long", "ago", "earlier", "then", "later", "after", "by", "last", "yesterday", "formerly"},
             "will-had": {"tomorrow", "soon", "next", "later", "eventually", "someday", "shortly", "then", "afterwards", "future", "earlier", "yesterday", "previously", "before", "ago", "already", "once", "formerly", "last", "recently"},
             "myself-yourself": {"I", "me", "my", "mine", "we", "us", "our", "myself", "you", "your", "yours", "yourself", "ye", "thou", "thy", "thee"},
             "or-but": {"either", "whether", "neither", "or", "nor", "not", "but", "rather", "instead", "never", "nothing", "no", "none", "hardly"},
             "and-nor": {"both", "and", "neither", "nor", "either", "also", "too", "plus", "along", "together", "not", "no", "never", "none"},
             "he-she": {"king", "queen", "man", "woman", "father", "mother", "brother", "sister", "son", "daughter", "husband", "wife", "boy", "girl", "prince", "princess", "uncle", "aunt", "nephew", "niece", "actor", "actress", "lord", "lady", "duke", "duchess", "monk", "nun", "he", "she", "his", "her", "him", "himself", "herself", "mr", "mrs", "sir", "madam", "gentleman", "male", "female", "guy", "gal", "dad", "mom", "papa", "mama", "grandfather", "grandmother", "hero", "heroine", "god", "goddess", "waiter", "waitress", "groom", "bride", "boyfriend", "girlfriend", "widow", "widower", "emperor", "empress", "baron", "baroness", "steward", "stewardess", "host", "hostess", "chairman", "chairwoman", "spokesman", "spokeswoman", "businessman", "businesswoman", "policeman", "policewoman", "priest", "priestess", "bull", "cow", "stallion", "mare", "rooster", "hen", "lion", "lioness"}}
PREDICTIONS = {"pred_a_registered_cues_rank_in_top_twenty": "top 20", "pred_b_registered_cue_signs_match": "signs", "pred_c_top_tokens_are_family_cue_words": ">= 5 of top 10"}


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "cpu_lane"}, indent=2)); return
    import torch, fastload
    m = fastload.load_model_fast().eval(); F = torch.nn.functional
    W = m.lm_head.weight.detach().float(); O = m.transformer.h[LAYER].attn.c_proj.weight.detach().float()[:, HEAD * D:(HEAD + 1) * D]
    block0 = m.transformer.h[0]
    ids = torch.arange(N_VOCAB)
    with torch.no_grad():
        x = F.rms_norm(m.transformer.wte(ids)[:, None, :], (m.config.n_embd,))     # (N, 1, 1152): each token alone at position 0
        vals = []
        for start in range(0, N_VOCAB, 2048):
            _, v = block0.attn(x[start:start + 2048], None); vals.append(v[:, 0, HEAD].float())
        V1 = torch.cat(vals)                                                        # (N, 128)
        writes = V1 @ O.T                                                           # (N, 1152)
    texts = [L.ENCODING.decode([i]) for i in range(N_VOCAB)]
    alpha = [i for i, t in enumerate(texts) if t.startswith(" ") and t[1:].isalpha()]
    report = {}
    for name, ((pos, neg), (cue_pos, cue_neg)) in CONTRASTS.items():
        u = W[L._single(pos)] - W[L._single(neg)]
        S = writes @ u
        Sa = S[alpha]; order_pos = [alpha[i] for i in torch.argsort(Sa, descending=True)[:20].tolist()]; order_neg = [alpha[i] for i in torch.argsort(Sa)[:20].tolist()]
        cp, cn = L._single(cue_pos), L._single(cue_neg)
        top_pos_words = [texts[i].strip() for i in order_pos]; top_neg_words = [texts[i].strip() for i in order_neg]
        cls = CUE_CLASS[name]
        report[name] = {"cue_pos": cue_pos, "cue_neg": cue_neg, "S_cue_pos": float(S[cp]), "S_cue_neg": float(S[cn]), "cue_pos_rank": int((Sa > S[cp]).sum()) + 1, "cue_neg_rank": int((Sa < S[cn]).sum()) + 1,
                        "top20_positive": top_pos_words, "top20_negative": top_neg_words, "cue_class_hits_top10_positive": sum(w.lower() in cls for w in top_pos_words[:10]),
                        "cue_class_hits_top10_negative": sum(w.lower() in cls for w in top_neg_words[:10])}
        print(name, "cue ranks", report[name]["cue_pos_rank"], report[name]["cue_neg_rank"], "top+", top_pos_words[:10], "top-", top_neg_words[:10])
    predictions = {"pred_a_registered_cues_rank_in_top_twenty": all(r["cue_pos_rank"] <= 20 and r["cue_neg_rank"] <= 20 for r in report.values()),
                   "pred_b_registered_cue_signs_match": all(r["S_cue_pos"] > 0 > r["S_cue_neg"] for r in report.values()),
                   "pred_c_top_tokens_are_family_cue_words": all(r["cue_class_hits_top10_positive"] >= 5 for r in report.values())}
    OUT.write_text(json.dumps({"schema": "head_8_1_vocabulary_table_result_v146", "candidate_id": CANDIDATE_ID, "n_vocab_scored": len(alpha), "contrasts": report, "predictions": predictions, "forwards": 0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
