#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_rank_scales_sublinearly pred_c_classes_separable pred_d_number_difference_is_rank1 pred_e_mlp1_amplifies_embedding_number pred_f_radial_share_small
"""MLP 1 as a token table, with class structure and exponential class scaling (v287; Logan, 2026-09-19). The dossier already has: MLP 1 is 79% a context-free token
lookup table (RESULTS §250), its corpus output is one register direction plus a long tail (§§16-17), and its plural - singular write at a noun is rank-1 (tier-5 v167).
Not on the record: how the table's structure SCALES with class size and what it does per class. Single-token inputs (position 0 -- the context-free fold), seven
classes from `dod_token_classes_v287.json` (lexicon noun pairs, vocabulary s-pairs, verb present/past pairs, numbers, punctuation, function words, adjectives),
sizes 4 / 16 / 64 / 256 (first n of each list, where available). Captured at position 0: x0 = rms(embedding), attn0 / mlp0 / attn1 writes, the block-1 pre-MLP
residual x1, MLP 1's hidden h1 and its write. Analyses (exact, CPU after capture): (A) closure of x1 as the lambda-weighted sum of its four writers; (B) r90 =
directions for 90% of centred energy of the write matrix, per class and for the union, at each size; (C) between-class share of variance of the writes at each
size; (D) plural - singular (and past - present) differences: top-singular energy share and its scaling with the number of pairs; (E) MLP 1 vs the embedding on the
number contrast: median norm ratio and cosine of the mean differences; (F) radial share: the write's energy along the current residual direction.
PREDICTIONS (scored as written; failures preserved; priors unsure except a)
    pred_a_writer_closure                  x1 = lambda-recurrence of (x0, attn0, mlp0, attn1) within relative 1e-4, every token
    pred_b_rank_scales_sublinearly         union of classes: r90(256 per class) <= 2 x r90(64 per class)  (structure, not noise)
    pred_c_classes_separable               at 64 per class, between-class variance of the MLP-1 writes >= 0.30 of total
    pred_d_number_difference_is_rank1      over the 126 lexicon noun pairs, the top singular direction of the plural - singular MLP-1 differences carries >= 0.50 of energy
    pred_e_mlp1_amplifies_embedding_number median ||d mlp1|| / ||d x0|| >= 1.0 over the lexicon pairs and cos(mean d mlp1, mean d x0) >= 0.30
    pred_f_radial_share_small              mean over all tokens of (write . x1_hat)^2 / ||write||^2 <= 0.10  (MLP 1 is not a norm function on single tokens)
PRICE (registered maximum): ~1,900 single-token rows in batches of 256 = 8 forwards; 0 backwards; 0 fits. Bar <= 12.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_token_table_scaling_v287_result.json"
CANDIDATE_ID = "mlp1.token_table.scaling_v287"
SPEC = json.load(open(ROOT / "ops/dod_token_classes_v287.json"))
SIZES, BATCH, CLOSURE_TOL, RANK_RATIO, BETWEEN_MIN, RANK1_MIN, AMP_MIN, COS_MIN, RADIAL_MAX = (4, 16, 64, 256), 256, 1e-4, 2.0, 0.30, 0.50, 1.0, 0.30, 0.10
FORWARDS_MAX = 12
PREDICTIONS = {"pred_a_writer_closure": "<= 1e-4", "pred_b_rank_scales_sublinearly": "r90(256) <= 2 r90(64)", "pred_c_classes_separable": ">= 0.30", "pred_d_number_difference_is_rank1": ">= 0.50", "pred_e_mlp1_amplifies_embedding_number": ">= 1.0 and >= 0.30", "pred_f_radial_share_small": "<= 0.10"}


def classes():
    """token id lists per class (singular / present forms for the pair classes) and the pair lists."""
    c = {"nouns": [a for _, a, _ in SPEC["noun_pairs_lexicon"]], "vocab_nouns": [a for _, a, _ in SPEC["noun_pairs_vocab"]], "verbs": [a for _, a, _ in SPEC["verb_pairs"]],
         "numbers": [t for _, t in SPEC["numbers"]], "punctuation": [t for _, t in SPEC["punctuation"]], "function_words": [t for _, t in SPEC["function_words"]], "adjectives": [t for _, t in SPEC["adjectives"]]}
    pairs = {"noun_pairs_lexicon": [(a, b) for _, a, b in SPEC["noun_pairs_lexicon"]], "noun_pairs_vocab": [(a, b) for _, a, b in SPEC["noun_pairs_vocab"]], "verb_pairs": [(a, b) for _, a, b in SPEC["verb_pairs"]]}
    return c, pairs


def main() -> None:
    cls, pairs = classes()
    all_ids = sorted({t for v in cls.values() for t in v} | {t for ps in pairs.values() for p in ps for t in p})
    plan = {"candidate_id": CANDIDATE_ID, "tokens": len(all_ids), "class_sizes": {k: len(v) for k, v in cls.items()}, "pair_counts": {k: len(v) for k, v in pairs.items()}, "sizes": SIZES, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "rank_ratio": RANK_RATIO, "between_min": BETWEEN_MIN, "rank1_min": RANK1_MIN, "amp_min": AMP_MIN, "cos_min": COS_MIN, "radial_max": RADIAL_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    cap = {k: [] for k in ("x0", "attn0", "mlp0", "attn1", "x1", "h1", "mlp1")}; forwards, closure = 0, 0.0
    with torch.no_grad():
        for s in range(0, len(all_ids), BATCH):
            ids = torch.tensor(all_ids[s:s + BATCH], device="cuda").unsqueeze(1)             # [n, 1]: single-token inputs at position 0
            x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1):
                block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                cap[f"attn{l}"].append(attention[:, 0].float().cpu())
                if l == 0: cap["x0"].append(x0[:, 0].float().cpu()); m = block.mlp(xin); cap["mlp0"].append(m[:, 0].float().cpu()); x = x + m
                else:
                    cap["x1"].append(x[:, 0].float().cpu()); h = dod_units.hidden(model, block.mlp, xin); cap["h1"].append(h[:, 0].float().cpu()); cap["mlp1"].append(block.mlp(xin)[:, 0].float().cpu())
            forwards += 1
    C = {k: torch.cat(v) for k, v in cap.items()}; index = {t: i for i, t in enumerate(all_ids)}
    l00, l01, l10, l11 = (float(blocks[0].lambdas[0]), float(blocks[0].lambdas[1]), float(blocks[1].lambdas[0]), float(blocks[1].lambdas[1]))
    recon = l10 * (l00 * C["x0"] + l01 * C["x0"] + C["attn0"] + C["mlp0"]) + l11 * C["x0"] + C["attn1"]
    closure = float(((recon - C["x1"]).norm(dim=1) / C["x1"].norm(dim=1)).max())
    W = C["mlp1"]
    def r90(M):
        Mc = M - M.mean(0, keepdim=True); s_ = torch.linalg.svdvals(Mc); e = (s_ ** 2).cumsum(0) / (s_ ** 2).sum(); return int((e < 0.90).sum()) + 1
    def rows(ids): return W[[index[t] for t in ids]]
    rank = {}
    for n in SIZES:
        per = {k: r90(rows(v[:n])) for k, v in cls.items() if len(v) >= n}
        union = [t for k, v in cls.items() for t in v[:n]]; rank[str(n)] = {"per_class": per, "union": r90(rows(union)), "union_tokens": len(union)}
    between = {}
    for n in SIZES:
        groups = [rows(v[:n]) for v in cls.values() if len(v) >= n]
        allr = torch.cat(groups); mu = allr.mean(0); tot = ((allr - mu) ** 2).sum()
        btw = sum(g.shape[0] * ((g.mean(0) - mu) ** 2).sum() for g in groups); between[str(n)] = float(btw / tot)
    diffs = {}
    for name, ps in pairs.items():
        d = {}
        for n in list(SIZES) + [len(ps)]:
            if n > len(ps): continue
            sub = ps[:n]; D = torch.stack([W[index[b]] - W[index[a]] for a, b in sub]); s_ = torch.linalg.svdvals(D); top1 = float(s_[0] ** 2 / (s_ ** 2).sum())
            D0 = torch.stack([C["x0"][index[b]] - C["x0"][index[a]] for a, b in sub])
            ratio = float((D.norm(dim=1) / D0.norm(dim=1)).median()); cosm = float(torch.nn.functional.cosine_similarity(D.mean(0), D0.mean(0), dim=0))
            d[str(n)] = {"top1_energy": top1, "median_norm_ratio_mlp1_over_x0": ratio, "cos_mean_diff_mlp1_x0": cosm, "r90": r90(D) if n >= 4 else None}
        diffs[name] = d
    xhat = C["x1"] / C["x1"].norm(dim=1, keepdim=True); radial = float((((W * xhat).sum(1) ** 2) / (W.norm(dim=1) ** 2)).mean())
    lex = diffs["noun_pairs_lexicon"][str(len(pairs["noun_pairs_lexicon"]))]
    report = {"closure_max": closure, "rank": rank, "between_class_share": between, "pair_differences": diffs, "radial_share_mean": radial, "write_norm_mean": float(W.norm(dim=1).mean()), "x1_norm_mean": float(C["x1"].norm(dim=1).mean())}
    print("closure", closure, "| r90 union", {n: v["union"] for n, v in rank.items()}, "| between", {n: round(v, 3) for n, v in between.items()}, "| radial", round(radial, 4))
    print("lexicon noun pairs", {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for n, d in diffs["noun_pairs_lexicon"].items()})
    print("verb pairs", {n: round(d["top1_energy"], 3) for n, d in diffs["verb_pairs"].items()}, "vocab pairs", {n: round(d["top1_energy"], 3) for n, d in diffs["noun_pairs_vocab"].items()})
    predictions = {"pred_a_writer_closure": closure <= CLOSURE_TOL, "pred_b_rank_scales_sublinearly": rank["256"]["union"] <= RANK_RATIO * rank["64"]["union"], "pred_c_classes_separable": between["64"] >= BETWEEN_MIN,
                   "pred_d_number_difference_is_rank1": lex["top1_energy"] >= RANK1_MIN, "pred_e_mlp1_amplifies_embedding_number": lex["median_norm_ratio_mlp1_over_x0"] >= AMP_MIN and lex["cos_mean_diff_mlp1_x0"] >= COS_MIN, "pred_f_radial_share_small": radial <= RADIAL_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_token_table_scaling_result_v287", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
