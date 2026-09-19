#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_shared_direction_fixed_across_lengths pred_c_shared_direction_fixed_across_phrases pred_d_shared_direction_is_the_corpus_top_direction pred_e_shared_direction_orthogonal_to_table
"""MLP 1: the shared in-context direction (v294). v293: ~40% of the non-table remainder of MLP 1's write is ONE direction shared across 224 targets
(grand mean), orthogonal to both the token's and the context's table entries. Is it one fixed direction of MLP 1 -- the dossier's "one register
direction" that dominates its corpus output (RESULTS sections 16-17) -- or does it depend on the context? Three context phrases (A ", and of the very"
cyclic; B " I said that we"; C " 1 2 3 4 5") x lengths 1 / 8 / 64 x 7 classes x 32 targets: u(phrase, k) = unit grand-mean remainder. Corpus
reference: MLP 1's writes at every position >= 1 of the 128 natural rows (v272 FineWeb + v273 Pile; 24 tokens each; 2,944 writes), top right
singular vector of the uncentred write matrix (PC1) and the unit mean write.
PREDICTIONS (scored as written; failures preserved; priors from v293 and the dossier)
    pred_a_pair_closure                          Down[cross + context-only] = W - T within relative 1e-3 on every row
    pred_b_shared_direction_fixed_across_lengths  within each phrase, min pairwise |cos| of u over lengths 1 / 8 / 64 >= 0.80
    pred_c_shared_direction_fixed_across_phrases  at length 8, min pairwise |cos| of u across the three phrases >= 0.70. Prior: unsure.
    pred_d_shared_direction_is_the_corpus_top_direction  |cos(u(phrase, 8), corpus PC1)| >= 0.70 for all three phrases
    pred_e_shared_direction_orthogonal_to_table  median |cos(u(phrase, 8), table entry)| over the 224 targets <= 0.15 for all three phrases
PRICE (registered maximum): 9 context batches + 1 table batch + 2 natural batches = 12 forwards; 0 backwards; 0 fits (an SVD of captured writes is a summary, not a fit). Bar <= 14.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_shared_direction_v294_result.json"
CANDIDATE_ID = "mlp1.token_table.shared_direction_v294"
PHRASES = {"A": (",", " and", " of", " the", " very"), "B": (" I", " said", " that", " we"), "C": (" 1", " 2", " 3", " 4", " 5")}
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, LEN_MIN, PHRASE_MIN, PC_MIN, ORTH_MAX = 1e-3, 0.80, 0.70, 0.70, 0.15
FORWARDS_MAX = 14
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_shared_direction_fixed_across_lengths": ">= 0.80 x 3", "pred_c_shared_direction_fixed_across_phrases": ">= 0.70", "pred_d_shared_direction_is_the_corpus_top_direction": ">= 0.70 x 3", "pred_e_shared_direction_orthogonal_to_table": "<= 0.15 x 3"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fills = {n: [L._single(t) for t in ph] for n, ph in PHRASES.items()}; filler = lambda n, k: [fills[n][i % len(fills[n])] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrases": {n: list(p) for n, p in PHRASES.items()}, "natural": [p.name for p in NATURAL], "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "len_min": LEN_MIN, "phrase_min": PHRASE_MIN, "pc_min": PC_MIN, "orth_max": ORTH_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    # corpus reference: MLP 1's writes at every position >= 1 of the natural rows
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]; nat = torch.tensor([r["ids"] for r in recs], device="cuda"); corpus = []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,))); x = x + m
            corpus.append(m[:, 1:].reshape(-1, m.shape[-1]).float().cpu()); forwards += 1
    corpus = torch.cat(corpus); pc1 = torch.linalg.svd(corpus, full_matrices=False).Vh[0]; mean_dir = corpus.mean(0) / corpus.mean(0).norm()
    unit = lambda v: v / v.norm(); closure, u, per = 0.0, {}, {}
    for name in PHRASES:
        for k in LENGTHS:
            toks = torch.tensor([filler(name, k) + [t] for t in targets], device="cuda"); c = v289.capture(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
            W = c["mlp1"]; cc = F.rms_norm(c["x1"], (T.shape[-1],)) - n_tab; Lc, Rc = cc @ Lw.T, cc @ Rw.T
            cross = (Lt * Rc + Lc * Rt) @ Dw.T; only = (Lc * Rc) @ Dw.T; change = W - T
            closure = max(closure, float((((cross + only) - change).norm(dim=1) / change.norm(dim=1).clamp_min(1e-6)).max()))
            alpha = (W * T).sum(1) / (T * T).sum(1); R = W - alpha[:, None] * T; g = R.mean(0); u[(name, k)] = unit(g)
            per[f"{name}|{k}"] = {"alpha_median": float(alpha.median()), "grand_mean_share": float((g.norm() ** 2 * len(R)) / (R * R).sum()), "cos_pc1": float(abs(u[(name, k)] @ pc1)), "cos_corpus_mean": float(abs(u[(name, k)] @ mean_dir)),
                                  "median_abs_cos_table": float(((T @ u[(name, k)]).abs() / T.norm(dim=1)).median()), "cos_x0_mean": float(abs(u[(name, k)] @ unit(tab["x0"].mean(0))))}
    across_len = {name: min(float(abs(u[(name, a_)] @ u[(name, b_)])) for a_ in LENGTHS for b_ in LENGTHS if a_ < b_) for name in PHRASES}
    names = list(PHRASES); across_phr = min(float(abs(u[(a_, 8)] @ u[(b_, 8)])) for a_ in names for b_ in names if a_ < b_)
    report = {"closure_max": closure, "corpus_writes": int(corpus.shape[0]), "corpus_pc1_energy_share": float((torch.linalg.svdvals(corpus)[0] ** 2) / (corpus ** 2).sum()), "corpus_mean_vs_pc1": float(abs(mean_dir @ pc1)),
              "min_cos_across_lengths": across_len, "min_cos_across_phrases_at_8": across_phr, "per": per}
    print(json.dumps(report, indent=1, default=float))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_shared_direction_fixed_across_lengths": all(v >= LEN_MIN for v in across_len.values()), "pred_c_shared_direction_fixed_across_phrases": across_phr >= PHRASE_MIN,
                   "pred_d_shared_direction_is_the_corpus_top_direction": all(per[f"{n}|8"]["cos_pc1"] >= PC_MIN for n in PHRASES), "pred_e_shared_direction_orthogonal_to_table": all(per[f"{n}|8"]["median_abs_cos_table"] <= ORTH_MAX for n in PHRASES)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_shared_direction_result_v294", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
