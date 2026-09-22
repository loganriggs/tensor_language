"""Run E0-E4 on PR-penalised DCT factors for an Elriggs checkpoint, using this repository's own model loader and a reviewed
port of AJ's fitting loop (circuit_checks.dct_fit) instead of importing the cloned repo. Everything downstream of the fit is
the handoff's circuit_checks code, unchanged.

  python scripts/run_checks.py --arch bilinear --device cuda --train-contexts 4 --heldout-contexts 8 --factors 4 \
      --iterations 5 --penalty-weights 0 1 10 100 --seeds 0 --out results/e0_ajscale.json          # AJ's scale
  python scripts/run_checks.py --arch bilinear --texts-source fineweb --out results/fineweb.json      # off-distribution prompts

Output JSON: per (seed, penalty weight): fit traces (objective / normalised PR per iteration), per factor: held-out energy,
PR (AJ's range), E1 completeness by pathway group, E2 alignment + neuron read-off baseline, E3 finite ablations; E4 cross-seed
and cross-split matching with a random-dictionary null; E2 null.
"""
import argparse, csv, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec, participation_ratio
from circuit_checks.checks import (all_units, random_units, top_units, hidden_response, output_score, pathway_completeness,
                                   effective_unit_directions, factor_alignment_to_units, neuron_readoff_candidates,
                                   ablation_effect, match_factors, random_match_null, jaccard, factor_similarity,
                                   span_alignment, match_factors_span, random_match_null_span)
from circuit_checks.tensorgpt import TensorGPTSpans, read_weights
from circuit_checks.inrepo_model import load, collect_states, ROOT
from circuit_checks.dct_fit import SpanDCT, evaluate

ADVBENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aj-repo", "harmful_behaviors.csv")


def advbench_texts(train, held):
    """AJ's read_texts: unique `target` strings, shuffled with random.Random(1729)."""
    texts = list(dict.fromkeys(r["target"].strip() for r in csv.DictReader(open(ADVBENCH))))
    random.Random(1729).shuffle(texts)
    assert train + held <= len(texts)
    return texts[:train], texts[train:train + held]


def fineweb_texts(train, held, seq_len):
    """Off-distribution prompts: the first `seq_len` GPT-2 tokens of FineWeb documents (row cache n192_skip7000), decoded."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu")
    texts = [enc.decode(r[:seq_len].tolist()) for r in rows[:train + held]]
    return texts[:train], texts[train:train + held]


def ctx_at(state, i):
    return tuple(s[i:i + 1] for s in state)


def subset(state, idx):
    return tuple(s[idx] for s in state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="bilinear")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--target-layer", type=int, default=12)
    ap.add_argument("--factors", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--factor-batch", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--sequence-length", type=int, default=32)
    ap.add_argument("--target-positions", type=int, default=3)
    ap.add_argument("--train-contexts", type=int, default=32)
    ap.add_argument("--heldout-contexts", type=int, default=64)
    ap.add_argument("--eval-contexts", type=int, default=16, help="held-out contexts used for E1-E3")
    ap.add_argument("--texts-source", default="advbench", choices=["advbench", "fineweb"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--penalty-weights", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--random-k", type=int, default=5)
    ap.add_argument("--ablation-scales", type=float, nargs="+", default=[0.05, 0.2])
    ap.add_argument("--skip-readoff", action="store_true")
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--out", default="results/pr_dct_checks.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); t_start = time.time()

    from transformers import GPT2Tokenizer
    model, cfg, meta = load(a.arch, a.device)
    tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    if a.texts_source == "advbench":
        splits = advbench_texts(a.train_contexts, a.heldout_contexts)
    else:
        splits = fineweb_texts(a.train_contexts, a.heldout_contexts, a.sequence_length)
    train, held = (collect_states(model, tok, s, a.source_layer, a.sequence_length, a.device) for s in splits)

    spans = TensorGPTSpans(model, a.source_layer, a.target_layer, slice(-a.target_positions, None))
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):                       # adapter parity against the repository's Block code
        v, x0, v1 = ctx_at(held, 0)
        for block in model.transformer.h[a.source_layer:a.target_layer]:
            v, v1 = block(v, v1, x0)
        ref = v[:, -a.target_positions:].mean(1)[0]; ours = spans.make_span(ctx_at(held, 0))(torch.zeros(cfg.n_embd, device=a.device))[0]
        parity = float((ref - ours).abs().max() / ref.abs().max()); assert parity < 1e-4, parity
    print(f"adapter parity vs repository blocks: {parity:.2e}", flush=True)
    widths = {L: spans.mlp_width(L) for L in spans.all_layers}
    aj_layers = spans.aj_pr_layers
    d = cfg.n_embd
    resid_norm = float(train[0].norm(dim=-1).mean())
    eval_ctxs = [ctx_at(held, i) for i in range(min(a.eval_contexts, len(held[0])))]
    held_ctxs = [ctx_at(held, i) for i in range(len(held[0]))]
    feature_dim = sum(widths[L] for L in aj_layers)

    def contexts_of(state):
        return [ctx_at(state, i) for i in range(len(state[0]))]

    def fit(state, seed, weight, scale=None):
        dic = SpanDCT(a.factors, weight, 1.0 if scale is None else scale)
        with sdpa_kernel(SDPBackend.MATH):
            dic.fit(spans.make_span, contexts_of(state), d, d, aj_layers, max_iters=a.iterations, factor_batch=a.factor_batch,
                    beta=a.beta, seed=seed, device=a.device, feature_dim=feature_dim)
        return dic

    def penalty_scale(dic, state):
        with sdpa_kernel(SDPBackend.MATH):
            return evaluate(dic, spans.make_span, contexts_of(state), aj_layers)["mean_total_energy"] / a.factors

    # ---- E2 prep: effective read directions of every unit per layer (bilinear only)
    eff = {}
    if not a.skip_readoff and read_weights(model.transformer.h[a.source_layer].mlp) is not None:
        with sdpa_kernel(SDPBackend.MATH):
            for L in spans.all_layers:
                WL, WR = read_weights(model.transformer.h[L].mlp)
                eff[L] = effective_unit_directions(spans.mlp_input_fn(eval_ctxs[0], L), WL, WR, range(WL.shape[0]), d)

    def evaluate_factor(u, l, r):
        g = torch.Generator().manual_seed(0); rec = {}
        with sdpa_kernel(SDPBackend.MATH):
            en, prs = [], []
            for c in eval_ctxs:
                sp = spans.make_span(c)
                en.append(output_score(sp, u, l, r).item() ** 2)
                resp = hidden_response(sp, l, r, aj_layers)
                prs.append(participation_ratio(torch.cat([resp[k] for k in aj_layers])).item())
            rec["heldout_energy"] = float(np.mean(en)); rec["heldout_pr"] = float(np.median(prs)); rec["heldout_pr_per_context"] = prs
            rec["cos_l_r"] = float((torch.nn.functional.normalize(l, dim=0) @ torch.nn.functional.normalize(r, dim=0)).abs())
            groups = {f"top{k}_aj_range": (lambda sp, k=k: top_units(hidden_response(sp, l, r, aj_layers), k)) for k in a.topk}
            groups[f"random{a.random_k}_aj_range"] = lambda sp: random_units({L: widths[L] for L in aj_layers}, a.random_k, g,
                                                                             exclude=top_units(hidden_response(sp, l, r, aj_layers), a.random_k))
            groups["all_aj_range_mlps"] = lambda sp: all_units(widths, aj_layers)
            groups["source_block_mlp"] = lambda sp: all_units(widths, [a.source_layer])
            groups["all_attention"] = lambda sp: FreezeSpec(attn_layers=set(spans.all_layers))
            groups["all_mlps_and_attention"] = lambda sp: FreezeSpec(mlp_masks=all_units(widths, spans.all_layers).mlp_masks, attn_layers=set(spans.all_layers))
            comp = pathway_completeness(spans.make_span, eval_ctxs, u, l, r, groups)
            rec["E1_completeness"] = {k: v["mean"] for k, v in comp.items() if not k.startswith("_")}
            rec["E1_completeness_median"] = {k: v["median"] for k, v in comp.items() if not k.startswith("_")}
            rec["E1_full_score_mean_abs"] = comp["_full_score"]["mean_abs"]
            if eff:
                rec["E2_alignment"] = {}
                for L, (lefts, rights) in eff.items():
                    idx, score = factor_alignment_to_units(l, r, lefts, rights)
                    sidx, sscore, _ = span_alignment(l, r, lefts, rights)
                    rec["E2_alignment"][L] = {"unit": idx, "score": score, "span_unit": sidx, "span_score": sscore}
                sp0 = spans.make_span(eval_ctxs[0]); resp0 = hidden_response(sp0, l, r, aj_layers)
                flat = torch.cat([resp0[k].abs() for k in aj_layers]); j = int(flat.argmax()); off = 0
                for L in aj_layers:
                    if j < off + widths[L]:
                        top_L, top_h = L, j - off; break
                    off += widths[L]
                lefts, rights = eff[top_L]
                cand = neuron_readoff_candidates(sp0, lefts[top_h:top_h + 1], rights[top_h:top_h + 1], [top_h], top_L, aj_layers)[0]
                fac_energy = output_score(sp0, u, l, r).item() ** 2
                rec["E2_readoff_top_unit"] = {"layer": top_L, "unit": top_h, "energy": cand["energy"], "pr": cand["pr"],
                                              "factor_energy_same_ctx": fac_energy, "energy_ratio": cand["energy"] / max(fac_energy, 1e-30)}
            rec["E3_ablation"] = {}
            for s in a.ablation_scales:
                al = s * resid_norm
                for k in a.topk:
                    rec["E3_ablation"][f"top{k}@{s}"] = ablation_effect(spans.make_span, eval_ctxs, u, l, r, al, al,
                                                                        lambda sp, k=k: top_units(hidden_response(sp, l, r, aj_layers), k))["mean"]
                rec["E3_ablation"][f"random{a.random_k}@{s}"] = ablation_effect(spans.make_span, eval_ctxs, u, l, r, al, al,
                                                                                lambda sp: random_units({L: widths[L] for L in aj_layers}, a.random_k, g))["mean"]
        return rec

    results = {"args": vars(a), "model": meta, "feature_dim": feature_dim, "resid_norm": resid_norm, "fits": [], "stability": {}, "E2_null": {}}
    if eff:
        g0 = torch.Generator().manual_seed(123)
        for L, (lefts, rights) in eff.items():
            sc = [factor_alignment_to_units(torch.randn(d, generator=g0).to(lefts.device), torch.randn(d, generator=g0).to(lefts.device), lefts, rights)[1] for _ in range(20)]
            g1 = torch.Generator().manual_seed(124)
            ss = [span_alignment(torch.randn(d, generator=g1).to(lefts.device), torch.randn(d, generator=g1).to(lefts.device), lefts, rights)[1] for _ in range(20)]
            results["E2_null"][L] = {"mean": float(np.mean(sc)), "max": float(np.max(sc)), "span_mean": float(np.mean(ss)), "span_max": float(np.max(ss))}
    dicts = {}
    for seed in a.seeds:
        t0 = time.time(); base = fit(train, seed, 0.0); scale = penalty_scale(base, train)
        for w in a.penalty_weights:
            dic = base if w == 0 else fit(train, seed, w, scale); dicts[(seed, w)] = dic
            with sdpa_kernel(SDPBackend.MATH):
                ev_train = evaluate(dic, spans.make_span, contexts_of(train), aj_layers)
                ev_held = evaluate(dic, spans.make_span, held_ctxs, aj_layers)
            factors = [evaluate_factor(dic.U[:, f], dic.L[:, f], dic.R[:, f]) for f in range(a.factors)]
            results["fits"].append({"seed": seed, "penalty_weight": w, "penalty_scale": scale, "trace_objective": dic.objective_values,
                                    "trace_energy": dic.score_energy_values, "trace_normalized_pr": dic.normalized_pr_values,
                                    "train": ev_train, "heldout": ev_held, "factors": factors, "seconds": time.time() - t0})
            json.dump(results, open(a.out, "w"), indent=1, default=float)
            print(f"seed={seed} w={w}: train energy {ev_train['mean_total_energy']:.4g} heldout {ev_held['mean_total_energy']:.4g} | median heldout PR "
                  f"{ev_held['median_participation_ratio']:.1f} | median top1 completeness {np.median([x['E1_completeness'][f'top{a.topk[0]}_aj_range'] for x in factors]):.3f}"
                  f" | source-MLP {np.median([x['E1_completeness']['source_block_mlp'] for x in factors]):.3f} attention {np.median([x['E1_completeness']['all_attention'] for x in factors]):.3f}"
                  f" | {time.time() - t0:.0f}s", flush=True)

    null = random_match_null(d, d, a.factors, n_draws=10); results["stability"]["random_null_mean"] = float(null.mean()); results["stability"]["random_null_max"] = float(null.max())
    snull = random_match_null_span(d, d, a.factors, n_draws=10); results["stability"]["span_random_null_mean"] = float(snull.mean()); results["stability"]["span_random_null_max"] = float(snull.max())
    trip = lambda dic: (dic.U, dic.L, dic.R)
    for w in a.penalty_weights:
        results["stability"][f"seeds_w{w}"] = [match_factors(trip(dicts[(s1, w)]), trip(dicts[(s2, w)])).tolist() for i, s1 in enumerate(a.seeds) for s2 in a.seeds[i + 1:]]
        results["stability"][f"seeds_w{w}_span"] = [match_factors_span(trip(dicts[(s1, w)]), trip(dicts[(s2, w)])).tolist() for i, s1 in enumerate(a.seeds) for s2 in a.seeds[i + 1:]]
    n = len(train[0]); half = n // 2
    if half >= 1 and not a.skip_split:
        A = subset(train, slice(0, half)); B = subset(train, slice(half, 2 * half))
        for w in a.penalty_weights:
            sA = penalty_scale(fit(A, 0, 0.0), A) if w else None; sB = penalty_scale(fit(B, 0, 0.0), B) if w else None
            dA, dB = fit(A, 0, w, sA), fit(B, 0, w, sB)
            results["stability"][f"split_w{w}"] = match_factors(trip(dA), trip(dB)).tolist(); results["stability"][f"split_w{w}_span"] = match_factors_span(trip(dA), trip(dB)).tolist()
            with sdpa_kernel(SDPBackend.MATH):
                sp0 = spans.make_span(eval_ctxs[0])
                def units_of(dic, f):
                    spec = top_units(hidden_response(sp0, dic.L[:, f], dic.R[:, f], aj_layers), 5)
                    return {(L, int(i)) for L, m in spec.mlp_masks.items() for i in m.nonzero().flatten()}
                from scipy.optimize import linear_sum_assignment
                S = factor_similarity(trip(dA), trip(dB)).cpu().numpy(); rows, cols = linear_sum_assignment(-S)
                results["stability"][f"split_w{w}_top5_jaccard"] = [jaccard(units_of(dA, i), units_of(dB, j)) for i, j in zip(rows, cols)]
    results["seconds"] = time.time() - t_start
    json.dump(results, open(a.out, "w"), indent=1, default=float)
    print("stability:", {k: (round(float(np.mean(v)), 3) if isinstance(v, list) and v and not isinstance(v[0], list) else v if not isinstance(v, list) else [round(float(np.mean(x)), 3) for x in v]) for k, v in results["stability"].items()})
    print("wrote", a.out, f"({results['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
