"""Run E1-E4 on AJ's PR-penalized DCT factors for a TensorGPT checkpoint.

Reuses AJ's own fitting code (dct.AsymmetricQuadraticDCT and
sparse-asymmetric/pr_penalty_experiment.ParticipationRegularizedDCT) so that we
are testing *his* factors, then evaluates them with circuit_checks.

  # smoke test, CPU, tiny random model (checks the pipeline runs end to end)
  python scripts/run_tensorgpt_checks.py --repo /path/to/redesigned-octo-couscous --tiny

  # real run
  python scripts/run_tensorgpt_checks.py --repo ... --arch bilinear --device cuda \
      --train-contexts 32 --heldout-contexts 64 --seeds 0 1 2 --penalty-weights 0 0.1 1

Outputs one JSON with, per (seed, penalty weight, factor): heldout energy, PR,
E1 completeness by pathway group, E2 neuron alignment + readoff baseline,
E3 finite ablations; and E4 cross-seed / cross-split matching.
"""
import argparse, importlib.util, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec, participation_ratio
from circuit_checks.checks import (
    all_units, random_units, top_units, hidden_response, output_score,
    pathway_completeness, effective_unit_directions, factor_alignment_to_units,
    neuron_readoff_candidates, ablation_effect, match_factors, random_match_null, jaccard,
)
from circuit_checks.tensorgpt import TensorGPTSpans, read_weights


# --------------------------------------------------------------- loading AJ's code

def load_aj_modules(repo, stub_transformers=False):
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "sparse-asymmetric"))
    if stub_transformers:
        try:
            import transformers  # noqa: F401
        except ImportError:
            sys.modules["transformers"] = types.SimpleNamespace(GPT2Tokenizer=None)
    import tensor_model, dct
    import intermediate_mlp_pr_experiment as imp
    # pr_penalty_experiment.py has `from ..tensor_model import load_tensor_gpt`,
    # which fails when imported as a script module. Load it with that line fixed
    # rather than editing AJ's file.
    path = os.path.join(repo, "sparse-asymmetric", "pr_penalty_experiment.py")
    src = open(path).read().replace("from ..tensor_model import", "from tensor_model import")
    mod = types.ModuleType("pr_penalty_experiment")
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return tensor_model, dct, imp, mod


def ctx_at(state, i):
    return tuple(s[i:i + 1] for s in state)


def subset(state, idx):
    return tuple(s[idx] for s in state)


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tiny", action="store_true", help="tiny random model + random tokens (smoke test)")
    ap.add_argument("--arch", default="bilinear")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--target-layer", type=int, default=12)
    ap.add_argument("--factors", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--factor-batch", type=int, default=4)
    ap.add_argument("--sequence-length", type=int, default=32)
    ap.add_argument("--target-positions", type=int, default=3)
    ap.add_argument("--train-contexts", type=int, default=32)
    ap.add_argument("--heldout-contexts", type=int, default=64)
    ap.add_argument("--eval-contexts", type=int, default=16, help="held-out contexts used for E1-E3")
    ap.add_argument("--texts-file", default=None, help="one prompt per line; default: AJ's harmful_behaviors.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--penalty-weights", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--random-k", type=int, default=5)
    ap.add_argument("--ablation-scales", type=float, nargs="+", default=[0.05, 0.2],
                    help="finite perturbation size as a fraction of mean residual norm at source")
    ap.add_argument("--skip-readoff", action="store_true")
    ap.add_argument("--out", default="pr_dct_checks.json")
    a = ap.parse_args()

    tm, dct, imp, prp = load_aj_modules(a.repo, stub_transformers=a.tiny)
    torch.manual_seed(0)

    # ---- model + contexts
    if a.tiny:
        a.device, a.source_layer, a.target_layer = "cpu", 1, 4
        a.train_contexts, a.heldout_contexts, a.eval_contexts = 4, 6, 3
        a.factors, a.iterations, a.sequence_length = 3, 2, 12
        a.topk, a.random_k, a.seeds, a.penalty_weights = [1, 2], 2, [0, 1], [0.0, 0.1]
        cfg = tm.TensorGPTConfig(vocab_size=100, n_layer=5, n_head=2, n_embd=32, bilinear=True)
        model = tm.TensorGPT(cfg)
        for p in model.parameters():
            if p.dim() == 2:
                torch.nn.init.normal_(p, std=0.2)
        model = model.eval()

        def states_for(n, seed):
            g = torch.Generator().manual_seed(seed)
            ids = torch.randint(0, 100, (n, a.sequence_length), generator=g)
            out = [model.state_before_block(ids[i:i + 1], a.source_layer) for i in range(n)]
            return tuple(torch.cat([o[k].float() for o in out]) for k in ("values", "initial_values", "first_values"))
        train, held = states_for(a.train_contexts, 0), states_for(a.heldout_contexts, 1)
    else:
        from transformers import GPT2Tokenizer
        model, cfg, _ = tm.load_tensor_gpt(imp.REPOSITORIES[a.arch], a.device)
        tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        if a.texts_file:
            texts = [t.strip() for t in open(a.texts_file) if t.strip()]
            rng = np.random.default_rng(1729); rng.shuffle(texts)
            splits = texts[:a.train_contexts], texts[a.train_contexts:a.train_contexts + a.heldout_contexts]
        else:
            splits = imp.read_texts(a.train_contexts, a.heldout_contexts)
        train, held = (imp.collect_middle_inputs(model, tok, s, a.source_layer, a.sequence_length) for s in splits)
    model.requires_grad_(False)

    # prp.make_operators reads these attribute names
    op_args = types.SimpleNamespace(source_layer=a.source_layer, target_layer=a.target_layer,
                                    target_positions=a.target_positions)
    spans = TensorGPTSpans(model, a.source_layer, a.target_layer, slice(-a.target_positions, None))
    widths = {L: spans.mlp_width(L) for L in spans.all_layers}
    aj_layers = spans.aj_pr_layers
    d = cfg.n_embd
    resid_norm = float(train[0].detach().norm(dim=-1).mean())
    eval_ctxs = [ctx_at(held, i) for i in range(min(a.eval_contexts, len(held[0])))]

    def fit(state, seed, weight, scale=None):
        clean, delta, feats = prp.make_operators(model, state, op_args)
        torch.manual_seed(seed)
        with sdpa_kernel(SDPBackend.MATH):
            if weight == 0:
                dic = dct.AsymmetricQuadraticDCT(num_factors=a.factors)
                dic.fit(delta, state[0], clean, batch_size=1, factor_batch_size=a.factor_batch, max_iters=a.iterations)
            else:
                dic = prp.ParticipationRegularizedDCT(a.factors, weight, scale)
                dic.fit(delta, feats, state[0], clean, batch_size=1, factor_batch_size=a.factor_batch, max_iters=a.iterations)
        return dic

    def penalty_scale(dic, state):
        clean, delta, feats = prp.make_operators(model, state, op_args)
        with sdpa_kernel(SDPBackend.MATH):
            return prp.evaluate(dic, delta, feats, state[0], clean)["mean_total_energy"] / a.factors

    # ---- readoff prep: effective read directions of every unit, per layer (bilinear only)
    eff = {}
    if not a.skip_readoff and read_weights(model.transformer.h[a.source_layer].mlp) is not None:
        with sdpa_kernel(SDPBackend.MATH):
            for L in spans.all_layers:
                WL, WR = read_weights(model.transformer.h[L].mlp)
                eff[L] = effective_unit_directions(spans.mlp_input_fn(eval_ctxs[0], L), WL, WR,
                                                   range(WL.shape[0]), d)

    def evaluate_factor(u, l, r):
        g = torch.Generator().manual_seed(0)
        rec = {}
        with sdpa_kernel(SDPBackend.MATH):
            # heldout energy + PR (AJ's definition and range)
            en, prs = [], []
            for c in eval_ctxs:
                sp = spans.make_span(c)
                en.append(output_score(sp, u, l, r).item() ** 2)
                resp = hidden_response(sp, l, r, aj_layers)
                prs.append(participation_ratio(torch.cat([resp[k] for k in aj_layers])).item())
            rec["heldout_energy"] = float(np.mean(en))
            rec["heldout_pr"] = float(np.median(prs))

            # E1: completeness by pathway
            groups = {f"top{k}_aj_range": (lambda sp, k=k: top_units(hidden_response(sp, l, r, aj_layers), k))
                      for k in a.topk}
            groups[f"random{a.random_k}_aj_range"] = lambda sp: random_units(
                {L: widths[L] for L in aj_layers}, a.random_k, g,
                exclude=top_units(hidden_response(sp, l, r, aj_layers), a.random_k))
            groups["all_aj_range_mlps"] = lambda sp: all_units(widths, aj_layers)
            groups["source_block_mlp"] = lambda sp: all_units(widths, [a.source_layer])
            groups["all_attention"] = lambda sp: FreezeSpec(attn_layers=set(spans.all_layers))
            rec["E1_completeness"] = {k: v["mean"] for k, v in pathway_completeness(
                spans.make_span, eval_ctxs, u, l, r, groups).items() if not k.startswith("_")}

            # E2: does the factor just point at one neuron?
            if eff:
                rec["E2_alignment"] = {}
                for L, (lefts, rights) in eff.items():
                    idx, score = factor_alignment_to_units(l, r, lefts, rights)
                    rec["E2_alignment"][L] = {"unit": idx, "score": score}
                sp0 = spans.make_span(eval_ctxs[0])
                resp0 = hidden_response(sp0, l, r, aj_layers)
                flat = torch.cat([resp0[k].abs() for k in aj_layers])
                j = int(flat.argmax()); off = 0
                for L in aj_layers:
                    if j < off + widths[L]:
                        top_L, top_h = L, j - off; break
                    off += widths[L]
                lefts, rights = eff[top_L]
                cand = neuron_readoff_candidates(sp0, lefts[top_h:top_h + 1], rights[top_h:top_h + 1],
                                                 [top_h], top_L, aj_layers)[0]
                fac_energy = output_score(sp0, u, l, r).item() ** 2
                rec["E2_readoff_top_unit"] = {"layer": top_L, "unit": top_h, "energy": cand["energy"],
                                              "pr": cand["pr"], "factor_energy_same_ctx": fac_energy,
                                              "energy_ratio": cand["energy"] / max(fac_energy, 1e-30)}

            # E3: finite ablations on held-out contexts
            rec["E3_ablation"] = {}
            for s in a.ablation_scales:
                al = s * resid_norm
                for k in a.topk:
                    rec["E3_ablation"][f"top{k}@{s}"] = ablation_effect(
                        spans.make_span, eval_ctxs, u, l, r, al, al,
                        lambda sp, k=k: top_units(hidden_response(sp, l, r, aj_layers), k))["mean"]
                rec["E3_ablation"][f"random{a.random_k}@{s}"] = ablation_effect(
                    spans.make_span, eval_ctxs, u, l, r, al, al,
                    lambda sp: random_units({L: widths[L] for L in aj_layers}, a.random_k, g))["mean"]
        return rec

    results = {"args": vars(a), "fits": [], "stability": {}, "E2_null": {}}
    if eff:  # alignment a random (l, r) pair gets, for calibrating E2 scores
        g0 = torch.Generator().manual_seed(123)
        for L, (lefts, rights) in eff.items():
            sc = [factor_alignment_to_units(torch.randn(d, generator=g0).to(lefts.device),
                                            torch.randn(d, generator=g0).to(lefts.device), lefts, rights)[1]
                  for _ in range(20)]
            results["E2_null"][L] = {"mean": float(np.mean(sc)), "max": float(np.max(sc))}
    dicts = {}
    for seed in a.seeds:
        base = fit(train, seed, 0.0)
        scale = penalty_scale(base, train)
        for w in a.penalty_weights:
            dic = base if w == 0 else fit(train, seed, w, scale)
            dicts[(seed, w)] = dic
            factors = []
            for f in range(a.factors):
                u, l, r = dic.U[:, f], dic.L[:, f], dic.R[:, f]
                factors.append(evaluate_factor(u, l, r))
            results["fits"].append({"seed": seed, "penalty_weight": w, "penalty_scale": scale, "factors": factors})
            with open(a.out, "w") as fh:
                json.dump(results, fh, indent=1, default=float)
            print(f"seed={seed} w={w}: median heldout PR "
                  f"{np.median([x['heldout_pr'] for x in factors]):.1f}, "
                  f"median top1 completeness {np.median([x['E1_completeness'][f'top{a.topk[0]}_aj_range'] for x in factors]):.3f}")

    # ---- E4 stability: across seeds, and across disjoint train halves
    null = random_match_null(d, d, a.factors, n_draws=10)
    results["stability"]["random_null_mean"] = float(null.mean())
    trip = lambda dic: (dic.U, dic.L, dic.R)
    for w in a.penalty_weights:
        sims = [match_factors(trip(dicts[(s1, w)]), trip(dicts[(s2, w)])).tolist()
                for i, s1 in enumerate(a.seeds) for s2 in a.seeds[i + 1:]]
        results["stability"][f"seeds_w{w}"] = sims
    n = len(train[0]); half = n // 2
    if half >= 1:
        A = subset(train, slice(0, half)); B = subset(train, slice(half, 2 * half))
        for w in a.penalty_weights:
            sA = penalty_scale(fit(A, 0, 0.0), A) if w else None
            sB = penalty_scale(fit(B, 0, 0.0), B) if w else None
            dA, dB = fit(A, 0, w, sA), fit(B, 0, w, sB)
            results["stability"][f"split_w{w}"] = match_factors(trip(dA), trip(dB)).tolist()
            # neuron-set overlap of matched factors' top-5 units on a shared eval context
            with sdpa_kernel(SDPBackend.MATH):
                sp0 = spans.make_span(eval_ctxs[0])
                def units_of(dic, f):
                    spec = top_units(hidden_response(sp0, dic.L[:, f], dic.R[:, f], aj_layers), 5)
                    return {(L, int(i)) for L, m in spec.mlp_masks.items() for i in m.nonzero().flatten()}
                from scipy.optimize import linear_sum_assignment
                from circuit_checks.checks import factor_similarity
                S = factor_similarity(trip(dA), trip(dB)).cpu().numpy()
                rows, cols = linear_sum_assignment(-S)
                results["stability"][f"split_w{w}_top5_jaccard"] = [
                    jaccard(units_of(dA, i), units_of(dB, j)) for i, j in zip(rows, cols)]
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("stability:", {k: (np.round(np.mean(v), 3) if isinstance(v, list) and v and not isinstance(v[0], list)
                          else v if not isinstance(v, list) else [np.round(np.mean(x), 3) for x in v])
                         for k, v in results["stability"].items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
