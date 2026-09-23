"""Replication of AJ's branch-specific DCT (plans/BRANCH_DCT_PLAN_V1.md) with its closed form.

  python scripts/run_branch_dct_v1.py --smoke
  python scripts/run_branch_dct_v1.py --arch bilinear --device cuda --out results/branch_dct_bilinear.json
  python scripts/run_branch_dct_v1.py --arch bilinear-attn --device cuda --out results/branch_dct_bilinear-attn.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import grad_and_value, jvp, vjp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel
from scipy.optimize import linear_sum_assignment

from circuit_checks.core import FreezeSpec, apply_freeze
from circuit_checks.tensorgpt import TensorGPTSpans, hidden_features, down_project
from circuit_checks.checks import all_units, factor_similarity, match_factors_span
from circuit_checks.inrepo_model import load, state_before_block, collect_states, ROOT
from scripts.run_checks import advbench_texts, fineweb_texts

BARS = dict(parity=1e-6, linear=1e-4, heldout=0.9, ood=0.7, swap=0.1, top1=0.9, closed_form_rank=10, closed_form_count=20, curvature=0.2, seed_matches=3)


class BranchSpans(TensorGPTSpans):
    """Perturbation (theta_l, theta_r) enters the source block's MLP as Left(n + theta_l) * Right(n + theta_r); `mlp_add` adds a
    vector to the source MLP's output instead (used for the downstream Jacobian)."""
    def _run2(self, ctx, theta_l, theta_r, freeze=None, mlp_add=None):
        values, init, first = ctx; hidden = {}
        for rel, block in enumerate(self.blocks):
            L = self.source + rel
            values = block.lambdas[0] * values + block.lambdas[1] * init
            attn, first = block.attn(F.rms_norm(values, (self.d,)), first)
            if freeze is not None and L in freeze.attn_layers:
                attn = apply_freeze(attn, torch.ones(attn.shape[-1], device=attn.device), None)
            values = values + attn
            n = F.rms_norm(values, (self.d,))
            if rel == 0:
                h = block.mlp.Left(n + theta_l) * block.mlp.Right(n + theta_r)
            else:
                h = hidden_features(block.mlp, n)
            if freeze is not None and L in freeze.mlp_masks:
                h = apply_freeze(h, freeze.mlp_masks[L].to(h.device), None)
            hidden[L] = h[:, self.tp].mean(1)[0]
            out_mlp = down_project(block.mlp, h)
            if rel == 0 and mlp_add is not None:
                out_mlp = out_mlp + mlp_add
            values = values + out_mlp
        if self.final_norm:
            values = F.rms_norm(values, (self.d,))
        return values[:, self.tp].mean(1)[0], hidden

    def make_branch_span(self, ctx):
        return lambda tl, tr, freeze=None: self._run2(ctx, tl, tr, freeze)[0]

    def make_mlp_add_span(self, ctx, freeze=None):
        z = torch.zeros(self.d, device=ctx[0].device)
        return lambda m: self._run2(ctx, z, z, freeze, mlp_add=m)[0]


def ordered_cross(fn, l, r):
    """AJ's ordered_cross_hessian_outputs: d/dbeta d/dalpha fn(alpha l, beta r) at 0, l on the Left branch, r on the Right."""
    zl, zr = torch.zeros_like(l), torch.zeros_like(r)
    left_at = lambda cr: jvp(lambda cl: fn(cl, cr), (zl,), (l,))[1]
    return jvp(left_at, (zr,), (r,))[1]


class BranchDCT:
    """Port of BilinearQuadraticDCT.fit (same init, QR, gradient objective, fixed-point update)."""
    def __init__(self, num_factors):
        self.num_factors = num_factors

    def fit(self, make_span, contexts, d, d_out, max_iters, factor_batch, seed, device):
        torch.manual_seed(seed)
        self.L = F.normalize(torch.randn(d, self.num_factors, device=device), dim=0)
        self.R = F.normalize(torch.randn(d, self.num_factors, device=device), dim=0)
        self.U = F.normalize(torch.randn(d_out, self.num_factors, device=device), dim=0)
        self.score_energy_trace = []
        for _ in range(max_iters):
            with torch.no_grad():
                self.L, _ = torch.linalg.qr(self.L); self.R, _ = torch.linalg.qr(self.R)
            energy = torch.zeros(self.num_factors, device=device); gU, gL, gR = torch.zeros_like(self.U), torch.zeros_like(self.L), torch.zeros_like(self.R); n = 0
            for c in contexts:
                span = make_span(c)
                def objective(u, l, r):
                    s = u.float() @ ordered_cross(span, l, r).float()
                    return 0.5 * s.square(), s
                grads, (_, s) = vmap(grad_and_value(objective, argnums=(0, 1, 2), has_aux=True), in_dims=(1, 1, 1), out_dims=((1, 1, 1), (0, 0)), chunk_size=factor_batch)(self.U, self.L, self.R)
                with torch.no_grad():
                    energy += s.square(); gU += grads[0]; gL += grads[1]; gR += grads[2]; n += 1
            with torch.no_grad():
                self.U = F.normalize(gU / n, dim=0); self.L = F.normalize(gL / n, dim=0); self.R = F.normalize(gR / n, dim=0)
                self.score_energy_trace.append(float((energy / n).sum()))
        return self.U, self.L, self.R


def evaluate(dic, make_span, contexts):
    direct, swapped, asym = [], [], []
    with torch.no_grad():
        for c in contexts:
            span = make_span(c); dd, ss, aa = [], [], []
            for f in range(dic.num_factors):
                u, l, r = dic.U[:, f], dic.L[:, f], dic.R[:, f]
                D_ = ordered_cross(span, l, r).float(); S_ = ordered_cross(span, r, l).float()
                dd.append((u @ D_) ** 2); ss.append((u @ S_) ** 2); aa.append((D_ - S_).norm() / D_.norm().clamp_min(1e-30))
            direct.append(torch.stack(dd)); swapped.append(torch.stack(ss)); asym.append(torch.stack(aa))
    direct, swapped, asym = torch.stack(direct), torch.stack(swapped), torch.stack(asym)
    return dict(mean_total_energy=float(direct.sum(1).mean()), mean_swapped_total_energy=float(swapped.sum(1).mean()),
                swapped_to_direct_energy_ratio=float(swapped.sum(1).mean() / direct.sum(1).mean().clamp_min(1e-30)),
                median_response_swap_asymmetry=float(asym.median()), factor_mean_energy=direct.mean(0).cpu().tolist())


def source_weight_alignment(dic, mlp):
    A = mlp.Left.weight.float(); B = mlp.Right.weight.float(); An, Bn = F.normalize(A, dim=1), F.normalize(B, dim=1)
    ordered = (dic.L.T @ An.T).abs() * (dic.R.T @ Bn.T).abs(); swapped = (dic.L.T @ Bn.T).abs() * (dic.R.T @ An.T).abs()
    best, units = ordered.max(1); local = (A @ dic.L) * (B @ dic.R); energy = local.square()
    return dict(best_ordered_joint_weight_alignment=best.cpu().tolist(), best_swapped_joint_weight_alignment=swapped.max(1).values.cpu().tolist(),
                best_source_unit_indices=units.cpu().tolist(), source_hidden_top1_energy_fraction=(energy.max(0).values / energy.sum(0).clamp_min(1e-30)).cpu().tolist(),
                fitted_unit_by_energy=energy.argmax(0).cpu().tolist())


def cross_seed_alignment(a, b):
    direct = ((a.U.T @ b.U).abs() * (a.L.T @ b.L).abs() * (a.R.T @ b.R).abs()).cpu()
    rows, cols = linear_sum_assignment(direct.numpy(), maximize=True); m = direct[rows, cols]
    return dict(matched_direct_similarity=m.tolist(), median_direct_similarity=float(m.median()), maximum_direct_similarity=float(m.max()), direct_matches_above_0_8=int((m > 0.8).sum()),
                span_matches=match_factors_span((a.U, a.L, a.R), (b.U, b.L, b.R)).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--arch", default="bilinear"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--source-layer", type=int, default=8); ap.add_argument("--target-layer", type=int, default=12); ap.add_argument("--sequence-length", type=int, default=32)
    ap.add_argument("--train-contexts", type=int, default=32); ap.add_argument("--heldout-contexts", type=int, default=64); ap.add_argument("--ood-contexts", type=int, default=64)
    ap.add_argument("--factors", type=int, default=8); ap.add_argument("--iterations", type=int, default=30); ap.add_argument("--factor-batch", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--out", default="results/branch_dct.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer, a.target_layer = 1, 3; a.sequence_length = 8; a.train_contexts, a.heldout_contexts, a.ood_contexts = 3, 2, 2
        a.factors, a.iterations, a.seeds, a.factor_batch = 3, 3, [0, 1], 3
        sq = a.arch == "bilinear-attn"
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=sq, squared_attn=sq)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False)
        def mk(n, seed): return [torch.randint(0, 64, (1, a.sequence_length), generator=torch.Generator().manual_seed(seed + i)) for i in range(n)]
        sets = {"train": mk(3, 10), "heldout": mk(2, 20), "ood_wikitext": mk(2, 30), "ood_fineweb": mk(2, 40)}
    else:
        model, cfg, meta = load(a.arch, a.device)
        from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        tr, he = advbench_texts(a.train_contexts, a.heldout_contexts)
        from datasets import load_dataset
        wt = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip() and not t.strip().startswith("=")][:a.ood_contexts]
        fw = fineweb_texts(0, a.ood_contexts, a.sequence_length)[1]
        enc = lambda texts: [tok(t, return_tensors="pt", truncation=True, padding="max_length", max_length=a.sequence_length).input_ids for t in texts]
        sets = {"train": enc(tr), "heldout": enc(he), "ood_wikitext": enc(wt), "ood_fineweb": enc(fw)}
    dev = a.device; d = cfg.n_embd
    spans = BranchSpans(model, a.source_layer, a.target_layer, slice(-3, None))
    widths = {L: spans.mlp_width(L) for L in spans.all_layers}
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = {k: [ctx_of(ids) for ids in v] for k, v in sets.items()}
    zero = torch.zeros(d, device=dev)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_branch_span(ctxs["train"][0])(zero, zero)                                        # warm rotary caches
        spans.make_span(ctxs["train"][0])(zero)

    # ---- controls ------------------------------------------------------------------------------------------------------------------------
    mlp = model.transformer.h[a.source_layer].mlp; A_ = mlp.Left.weight.float(); B_ = mlp.Right.weight.float(); Dw = mlp.Down.weight.float()
    with sdpa_kernel(SDPBackend.MATH):
        c0 = ctxs["train"][0]; bs = spans.make_branch_span(c0)
        with torch.no_grad():
            parity = float((bs(zero, zero) - spans.make_span(c0)(zero)[0]).abs().max() / spans.make_span(c0)(zero)[0].abs().max())
        downstream = {L for L in spans.all_layers if L != a.source_layer}
        frz = FreezeSpec(mlp_masks=all_units(widths, list(downstream)).mlp_masks, attn_layers=set(spans.all_layers))   # everything but the source MLP: G linear
        g = torch.Generator().manual_seed(0); lin_err = []; swap_asym = []
        for _ in range(4 if not a.smoke else 2):
            l = F.normalize(torch.randn(d, generator=g), dim=0).to(dev); r = F.normalize(torch.randn(d, generator=g), dim=0).to(dev)
            score_frozen = ordered_cross(lambda tl, tr: bs(tl, tr, frz), l, r)
            m_lr = Dw @ ((A_ @ l) * (B_ @ r))
            lin = jvp(spans.make_mlp_add_span(c0, frz), (zero,), (m_lr,))[1]
            lin_err.append(float((score_frozen - lin).norm() / lin.norm().clamp_min(1e-30)))
            Dd = ordered_cross(bs, l, r); Ss = ordered_cross(bs, r, l); swap_asym.append(float((Dd - Ss).norm() / Dd.norm().clamp_min(1e-30)))
    linear_ctrl = float(max(lin_err))
    print(f"[controls] branch-span parity {parity:.2e} | linear-downstream identity max rel err {linear_ctrl:.2e} | random-direction swap asymmetry median {np.median(swap_asym):.3f}", flush=True)

    # ---- fits ----------------------------------------------------------------------------------------------------------------------------
    dics = {}; fits = []
    for seed in a.seeds:
        t0 = time.time(); dic = BranchDCT(a.factors)
        with sdpa_kernel(SDPBackend.MATH):
            dic.fit(spans.make_branch_span, ctxs["train"], d, d, a.iterations, a.factor_batch, seed, dev)
        dics[seed] = dic
        with sdpa_kernel(SDPBackend.MATH):
            ev = {k: evaluate(dic, spans.make_branch_span, ctxs[k]) for k in ("train", "heldout", "ood_wikitext", "ood_fineweb")}
        for k in ("ood_wikitext", "ood_fineweb"):
            ev[k]["energy_retention_vs_heldout"] = ev[k]["mean_total_energy"] / max(ev["heldout"]["mean_total_energy"], 1e-30)
        swa = source_weight_alignment(dic, mlp)
        # ---- closed form per factor: w_c = D^T J_c^T u over training contexts; ranking; top singular pair of M_u; curvature share
        closed = []
        with sdpa_kernel(SDPBackend.MATH):
            for f in range(a.factors):
                u, l, r = dic.U[:, f], dic.L[:, f], dic.R[:, f]; ws = []; curv = []
                for c in ctxs["train"]:
                    gfn = spans.make_mlp_add_span(c); _, pull = vjp(gfn, zero); w = Dw.T @ pull(u)[0]; ws.append(w)
                    score = float(u @ ordered_cross(spans.make_branch_span(c), l, r)); term1 = float(w @ ((A_ @ l) * (B_ @ r)))
                    curv.append(abs(score - term1) / max(abs(score), 1e-30))
                wbar = torch.stack(ws).mean(0); rank_score = wbar.abs() * A_.norm(dim=1) * B_.norm(dim=1)
                order = rank_score.argsort(descending=True); unit = swa["fitted_unit_by_energy"][f]; rank = int((order == unit).nonzero().flatten()[0])
                M = A_.T @ (wbar[:, None] * B_); Us, S, Vh = torch.linalg.svd(M)
                closed.append(dict(fitted_unit=unit, closed_form_rank_of_fitted_unit=rank, closed_form_top10_units=order[:10].cpu().tolist(),
                                   cos_l_top_singular=float((l @ Us[:, 0]).abs()), cos_r_top_singular=float((r @ Vh[0]).abs()), singular_value_pr=float((S.square().sum() ** 2 / S.pow(4).sum())),
                                   top_singular_share=float(S[0] ** 2 / S.square().sum()), curvature_share_median=float(np.median(curv)), w_top1_share=float(wbar.abs().max() / wbar.abs().sum())))
        fits.append(dict(seed=seed, trace=dic.score_energy_trace, **ev, source_weight_alignment=swa, closed_form=closed, seconds=time.time() - t0))
        print(f"seed {seed}: train {ev['train']['mean_total_energy']:.4g} heldout {ev['heldout']['mean_total_energy']:.4g} wikitext ret {ev['ood_wikitext']['energy_retention_vs_heldout']:.3f} fineweb ret {ev['ood_fineweb']['energy_retention_vs_heldout']:.3f} | swapped/direct {ev['heldout']['swapped_to_direct_energy_ratio']:.3f} | top1 energy frac {np.round(swa['source_hidden_top1_energy_fraction'], 3).tolist()} | closed-form rank of fitted unit {[x['closed_form_rank_of_fitted_unit'] for x in closed]} | curvature share {np.round([x['curvature_share_median'] for x in closed], 3).tolist()} | {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(args=vars(a), controls=dict(parity=parity, linear=linear_ctrl, swap_asymmetry=swap_asym), fits=fits), open(a.out, "w"), indent=1, default=float)
    stability = [dict(left_seed=s1, right_seed=s2, **cross_seed_alignment(dics[s1], dics[s2])) for i, s1 in enumerate(a.seeds) for s2 in a.seeds[i + 1:]]
    ranks = [x["closed_form_rank_of_fitted_unit"] for f in fits for x in f["closed_form"]]; curv = [x["curvature_share_median"] for f in fits for x in f["closed_form"]]
    preds = dict(pred_a_instrument=parity <= BARS["parity"] and linear_ctrl <= BARS["linear"],
                 pred_b_replicates=all(f["heldout"]["mean_total_energy"] >= BARS["heldout"] * f["train"]["mean_total_energy"] and f["ood_wikitext"]["energy_retention_vs_heldout"] >= BARS["ood"]
                                       and f["heldout"]["swapped_to_direct_energy_ratio"] <= BARS["swap"] and float(np.median(f["source_weight_alignment"]["source_hidden_top1_energy_fraction"])) >= BARS["top1"] for f in fits),
                 pred_c_closed_form_units=sum(rk < BARS["closed_form_rank"] for rk in ranks) >= (BARS["closed_form_count"] if not a.smoke else 0),
                 pred_d_first_order=float(np.median(curv)) <= BARS["curvature"], pred_e_seed_instability=all(s["direct_matches_above_0_8"] <= BARS["seed_matches"] for s in stability))
    res = dict(args=vars(a), predictions=preds, controls=dict(parity=parity, linear=linear_ctrl, swap_asymmetry=swap_asym), fits=fits, cross_seed_stability=stability, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], closed_form_ranks=ranks, curvature=curv, matches=[s["direct_matches_above_0_8"] for s in stability]), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
