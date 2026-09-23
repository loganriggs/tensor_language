"""Rung 2 of the symmetric DCT lane (plans/SYMMETRIC_DCT_PLAN_V2.md): per-context rank, cross-context consistency and per-head
completeness of the interaction forms for the 16 readers on bilin18.

  python scripts/run_symmetric_dct_v2.py --smoke
  python scripts/run_symmetric_dct_v2.py --device cuda --out results/symmetric_dct_v2.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec, participation_ratio
from circuit_checks.checks import output_score
from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.heads import head_parity
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for

BARS = dict(parity=1e-5, heads_vs_layer=1e-4, symmetry=1e-3, rank=32, shared=0.25, few_heads=0.5, same_head=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--source-layer", type=int, default=8); ap.add_argument("--sequence-length", type=int, default=32)
    ap.add_argument("--heldout-contexts", type=int, default=16); ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--top-r", type=int, default=8)
    ap.add_argument("--forms", default="results/symmetric_dct_v1_forms.pt"); ap.add_argument("--out", default="results/symmetric_dct_v2.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.heldout_contexts = 3; a.sequence_length = 8; a.chunk = 8; a.top_r = 3
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); K = 3; readers = F.normalize(torch.randn(cfg.n_embd, K), dim=0)
        held_ids = [torch.randint(0, 64, (1, a.sequence_length), generator=torch.Generator().manual_seed(20 + i)) for i in range(3)]
        Hbar = None
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device); K = readers.shape[1]
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu")
        held_ids = [rows[96 + i:97 + i, :a.sequence_length].long() for i in range(a.heldout_contexts)]
        forms = torch.load(a.forms, weights_only=True); Hbar = forms["Hbar"].float()
        assert torch.allclose(forms["readers"].to(readers.device), readers, atol=1e-5), "readers differ from rung 1"
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; H = cfg.n_head
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True, per_head=True)
    layers = spans.all_layers

    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(ids) for ids in held_ids]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(ctxs[0])(torch.zeros(d, device=dev), None)                        # warm rotary caches outside transforms
    # ---- controls -------------------------------------------------------------------------------------------------------------------
    with sdpa_kernel(SDPBackend.MATH):
        v0, x0, f0 = ctxs[0]; xin = F.rms_norm(model.transformer.h[a.source_layer].lambdas[0] * v0 + model.transformer.h[a.source_layer].lambdas[1] * x0, (d,))
        parity = head_parity(model.transformer.h[a.source_layer].attn, xin, f0)
        sp0 = spans.make_span(ctxs[0]); u0 = readers[:, 0]; g = torch.Generator().manual_seed(0); v = F.normalize(torch.randn(d, generator=g), dim=0).to(dev)
        L0 = layers[len(layers) // 2]
        s_heads = float(output_score(sp0, u0, v, v, FreezeSpec(head_masks={L0: torch.ones(H)}))); s_layer = float(output_score(sp0, u0, v, v, FreezeSpec(attn_layers={L0})))
        heads_vs_layer = abs(s_heads - s_layer) / max(abs(s_layer), 1e-30)
    print(f"[controls] per-head replica parity {parity:.2e} | all-heads vs layer freeze {heads_vs_layer:.2e}", flush=True)

    # ---- per-context Hessians ----------------------------------------------------------------------------------------------------------
    Hc = []; t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for i, c in enumerate(ctxs):
            span = spans.make_span(c); Hc.append(full_hessian(lambda th: readers.T @ span(th, None)[0], d, dev, a.chunk).cpu())
            if i == 0 or (i + 1) % 8 == 0: print(f"[hessians] {i + 1}/{len(ctxs)} ({time.time() - t0:.0f}s)", flush=True)
    Hc = torch.stack(Hc)                                                                          # [N, K, d, d]
    sym = float(max((Hc[c, k] - Hc[c, k].T).norm() / Hc[c, k].norm() for c in range(len(ctxs)) for k in range(K)))
    if Hbar is None:
        Hbar = Hc.mean(0)
    N = Hc.shape[0]
    pr_c = np.zeros((N, K)); own_frac = np.zeros((N, K)); cos_bar = np.zeros((N, K)); v1c = torch.zeros(N, K, d); v1bar = torch.zeros(K, d)
    for k in range(K):
        e, V = torch.linalg.eigh(Hbar[k]); v1bar[k] = V[:, e.abs().argmax()]
        for c in range(N):
            B = Hc[c, k]; e, V = torch.linalg.eigh(B); order = e.abs().argsort(descending=True)
            pr_c[c, k] = participation_ratio(e).item(); top = V[:, order[:a.top_r]]; own_frac[c, k] = float((top.T @ B @ top).square().sum() / B.square().sum())
            v1c[c, k] = V[:, order[0]]; cos_bar[c, k] = float((B * Hbar[k]).sum() / (B.norm() * Hbar[k].norm()))
    norms = Hc.flatten(2).norm(dim=2)                                                             # [N, K]
    consistency = (Hbar.flatten(1).norm(dim=1).square() / norms.square().mean(0)).tolist()
    pair_cos = []
    for k in range(K):
        flat = Hc[:, k].flatten(1); G = flat @ flat.T; nrm = flat.norm(dim=1); C = G / (nrm[:, None] * nrm[None, :])
        pair_cos.append(float(C[np.triu_indices(N, 1)].mean()))
    print(f"[rank] per-context eigen-PR median {np.median(pr_c):.1f} (min {pr_c.min():.1f} max {pr_c.max():.1f}) | own top-{a.top_r} Frobenius fraction median {np.median(own_frac):.3f} | consistency ratio per reader {np.round(consistency, 3).tolist()} | mean pairwise context cosine {np.round(pair_cos, 3).tolist()}", flush=True)

    # ---- per-head completeness ----------------------------------------------------------------------------------------------------------
    heads = [(L, h) for L in layers for h in range(H)]
    def removals(sp, u, v):
        full = float(output_score(sp, u, v, v)); out = {}
        for (L, h) in heads:
            m = torch.zeros(H); m[h] = 1
            out[(L, h)] = 1 - float(output_score(sp, u, v, v, FreezeSpec(head_masks={L: m}))) / full if abs(full) > 1e-12 else float("nan")
        blocks = {L: 1 - float(output_score(sp, u, v, v, FreezeSpec(attn_layers={L}))) / full for L in layers}
        top3 = sorted(out, key=lambda hh: -abs(out[hh]))[:3]; masks = {}
        for (L, h) in top3:
            masks.setdefault(L, torch.zeros(H))[h] = 1
        t3 = 1 - float(output_score(sp, u, v, v, FreezeSpec(head_masks=masks))) / full
        return full, out, blocks, top3, t3
    rec = {"bar": [], "ctx": []}; t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for c in range(N):
            sp = spans.make_span(ctxs[c])
            for k in range(K):
                for name, v in (("bar", v1bar[k]), ("ctx", v1c[c, k])):
                    full, single, blocks, top3, t3 = removals(sp, readers[:, k], v.to(dev))
                    rec[name].append(dict(context=c, reader=k, full=full, top3=[list(map(int, hh)) for hh in top3], top3_removed=t3,
                                          best_head=list(map(int, top3[0])), best_head_removed=single[top3[0]], head_removals={f"{L}.{h}": single[(L, h)] for (L, h) in heads}, block_removals=blocks))
            print(f"[heads] context {c + 1}/{N} ({time.time() - t0:.0f}s): ctx-v1 top-3 removed median so far {np.nanmedian([r['top3_removed'] for r in rec['ctx']]):.2f}", flush=True)
    same_head = []
    for k in range(K):
        bh = [tuple(r["best_head"]) for r in rec["ctx"] if r["reader"] == k]; most = max(set(bh), key=bh.count); same_head.append(bh.count(most) / len(bh))
    t3_ctx = float(np.nanmedian([r["top3_removed"] for r in rec["ctx"]])); t3_bar = float(np.nanmedian([r["top3_removed"] for r in rec["bar"]]))
    preds = dict(pred_a_instrument=parity <= BARS["parity"] and heads_vs_layer <= BARS["heads_vs_layer"] and sym <= BARS["symmetry"],
                 pred_b_low_rank_per_context=float(np.median(pr_c)) <= BARS["rank"], pred_c_shared_energy=float(np.median(consistency)) >= BARS["shared"],
                 pred_d_few_heads=t3_ctx >= BARS["few_heads"], pred_e_same_head=float(np.median(same_head)) >= BARS["same_head"])
    res = dict(args=vars(a), predictions=preds, controls=dict(parity=parity, heads_vs_layer=heads_vs_layer, symmetry=sym),
               per_context=dict(eigen_pr=pr_c.tolist(), own_top_r_frobenius=own_frac.tolist(), cosine_to_mean_form=cos_bar.tolist(), frobenius_norms=norms.tolist()),
               consistency_ratio=consistency, mean_pairwise_context_cosine=pair_cos, same_head_fraction=same_head,
               summary=dict(median_eigen_pr=float(np.median(pr_c)), median_own_frac=float(np.median(own_frac)), median_consistency=float(np.median(consistency)),
                            median_top3_removed_ctx=t3_ctx, median_top3_removed_bar=t3_bar, median_best_head_removed_ctx=float(np.nanmedian([r["best_head_removed"] for r in rec["ctx"]])),
                            median_same_head=float(np.median(same_head))),
               completeness=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], summary=res["summary"]), indent=1), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
