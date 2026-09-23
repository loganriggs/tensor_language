"""Rung 1 of the symmetric DCT lane (plans/SYMMETRIC_DCT_PLAN_V1.md): exact interaction forms B_u for fixed reader
directions on bilin18, their rank, their generalisation, and mediator completeness of their top eigenvectors.

  python scripts/run_symmetric_dct_v1.py --smoke                       # tiny random model on CPU: exercises every code path
  python scripts/run_symmetric_dct_v1.py --device cuda --out results/symmetric_dct_v1.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, jvp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.core import FreezeSpec, mixed_hessian, participation_ratio
from circuit_checks.checks import hidden_response, top_units, random_units, all_units, output_score
from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.inrepo_model import load, state_before_block, ROOT

P = os.path.join(ROOT, "basis_aligned/polynomial_causal/direct_tensor_match")
BARS = dict(instrument=1e-3, freeze_all=1e-6, rank=32, generalise=0.5, off_dist=0.3, attention=0.9)


def full_hessian(f, d, dev, chunk):
    """H[k, i, j] = d^2 f_k / dtheta_i dtheta_j at 0 for f: R^d -> R^K, forward-over-reverse in chunks of tangents."""
    zero = torch.zeros(d, device=dev); E = torch.eye(d, device=dev)
    rows = vmap(lambda e: jvp(lambda th: jacrev(f)(th), (zero,), (e,))[1], chunk_size=chunk)(E)      # [d, K, d]
    return rows.permute(1, 0, 2).contiguous()


def readers_for(model, dev):
    w = torch.load(os.path.join(P, "EXPANDED_ROOT_EMPIRICAL_V1.pt"), weights_only=True)["writer"].to(dev).float()   # [1152, 16]
    W = model.lm_head.weight.float(); uw = W @ w
    R = W.T @ uw / uw.square().sum(0)
    return F.normalize(R, dim=0)                                                                                        # [1152, 16]


def ctx_at(state, i):
    return tuple(s[i:i + 1] for s in state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32)
    ap.add_argument("--fit-contexts", type=int, default=32)
    ap.add_argument("--heldout-contexts", type=int, default=16)
    ap.add_argument("--advbench-contexts", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--top-r", type=int, default=8)
    ap.add_argument("--out", default="results/symmetric_dct_v1.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.fit_contexts, a.heldout_contexts, a.advbench_contexts = 3, 2, 2; a.sequence_length = 8; a.chunk = 8; a.top_r = 3
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); K = 4
        readers = F.normalize(torch.randn(cfg.n_embd, K), dim=0)
        g = torch.Generator().manual_seed(1)
        mk = lambda n, seed: [torch.randint(0, 64, (1, a.sequence_length), generator=torch.Generator().manual_seed(seed + i)) for i in range(n)]
        fit_ids, held_ids, adv_ids = mk(3, 10), mk(2, 20), mk(2, 30)
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device); K = readers.shape[1]
        import tiktoken; enc = tiktoken.get_encoding("gpt2")
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu")
        fit_ids = [rows[i:i + 1, :a.sequence_length].long() for i in range(a.fit_contexts)]
        held_ids = [rows[96 + i:97 + i, :a.sequence_length].long() for i in range(a.heldout_contexts)]
        from transformers import GPT2Tokenizer; from scripts.run_checks import advbench_texts
        tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        _, adv = advbench_texts(32, a.advbench_contexts)
        adv_ids = [tok(t, return_tensors="pt", truncation=True, padding="max_length", max_length=a.sequence_length).input_ids for t in adv]
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True)
    widths = {L: spans.mlp_width(L) for L in spans.all_layers}

    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())

    def hess(ctx):
        span = spans.make_span(ctx); U = readers
        return full_hessian(lambda th: U.T @ span(th, None)[0], d, dev, a.chunk)                       # [K, d, d]

    # warm every block's rotary cos/sin cache outside any functorch transform (the cache is created on first call at this T;
    # created inside vmap/jvp it escapes the transform level and torch.func raises)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        c0 = ctx_of(fit_ids[0]); spans.make_span(c0)(torch.zeros(d, device=dev), None)
    # ---- instrument controls on the first fitting context --------------------------------------------------------------------
    with sdpa_kernel(SDPBackend.MATH):
        H0 = hess(c0); span0 = spans.make_span(c0)
        sym = float((H0 - H0.transpose(1, 2)).norm() / H0.norm())
        g = torch.Generator().manual_seed(0); errs = []
        for k in range(K):
            for _ in range(8 if not a.smoke else 3):
                l = F.normalize(torch.randn(d, generator=g), dim=0).to(dev); r = F.normalize(torch.randn(d, generator=g), dim=0).to(dev)
                direct = float(output_score(span0, readers[:, k], l, r)); viaH = float(l @ H0[k] @ r)
                errs.append(abs(direct - viaH) / max(abs(direct), 1e-12))
        instrument = float(np.median(errs)); instrument_max = float(max(errs))
        allspec = FreezeSpec(mlp_masks=all_units(widths, spans.all_layers).mlp_masks, attn_layers=set(spans.all_layers))
        v = torch.linalg.eigh(H0[0])[1][:, -1]
        # freeze-all control on the span WITHOUT the final norm: with every MLP and attention block frozen the residual pass-through
        # is linear and the mixed derivative must vanish; with the final rms_norm included its curvature alone leaves a remainder,
        # which is reported separately as `final_norm_only`.
        span_nonorm = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=False).make_span(c0)
        freeze_all = abs(float(output_score(span_nonorm, readers[:, 0], v, v, allspec))) / max(abs(float(output_score(span_nonorm, readers[:, 0], v, v))), 1e-30)
        final_norm_only = float(output_score(span0, readers[:, 0], v, v, allspec)) / float(output_score(span0, readers[:, 0], v, v))
    print(f"[controls] hessian vs jvp-of-jvp rel err median {instrument:.2e} max {instrument_max:.2e} | symmetry {sym:.2e} | freeze-all residual (no final norm) {freeze_all:.2e} | final-norm-only share {final_norm_only:.3f}", flush=True)

    # ---- Hessians on all contexts ------------------------------------------------------------------------------------------------
    def hess_all(id_list, name):
        out = []; t0 = time.time()
        with sdpa_kernel(SDPBackend.MATH):
            for i, ids in enumerate(id_list):
                out.append(hess(ctx_of(ids)).cpu())
                if i == 0 or (i + 1) % 8 == 0: print(f"[hessians] {name} {i + 1}/{len(id_list)} ({time.time() - t0:.0f}s)", flush=True)
        return torch.stack(out)                                                                            # [N, K, d, d]
    Hfit = hess_all(fit_ids, "fit"); Hheld = hess_all(held_ids, "heldout"); Hadv = hess_all(adv_ids, "advbench")
    Hbar = Hfit.mean(0)                                                                                    # [K, d, d]
    ev, V = torch.linalg.eigh(Hbar)                                                                        # ascending
    order = ev.abs().argsort(dim=1, descending=True)
    lam = torch.gather(ev, 1, order); Vs = torch.stack([V[k][:, order[k]] for k in range(K)])              # [K, d, d] columns sorted by |lambda|
    pr = participation_ratio(lam).tolist(); top = lam[:, :a.top_r].tolist(); Vr = Vs[:, :, :a.top_r]        # [K, d, r]

    def fractions(H):
        """Per context and reader: Frobenius fraction in span(V_r), and the AJ-score fraction relative to the context's own top-r eigvecs."""
        frob, score = [], []
        for c in range(H.shape[0]):
            fr, sc = [], []
            for k in range(K):
                B = H[c, k]; P_ = Vr[k]; proj = P_.T @ B @ P_
                fr.append(float(proj.square().sum() / B.square().sum()))
                own = torch.linalg.eigh(B)[1]; ownr = own[:, own.shape[1] - a.top_r:]
                s_fixed = float(torch.stack([(P_[:, j] @ B @ P_[:, j]) ** 2 for j in range(a.top_r)]).sum())
                s_own = float(torch.stack([(ownr[:, j] @ B @ ownr[:, j]) ** 2 for j in range(a.top_r)]).sum())
                sc.append(s_fixed / max(s_own, 1e-30))
            frob.append(fr); score.append(sc)
        return np.array(frob), np.array(score)
    frob_fit, score_fit = fractions(Hfit); frob_held, score_held = fractions(Hheld); frob_adv, score_adv = fractions(Hadv)
    # cross-context consistency of the top eigenvector: does v_1 of the mean form score with the same sign per context?
    sign_consistency = [float(np.mean([np.sign(float(Vr[k][:, 0] @ Hheld[c, k] @ Vr[k][:, 0])) == np.sign(float(lam[k, 0])) for c in range(Hheld.shape[0])])) for k in range(K)]
    print(f"[rank] PR of eigenvalues per reader {np.round(pr, 1).tolist()} | top-r Frobenius fraction: fit {np.median(frob_fit):.3f} heldout {np.median(frob_held):.3f} advbench {np.median(frob_adv):.3f} | score fraction heldout {np.median(score_held):.3f}", flush=True)

    # ---- mediator completeness of v_1 per reader on held-out contexts --------------------------------------------------------------
    held_ctx = [ctx_of(ids) for ids in held_ids]; comp = {}; g = torch.Generator().manual_seed(0)
    with sdpa_kernel(SDPBackend.MATH):
        for k in range(K):
            u = readers[:, k]; v = Vr[k][:, 0].to(dev); rec = {}
            groups = {"all_attention": lambda sp: FreezeSpec(attn_layers=set(spans.all_layers)),
                      "source_attention": lambda sp: FreezeSpec(attn_layers={a.source_layer}),
                      "all_mlps": lambda sp: all_units(widths, spans.all_layers),
                      "source_mlp": lambda sp: all_units(widths, [a.source_layer]),
                      "everything": lambda sp: allspec}
            for L in spans.all_layers:
                groups[f"attention_block_{L}"] = (lambda sp, L=L: FreezeSpec(attn_layers={L}))
            for kk in (1, 5, 20):
                groups[f"top{kk}_units"] = (lambda sp, kk=kk: top_units(hidden_response(sp, v, v, spans.all_layers), kk))
            groups["random5_units"] = lambda sp: random_units(widths, 5, g)
            full = []; fr = {n: [] for n in groups}
            for c in held_ctx:
                sp = spans.make_span(c); f0 = float(output_score(sp, u, v, v)); full.append(f0)
                for n, build in groups.items():
                    fr[n].append(1 - float(output_score(sp, u, v, v, build(sp))) / f0 if abs(f0) > 1e-12 else float("nan"))
            comp[k] = {n: float(np.nanmedian(x)) for n, x in fr.items()}; comp[k]["_full_score_mean_abs"] = float(np.mean(np.abs(full)))
            print(f"[completeness] reader {k}: attention {comp[k]['all_attention']:.2f} source-attn {comp[k]['source_attention']:.2f} mlps {comp[k]['all_mlps']:.2f} source-mlp {comp[k]['source_mlp']:.2f} top1 {comp[k]['top1_units']:.2f} top20 {comp[k]['top20_units']:.2f} everything {comp[k]['everything']:.2f}", flush=True)
    med = lambda key: float(np.median([comp[k][key] for k in range(K)]))
    preds = dict(pred_a_instrument=instrument <= BARS["instrument"] and sym <= BARS["instrument"] and freeze_all <= BARS["freeze_all"],
                 pred_b_low_rank=float(np.median(pr)) <= BARS["rank"], pred_c_generalises=float(np.median(frob_held)) >= BARS["generalise"],
                 pred_d_off_distribution=float(np.median(frob_adv)) >= BARS["off_dist"], pred_e_attention_carried=med("all_attention") >= BARS["attention"])
    torch.save(dict(Hbar=Hbar, eigenvalues=lam, eigenvectors_top=Vs[:, :, :64], readers=readers.cpu()), a.out.replace(".json", "_forms.pt"))
    res = dict(args=vars(a), predictions=preds, controls=dict(hessian_vs_jvp_median=instrument, hessian_vs_jvp_max=instrument_max, symmetry=sym, freeze_all=freeze_all, final_norm_only=final_norm_only),
               eigen=dict(participation_ratio=pr, top_eigenvalues=top, sign_consistency_heldout=sign_consistency, frobenius_norm=Hbar.flatten(1).norm(dim=1).tolist()),
               fractions=dict(frobenius=dict(fit=frob_fit.tolist(), heldout=frob_held.tolist(), advbench=frob_adv.tolist()), score=dict(fit=score_fit.tolist(), heldout=score_held.tolist(), advbench=score_adv.tolist()),
                              medians=dict(frob_fit=float(np.median(frob_fit)), frob_heldout=float(np.median(frob_held)), frob_advbench=float(np.median(frob_adv)), score_heldout=float(np.median(score_held)), score_advbench=float(np.median(score_adv)))),
               completeness=comp, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], medians=res["fractions"]["medians"], median_pr=float(np.median(pr))), indent=1), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
