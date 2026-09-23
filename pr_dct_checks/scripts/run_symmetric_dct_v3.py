"""Rung 3 of the symmetric DCT lane (plans/SYMMETRIC_DCT_PLAN_V3.md): fixed weight-space head-form dictionary fitted to each
context's exact interaction form.

  python scripts/run_symmetric_dct_v3.py --smoke
  python scripts/run_symmetric_dct_v3.py --device cuda --out results/symmetric_dct_v3.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for

BARS = dict(shortcut=1e-4, planted_r2=0.999, planted_coef=1e-3, noise=0.02, captured=0.5, blocks89=0.8, offsets=1.5, sparse=0.8)


def rot_matrix(t, inv_freq, D):
    """128 x 128 matrix of apply_rotary_emb at position t: pairs (k, k + D/2) rotated by t * inv_freq[k]."""
    R = torch.zeros(D, D, dtype=torch.float64); c = torch.cos(t * inv_freq.double()); s = torch.sin(t * inv_freq.double()); h = D // 2
    idx = torch.arange(h)
    R[idx, idx] = c; R[idx, idx + h] = s; R[idx + h, idx] = -s; R[idx + h, idx + h] = c
    return R


def build_dictionary(model, layers, T):
    """Forms F = sym(A^T R(-delta) C) indexed by (head-type a, offset delta). Returns factors A, C [n_ht, D, d], rotations R [T, D, D], index table."""
    A, C, table = [], [], []
    for L in layers:
        at = model.transformer.h[L].attn; H, D = at.n_head, at.head_dim
        for h in range(H):
            sl = slice(h * D, (h + 1) * D)
            for typ, (wq, wk) in (("qk", (at.c_q.weight, at.c_k.weight)), ("q2k2", (at.c_q2.weight, at.c_k2.weight))):
                A.append(wq.float()[sl].double()); C.append(wk.float()[sl].double()); table.append((L, h, typ))
    A = torch.stack(A); C = torch.stack(C)
    inv_freq = model.transformer.h[layers[0]].attn.rotary.inv_freq.float(); D = A.shape[1]
    R = torch.stack([rot_matrix(torch.tensor(-float(delta)), inv_freq, D) for delta in range(T)])                    # R(-delta)
    return A, C, R, table


class FormDictionary:
    """Exact inner products and Gram of sym(A^T R C) forms via head-space contractions."""
    def __init__(self, A, C, R, dev):
        self.A, self.C, self.R = A.to(dev), C.to(dev), R.to(dev); self.n_ht, self.D, self.d = A.shape; self.T = R.shape[0]; self.dev = dev
        self.n = self.n_ht * self.T

    def index(self, ht, delta):
        return ht * self.T + delta

    def inner(self, B):
        """<F_a, B> for all forms, B symmetric [d, d] (float64). <sym(A^T R C), B> = <R, A B C^T>."""
        M = torch.einsum("hpd,de,hqe->hpq", self.A, B, self.C)                                    # [n_ht, D, D] = A B C^T
        return torch.einsum("tpq,hpq->ht", self.R, M).reshape(-1)                                # <R_t, M_h>

    def gram(self):
        """G[a, b] = <sym(F_a), sym(F_b)> = 1/2 [tr(R_a^T AA R_b CC) + tr(R_a^T AC R_b^T CA)] with AA = A_a A_b^T etc."""
        n_ht, T, D = self.n_ht, self.T, self.D
        AA = torch.einsum("apd,bqd->abpq", self.A, self.A); CC = torch.einsum("bpd,aqd->abpq", self.C, self.C)      # CC[a,b] = C_b C_a^T
        AC = torch.einsum("apd,bqd->abpq", self.A, self.C); CA = torch.einsum("bpd,aqd->abpq", self.A, self.C)      # CA[a,b] = A_b C_a^T
        G = torch.zeros(n_ht, T, n_ht, T, dtype=torch.float64, device=self.dev)
        for a in range(n_ht):
            X1 = torch.einsum("tqp,bqr->tbpr", self.R, AA[a])                                   # R_t^T AA[a,b]  -> [T, n_ht, D, D]
            Y1 = torch.einsum("sqr,brz->sbqz", self.R, CC[a])                                   # R_s CC[a,b]    -> [T, n_ht, D, D]
            term1 = torch.einsum("tbpr,sbrp->tbs", X1, Y1)                                       # tr(R_t^T AA R_s CC)
            X2 = torch.einsum("tqp,bqr->tbpr", self.R, AC[a])                                   # R_t^T AC[a,b]
            Y2 = torch.einsum("srq,brz->sbqz", self.R, CA[a])                                   # R_s^T CA[a,b]
            term2 = torch.einsum("tbpr,sbrp->tbs", X2, Y2)
            G[a] = 0.5 * (term1 + term2).permute(0, 1, 2)
        return G.reshape(self.n, self.n)

    def materialise(self, ht, delta):
        return 0.5 * (self.A[ht].T @ self.R[delta] @ self.C[ht] + (self.A[ht].T @ self.R[delta] @ self.C[ht]).T)


def solve(G, b, idx, ridge):
    Gs = G[idx][:, idx]; bs = b[idx]
    Gs = Gs + ridge * Gs.diagonal().mean() * torch.eye(len(idx), dtype=Gs.dtype, device=Gs.device)
    alpha = torch.linalg.solve(Gs, bs)
    return alpha, float(alpha @ bs)                                                              # fitted energy = alpha^T b (LS projection)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--heldout-contexts", type=int, default=16); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-6); ap.add_argument("--out", default="results/symmetric_dct_v3.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.heldout_contexts = 3; a.sequence_length = 8; a.chunk = 8
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); K = 3; readers = F.normalize(torch.randn(cfg.n_embd, K), dim=0)
        held_ids = [torch.randint(0, 64, (1, a.sequence_length), generator=torch.Generator().manual_seed(20 + i)) for i in range(3)]
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device); K = readers.shape[1]
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu")
        held_ids = [rows[96 + i:97 + i, :a.sequence_length].long() for i in range(a.heldout_contexts)]
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; T = a.sequence_length
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True); layers = spans.all_layers
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(ids) for ids in held_ids]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(ctxs[0])(torch.zeros(d, device=dev), None)
    # ---- dictionary + Gram --------------------------------------------------------------------------------------------------------
    A, C, R, table = build_dictionary(model, layers, T); dic = FormDictionary(A, C, R, dev); t0 = time.time()
    G = dic.gram(); print(f"[dictionary] {dic.n} forms ({dic.n_ht} head-types x {T} offsets); Gram in {time.time() - t0:.0f}s; diag range {float(G.diagonal().min()):.3g}..{float(G.diagonal().max()):.3g}", flush=True)
    idx_all = torch.arange(dic.n, device=dev)
    blocks89 = [i for i in range(dic.n) if table[i // T][0] in (a.source_layer, a.source_layer + 1)]
    delta0 = [i for i in range(dic.n) if i % T == 0]
    # ---- controls --------------------------------------------------------------------------------------------------------------------
    g = torch.Generator().manual_seed(0); short_err = []; gram_err = []
    Btest = torch.randn(d, d, generator=g, dtype=torch.float64); Btest = (Btest + Btest.T).to(dev)
    b_test = dic.inner(Btest)
    picks = [(int(torch.randint(0, dic.n_ht, (1,), generator=g)), int(torch.randint(0, T, (1,), generator=g))) for _ in range(6)]
    mats = {p: dic.materialise(*p) for p in picks}
    for p in picks:
        short_err.append(abs(float((mats[p] * Btest).sum()) - float(b_test[dic.index(*p)])) / max(abs(float((mats[p] * Btest).sum())), 1e-30))
    for i in range(6):
        p, q = picks[i], picks[(i + 1) % 6]; direct = float((mats[p] * mats[q]).sum()); viaG = float(G[dic.index(*p), dic.index(*q)])
        gram_err.append(abs(direct - viaG) / max(abs(direct), 1e-30))
    coef = torch.tensor([1.5, -0.7, 2.2], dtype=torch.float64, device=dev); planted = sum(c * mats[p] for c, p in zip(coef, picks[:3]))
    alpha_p, e_p = solve(G, dic.inner(planted), idx_all, a.ridge); r2_planted = e_p / float(planted.square().sum())
    coef_err = float(max(abs(float(alpha_p[dic.index(*p)]) - float(c)) for c, p in zip(coef, picks[:3])))
    print(f"[controls] shortcut max rel err {max(short_err):.2e} | gram max rel err {max(gram_err):.2e} | planted R2 {r2_planted:.6f} coef err {coef_err:.2e}", flush=True)
    # ---- Hessians and fits ---------------------------------------------------------------------------------------------------------
    rec = []; t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            span = spans.make_span(ctx); Hc = full_hessian(lambda th: readers.T @ span(th, None)[0], d, dev, a.chunk).double()
            for k in range(K):
                B = 0.5 * (Hc[k] + Hc[k].T); nB = float(B.square().sum()); b = dic.inner(B)
                al, e_all = solve(G, b, idx_all, a.ridge); r2 = e_all / nB
                _, e89 = solve(G, b, torch.tensor(blocks89, device=dev), a.ridge); _, e0 = solve(G, b, torch.tensor(delta0, device=dev), a.ridge)
                weight = (al.abs() * G.diagonal().sqrt()); order = weight.argsort(descending=True); top = {}
                for kk in (10, 50, 200):
                    if kk < dic.n:
                        _, ek = solve(G, b, order[:kk], a.ridge); top[kk] = ek / nB
                rec.append(dict(context=c, reader=k, norm2=nB, r2=r2, r2_blocks89=e89 / nB, r2_delta0=e0 / nB, r2_top=top, alpha=al.cpu().numpy().astype(np.float32),
                                top10=[(table[int(i) // T][0], table[int(i) // T][1], table[int(i) // T][2], int(i) % T) for i in order[:10]]))
            print(f"[fit] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): R2 median so far {np.median([r['r2'] for r in rec]):.3f} blocks8-9 {np.median([r['r2_blocks89'] for r in rec]):.3f} delta0 {np.median([r['r2_delta0'] for r in rec]):.3f}", flush=True)
            if c == 0:
                noise = []
                for _ in range(8 if not a.smoke else 3):
                    N_ = torch.randn(d, d, generator=g, dtype=torch.float64); N_ = (N_ + N_.T).to(dev); N_ = N_ * (nB ** 0.5 / N_.norm())
                    _, en = solve(G, dic.inner(N_), idx_all, a.ridge); noise.append(en / float(N_.square().sum()))
                print(f"[controls] noise baseline R2 median {np.median(noise):.4f}", flush=True)
    # ---- stability of coefficients across contexts, per reader ------------------------------------------------------------------------
    stab = []
    for k in range(K):
        al = np.stack([r["alpha"] for r in rec if r["reader"] == k]); Cm = np.corrcoef(al); stab.append(float(np.mean(Cm[np.triu_indices(len(al), 1)])))
    r2s = np.array([r["r2"] for r in rec]); ratio89 = np.array([r["r2_blocks89"] / max(r["r2"], 1e-12) for r in rec]); ratio0 = np.array([r["r2"] / max(r["r2_delta0"], 1e-12) for r in rec])
    ratio50 = np.array([r["r2_top"].get(50, np.nan) / max(r["r2"], 1e-12) for r in rec])
    preds = dict(pred_a_instrument=max(short_err) <= BARS["shortcut"] and max(gram_err) <= BARS["shortcut"] and r2_planted >= BARS["planted_r2"] and coef_err <= BARS["planted_coef"] and float(np.median(noise)) <= BARS["noise"],
                 pred_b_captured=float(np.median(r2s)) >= BARS["captured"], pred_c_blocks_8_9=float(np.median(ratio89)) >= BARS["blocks89"],
                 pred_d_offsets_matter=float(np.median(ratio0)) >= BARS["offsets"], pred_e_sparse=float(np.nanmedian(ratio50)) >= BARS["sparse"])
    for r in rec:
        del r["alpha"]
    res = dict(args=vars(a), predictions=preds, controls=dict(shortcut_max=max(short_err), gram_max=max(gram_err), planted_r2=r2_planted, planted_coef_err=coef_err, noise_r2=noise),
               dictionary=dict(n_forms=dic.n, head_types=dic.n_ht, offsets=T), summary=dict(median_r2=float(np.median(r2s)), median_r2_blocks89=float(np.median([r["r2_blocks89"] for r in rec])),
               median_r2_delta0=float(np.median([r["r2_delta0"] for r in rec])), median_ratio_blocks89=float(np.median(ratio89)), median_ratio_offsets=float(np.median(ratio0)),
               median_ratio_top50=float(np.nanmedian(ratio50)), median_r2_top=({kk: float(np.median([r["r2_top"][kk] for r in rec])) for kk in rec[0]["r2_top"]}), alpha_correlation_per_reader=stab),
               records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], summary=res["summary"]), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
