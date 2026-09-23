"""Circuits lane rung C1b (plans/CIRCUITS_PLAN_V2.md): weight-space dictionary sym((T^T a)(T'^T b)^T) over transport heads and
MLP unit forms, fitted to exact per-context interaction forms.

  python scripts/run_circuits_c1b.py --smoke
  python scripts/run_circuits_c1b.py --device cuda --c1a results/circuits_c1a.json --out results/circuits_c1b.json
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.tensorgpt import TensorGPTSpans
from circuit_checks.heads import bilinear_attn_forward
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for
from scripts.run_symmetric_dct_v3 import solve

BARS = dict(shortcut=1e-10, planted=0.999, physics=0.9, captured=0.5, transport=1.5, corr=0.5, tokens=0.8)


class RankOneDictionary:
    """Forms sym(x y^T) from vector pairs; exact inner products and Gram from the vectors."""
    def __init__(self, X, Y, labels):
        self.X, self.Y, self.labels = X, Y, labels; self.n = X.shape[0]

    def inner(self, B):
        return torch.einsum("nd,de,ne->n", self.X, B, self.Y)                                   # <sym(xy^T), B> = x^T B y for symmetric B

    def gram(self):
        XX = self.X @ self.X.T; YY = self.Y @ self.Y.T; XY = self.X @ self.Y.T
        return 0.5 * (XX * YY + XY * XY.T)

    def materialise(self, i):
        return 0.5 * (torch.outer(self.X[i], self.Y[i]) + torch.outer(self.Y[i], self.X[i]))


def head_transport(attn, h):
    """M_h = C_h W_v^h: the head's value map followed by its output-projection slice (rank <= head_dim)."""
    D = attn.head_dim; sl = slice(h * D, (h + 1) * D)
    Wv = attn.c_v.weight.float()[sl]                                                            # [D, d]
    Cp = attn.c_proj.weight.float()[:, sl]                                                       # [d, D]
    return Cp @ Wv                                                                               # [d, d]


def attention_mass(model, L, h, ctx, positions, source_layer, dev):
    """Sum over the last `positions` query positions of head h's pattern row sums (squared attention is unnormalised)."""
    values, init, first = ctx; blocks = model.transformer.h
    with torch.no_grad():
        x = values; v1 = first
        for l in range(source_layer, L):
            x, v1 = blocks[l](x, v1, init)
        b = blocks[L]; xin = b.lambdas[0] * x + b.lambdas[1] * init; n = F.rms_norm(xin, (xin.shape[-1],))
        at = b.attn; B_, T, C = n.shape; H, D = at.n_head, at.head_dim
        q = at.c_q(n).view(B_, T, H, D); k = at.c_k(n).view(B_, T, H, D); q2 = at.c_q2(n).view(B_, T, H, D); k2 = at.c_k2(n).view(B_, T, H, D)
        from circuit_checks.heads import _rot
        cos, sin = at.rotary(q); q, k = _rot(F.rms_norm(q, (D,)), cos, sin), _rot(F.rms_norm(k, (D,)), cos, sin); q2, k2 = _rot(F.rms_norm(q2, (D,)), cos, sin), _rot(F.rms_norm(k2, (D,)), cos, sin)
        s1 = torch.einsum("bqhd,bkhd->bhqk", q, k)[0, h]; s2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2)[0, h]
        p = (s1 / D) * (s2 / D); p = torch.tril(p)
        return float(p[-positions:].abs().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--contexts", type=int, default=8); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--c1a", default="results/circuits_c1a.json"); ap.add_argument("--n-heads", type=int, default=6); ap.add_argument("--n-units", type=int, default=600)
    ap.add_argument("--n-tokens", type=int, default=16); ap.add_argument("--ridge", type=float, default=1e-6); ap.add_argument("--out", default="results/circuits_c1b.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.contexts = 2; a.sequence_length = 8; a.chunk = 8; a.n_heads = 2; a.n_units = 20; a.n_tokens = 2
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); readers = F.normalize(torch.randn(cfg.n_embd, 2), dim=0)
        held_rows = torch.randint(0, 64, (2, a.sequence_length), generator=torch.Generator().manual_seed(20)); toks = [3, 5]
        heads = [(1, 0), (2, 1)]; units = [(1, i) for i in range(10)] + [(3, i) for i in range(10)]; c1a_heads = {}
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device)
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu"); held_rows = rows[96:96 + a.contexts, :a.sequence_length].long()
        c1a = json.load(open(a.c1a)); toks = c1a["tokens"][:a.n_tokens]
        hs = collections.Counter()
        for p in c1a["per_direction"]:
            for h, v in p["head_share"].items(): hs[h] += v
        heads = [tuple(int(z) for z in h.split(".")) for h, _ in hs.most_common(a.n_heads)]
        uc = collections.Counter(u for p in c1a["per_direction"] for u in p["stable_units"]); units = [tuple(int(z) for z in u.split(".")) for u, _ in uc.most_common(a.n_units)]
        c1a_heads = {p["name"]: p["top3_heads"] for p in c1a["per_direction"]}
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer
    W = model.lm_head.weight.float(); tokdirs = F.normalize(W[toks].T, dim=0)
    U = torch.cat([readers.to(dev), tokdirs.to(dev)], 1); names = [f"reader{k}" for k in range(readers.shape[1])] + [f"tok{t}" for t in toks]
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True); layers = spans.all_layers
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(held_rows[i:i + 1]) for i in range(len(held_rows))]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        spans.make_span(ctxs[0])(torch.zeros(d, device=dev), None)
    # ---- dictionary ---------------------------------------------------------------------------------------------------------------------
    transports = [("I", torch.eye(d, device=dev))] + [(f"{L}.{h}", head_transport(model.transformer.h[L].attn, h).to(dev)) for (L, h) in heads]
    X, Y, labels = [], [], []
    for (L, i) in units:
        mlp = model.transformer.h[L].mlp; a_ = mlp.Left.weight.float()[i].to(dev); b_ = mlp.Right.weight.float()[i].to(dev)
        for ti, (tn, T) in enumerate(transports):
            for tj, (tn2, T2) in enumerate(transports):
                if tj < ti: continue
                X.append(T.T @ a_); Y.append(T2.T @ b_); labels.append((f"{L}.{i}", tn, tn2))
                if tj != ti:
                    X.append(T2.T @ a_); Y.append(T.T @ b_); labels.append((f"{L}.{i}", tn2, tn))
    dic = RankOneDictionary(torch.stack(X).double(), torch.stack(Y).double(), labels); G = dic.gram(); idx_all = torch.arange(dic.n, device=dev)
    sub = {"identity_only": [i for i, l in enumerate(labels) if l[1] == "I" and l[2] == "I"], "heads_only": [i for i, l in enumerate(labels) if l[1] != "I" and l[2] != "I"]}
    print(f"[dictionary] {dic.n} forms: {len(units)} units x {len(transports)} transports ({[t for t, _ in transports]}); Gram cond ~{float(G.diagonal().max() / G.diagonal().min()):.1e}", flush=True)
    # ---- controls -------------------------------------------------------------------------------------------------------------------
    g = torch.Generator().manual_seed(0); Bt = torch.randn(d, d, generator=g, dtype=torch.float64); Bt = (Bt + Bt.T).to(dev); bt = dic.inner(Bt)
    picks = torch.randint(0, dic.n, (6,), generator=g).tolist(); mats = {p: dic.materialise(p) for p in picks}
    short = max(abs(float((mats[p] * Bt).sum()) - float(bt[p])) / max(abs(float((mats[p] * Bt).sum())), 1e-30) for p in picks)
    gerr = max(abs(float((mats[picks[i]] * mats[picks[(i + 1) % 6]]).sum()) - float(G[picks[i], picks[(i + 1) % 6]])) / max(abs(float((mats[picks[i]] * mats[picks[(i + 1) % 6]]).sum())), 1e-30) for i in range(6))
    planted = 1.5 * mats[picks[0]] - 0.7 * mats[picks[1]] + 2.2 * mats[picks[2]]; _, ep = solve(G, dic.inner(planted), idx_all, a.ridge); r2p = ep / float(planted.square().sum())
    noise = []
    # physics control: a unit's own curvature through the identity transport (theta added at the unit's block input) is captured by its own I,I form
    (Lp, ip) = units[0]; mlp = model.transformer.h[Lp].mlp
    with sdpa_kernel(SDPBackend.MATH):
        vals, init, first = ctxs[0]
        def unit_curv(th):
            x = vals; v1 = first
            for l in range(a.source_layer, Lp):
                x, v1 = model.transformer.h[l](x, v1, init)
            b = model.transformer.h[Lp]; xin = b.lambdas[0] * x + b.lambdas[1] * init
            at_, v1b = b.attn(F.rms_norm(xin, (d,)), v1); xm = xin + at_ + th
            n = F.rms_norm(xm, (d,)); h = (mlp.Left(n)[..., ip] * mlp.Right(n)[..., ip])
            return h[:, -3:].mean(1)[0][None]
        with torch.no_grad(): unit_curv(torch.zeros(d, device=dev))
        Hu = full_hessian(unit_curv, d, dev, a.chunk)[0].double(); Bu = 0.5 * (Hu + Hu.T)
        own = [i for i, l in enumerate(labels) if l[0] == f"{Lp}.{ip}" and l[1] == "I" and l[2] == "I"]
        _, eo = solve(G, dic.inner(Bu), torch.tensor(own, device=dev), a.ridge); physics = eo / float(Bu.square().sum())
    print(f"[controls] shortcut {short:.2e} gram {gerr:.2e} planted R2 {r2p:.6f} | physics (unit {Lp}.{ip} own curvature via I,I form) R2 {physics:.3f}", flush=True)
    # ---- fits ---------------------------------------------------------------------------------------------------------------------------
    rec = []; t0 = time.time(); K = U.shape[1]
    head_forms = {tn: [i for i, l in enumerate(labels) if tn in (l[1], l[2])] for tn, _ in transports if tn != "I"}
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            span = spans.make_span(ctx); Hc = full_hessian(lambda th: U.T @ span(th, None)[0], d, dev, a.chunk).double()
            masses = {tn: attention_mass(model, *[int(z) for z in tn.split(".")], ctx, 3, a.source_layer, dev) for tn, _ in transports if tn != "I"}
            for k in range(K):
                B = 0.5 * (Hc[k] + Hc[k].T); nB = float(B.square().sum()); b = dic.inner(B)
                al, e_all = solve(G, b, idx_all, a.ridge); r2 = e_all / nB
                r = dict(context=c, direction=k, name=names[k], norm2=nB, r2=r2, masses=masses)
                for sn, si in sub.items():
                    _, es = solve(G, b, torch.tensor(si, device=dev), a.ridge); r[f"r2_{sn}"] = es / nB
                w = al.abs() * G.diagonal().sqrt(); r["head_coef_mass"] = {tn: float(w[idx].sum()) for tn, idx in head_forms.items()}
                unit_mass = collections.Counter()
                for i in range(dic.n): unit_mass[labels[i][0]] += float(w[i])
                r["top_units"] = [u for u, _ in unit_mass.most_common(10)]; rec.append(r)
                if c == 0 and k == 0:
                    for _ in range(8 if not a.smoke else 3):
                        N_ = torch.randn(d, d, generator=g, dtype=torch.float64); N_ = (N_ + N_.T).to(dev); N_ = N_ * (nB ** 0.5 / N_.norm())
                        _, en = solve(G, dic.inner(N_), idx_all, a.ridge); noise.append(en / float(N_.square().sum()))
            print(f"[fit] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): R2 median {np.median([r['r2'] for r in rec]):.3f} identity-only {np.median([r['r2_identity_only'] for r in rec]):.3f} heads-only {np.median([r['r2_heads_only'] for r in rec]):.3f} | noise {np.median(noise):.4f}", flush=True)
    # ---- coefficient vs attention-mass correlation per direction, top-3 heads ---------------------------------------------------------------
    corrs = []
    for k in range(K):
        R = [r for r in rec if r["direction"] == k]
        for tn in list(head_forms)[:3]:
            cm = np.array([r["head_coef_mass"][tn] for r in R]); am = np.array([r["masses"][tn] for r in R])
            if len(R) > 2 and cm.std() > 0 and am.std() > 0: corrs.append(float(np.corrcoef(cm, am)[0, 1]))
    r2s = np.array([r["r2"] for r in rec]); ratio_t = np.array([r["r2"] / max(r["r2_identity_only"], 1e-12) for r in rec])
    r2_readers = float(np.median([r["r2"] for r in rec if r["name"].startswith("reader")])); r2_tokens = float(np.median([r["r2"] for r in rec if r["name"].startswith("tok")]))
    preds = dict(pred_a_instrument=short <= BARS["shortcut"] and gerr <= BARS["shortcut"] and r2p >= BARS["planted"] and physics >= BARS["physics"],
                 pred_b_captured=float(np.median(r2s)) >= BARS["captured"], pred_c_transport_matters=float(np.median(ratio_t)) >= BARS["transport"],
                 pred_d_coefficients_track_attention=(float(np.median(corrs)) if corrs else float("nan")) >= BARS["corr"], pred_e_tokens=r2_tokens >= BARS["tokens"] * r2_readers)
    res = dict(args=vars(a), predictions=preds, controls=dict(shortcut=short, gram=gerr, planted_r2=r2p, physics_r2=physics, noise_r2=noise), transports=[t for t, _ in transports], n_units=len(units), n_forms=dic.n,
               summary=dict(median_r2=float(np.median(r2s)), median_r2_identity_only=float(np.median([r["r2_identity_only"] for r in rec])), median_r2_heads_only=float(np.median([r["r2_heads_only"] for r in rec])),
                            median_transport_ratio=float(np.median(ratio_t)), median_corr_coef_vs_attention=(float(np.median(corrs)) if corrs else None), r2_readers=r2_readers, r2_tokens=r2_tokens,
                            top_units_overall=collections.Counter(u for r in rec for u in r["top_units"]).most_common(20)),
               records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls={k: v for k, v in res["controls"].items() if k != "noise_r2"}, summary={k: v for k, v in res["summary"].items() if k != "top_units_overall"}), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
