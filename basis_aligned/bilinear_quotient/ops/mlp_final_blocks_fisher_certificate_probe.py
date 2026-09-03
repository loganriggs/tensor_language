#!/usr/bin/env python
"""mlp_final_blocks_fisher_certificate_probe -- why variance-rank != function-rank for MLP16/17's writes (§2694): the RMSNorm
scale gauge (radial writes are inert), a second-order Fisher certificate that should reproduce §2694's CE ladder without
forwards, the Fisher-whitened (loss-optimal) rank-k basis, and whether the eight rank-8 quadratic forms share an eigenbasis.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_radial_gauge pred_c_certificate_mlp17 pred_d_fisher_basis_mlp17_k8 pred_e_no_shared_dictionary

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96-191 -- LOWER IS BETTER.
Preregistration: polynomial_causal/MLP_FINAL_BLOCKS_FISHER_CERTIFICATE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/mlp_final_blocks_fisher_certificate_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
import mlp_final_blocks_low_rank_surrogate_probe as S
import mlp_final_blocks_quadratic_form_rank_probe as Q

ROOT = R.ROOT
PREREG = R.POLY / "MLP_FINAL_BLOCKS_FISHER_CERTIFICATE_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "mlp_final_blocks_low_rank_surrogate_probe_results.json"
QSCRIPT = SELF.parent / "mlp_final_blocks_quadratic_form_rank_probe.py"
OUT = ROOT / "mlp_final_blocks_fisher_certificate_probe_results.json"
HASHES = {PREREG: "9c3e09c85e3debf27ff953008897d91ef59d966c295f9b6e6653fe632d5dc7a3", PRIOR: "8a88b71455c8087ca5de84cfa83e71df35080236932b4ccbd466ce0bc089326e",
          QSCRIPT: "0cab37eb14e5c2bf54f1b1b5cca35887549cde68ba602cb2dd9b613b325e6870",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "mlp_final_blocks_fisher_certificate_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI, HALF, CH = S.TI, S.HALF, S.CH
BLOCKS = [16, 17]
KS = [0, 1, 2, 4, 8, 16, 32, 64]
CERT_KS = [4, 8, 16, 32, 64]
S_EVAL, S_FIT = 4, 2
BARS = {"ce_tol": 1e-4, "radial_min": 0.5, "ratio_lo": 0.5, "ratio_hi": 2.0, "fisher17_k8_max": 0.05, "diag_max": 0.2}
NULLS = {"radial_max": 0.2, "ratio_lo": 0.25, "ratio_hi": 4.0, "fisher17_k8_min": 0.075, "diag_min": 0.5}


def check_hashes():
    for p, h in HASHES.items():
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def forward(m, idx, hook=None, grad_blocks=()):
    """Manual forward; hook(l, mw, x_pre_write) -> mw at every block.  Writes of grad_blocks become autograd leaves (returned)."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = R.rope(Tn)
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
    leaves = {}
    for l, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; h = F.rms_norm(x, (D,))
        def pr(lin):
            return R.rot(F.rms_norm(lin(h).view(B, Tn, NH, HD), (HD,)), cos, sin)
        v = a.c_v(h).view(B, Tn, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        pat = (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q), pr(a.c_k)) / HD) * (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q2), pr(a.c_k2)) / HD)
        pat = pat.masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        if l in grad_blocks:
            mw = mw.detach().requires_grad_(True); leaves[l] = mw
        if hook is not None:
            mw = hook(l, mw, x)
        x = x + mw + mlp.Down_bias
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30), leaves


@torch.no_grad()
def ce_of(m, rows, hook=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        lg, _ = forward(m, idx, hook)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


def scores(m, idx, tgt, gen, n_samples):
    """Returns per-block dict: g_true [N, D], samples [S, N, D], x_pre_write [N, D], mw [N, D]."""
    store = {}
    def hook(l, mw, x):
        store[l] = x.detach().reshape(-1, D)
        return mw
    with torch.enable_grad():
        lg, leaves = forward(m, idx, hook, grad_blocks=tuple(BLOCKS))
        lp = F.log_softmax(lg.float(), -1)
        loss_true = -lp.reshape(-1, V).gather(1, tgt.reshape(-1, 1)).sum()
        g_true = torch.autograd.grad(loss_true, [leaves[l] for l in BLOCKS], retain_graph=True)
        samp = []
        p = lp.exp().reshape(-1, V)
        for _ in range(n_samples):
            y = torch.multinomial(p.detach(), 1, generator=gen).squeeze(1)
            ll = lp.reshape(-1, V).gather(1, y[:, None]).sum()
            samp.append(torch.autograd.grad(ll, [leaves[l] for l in BLOCKS], retain_graph=True))
    out = {}
    for bi, l in enumerate(BLOCKS):
        out[l] = {"g": g_true[bi].reshape(-1, D).detach(), "s": torch.stack([s[bi].reshape(-1, D).detach() for s in samp]),
                  "x": store[l], "mw": leaves[l].detach().reshape(-1, D)}
    return out


def fisher_basis(G, C, mu, k):
    G = G.double(); C = C.double()
    e, Vg = torch.linalg.eigh(G); eps = 1e-3 * float(e.sum()) / D; e = e.clamp_min(0) + eps
    Gh = (Vg * e.sqrt()) @ Vg.T; Ghi = (Vg / e.sqrt()) @ Vg.T
    M = Gh @ C @ Gh; M = 0.5 * (M + M.T)
    ev, Vm = torch.linalg.eigh(M); Vm = Vm.flip(1); ev = ev.flip(0)
    Vk = Vm[:, :k]
    Pi = Ghi @ Vk @ Vk.T @ Gh                                  # oblique G-orthogonal projector (float64)
    return Pi.float(), ev.float()


def make_fisher_patch(bases, Pis):
    def hook(l, mw, x):
        if l not in Pis:
            return mw
        mu = bases[l]["mu"]
        return (mu + ((mw - mu).reshape(-1, D) @ Pis[l].T).reshape(mw.shape))
    return hook


def spec_eff_rank(e):
    e = e.clamp_min(0); p = e / e.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (6, 257), generator=g); h0, h1 = nat[:3], nat[3:]
        prior = {"ladder": {str(l): {str(k): float("nan") for k in KS} for l in BLOCKS}, "baseline": float("nan")}
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        h0, h1 = nat[:HALF], nat[HALF:2 * HALF]
        pj = json.load(open(PRIOR)); prior = {"ladder": pj["ce_added_ladder"], "baseline": pj["baseline_ce"]["natural_h1"]}
    m = R.load_model()
    bases = S.fit_bases(m, h0)
    C = {l: (bases[l]["U"].double() * bases[l]["ev"].double().clamp_min(0)) @ bases[l]["U"].double().T for l in BLOCKS}
    # ---- Fisher metric on the FIT half (S_FIT samples/position)
    G = {l: torch.zeros(D, D, dtype=torch.float64) for l in BLOCKS}; nG = 0
    for i in range(0, h0.shape[0], CH):
        sc = scores(m, h0[i:i + CH, :TI], h0[i:i + CH, 1:TI + 1], gen, S_FIT)
        for l in BLOCKS:
            s = sc[l]["s"].reshape(-1, D).double(); G[l] += s.T @ s
        nG += sc[BLOCKS[0]]["s"].shape[0] * sc[BLOCKS[0]]["s"].shape[1]
        print(json.dumps({"stage": "fisher_fit", "docs": i + CH}), flush=True)
    G = {l: (G[l] / nG).float() for l in BLOCKS}
    g_spec = {str(l): spec_eff_rank(torch.linalg.eigvalsh(G[l])) for l in BLOCKS}
    # ---- eval half: certificate terms, radial fractions
    cert1 = {l: {k: 0.0 for k in KS} for l in BLOCKS}; cert2 = {l: {k: 0.0 for k in KS} for l in BLOCKS}
    radial = {l: 0.0 for l in range(NL)}; radial_u1 = {l: 0.0 for l in BLOCKS}; n_pos = 0
    def rad_hook(l, mw, x):
        x18 = (x + mw + m.transformer.h[l].mlp.Down_bias).reshape(-1, D); w = mw.reshape(-1, D)
        radial[l] += float((((w * x18).sum(1)) ** 2 / (w.norm(dim=1) ** 2 * x18.norm(dim=1) ** 2 + 1e-30)).sum())
        if l in BLOCKS:
            radial_u1[l] += float(((x18 @ bases[l]["U"][:, 0]) ** 2 / (x18.norm(dim=1) ** 2)).sum())
        return mw
    for i in range(0, h1.shape[0], CH):
        idx = h1[i:i + CH, :TI]; tgt = h1[i:i + CH, 1:TI + 1]
        sc = scores(m, idx, tgt, gen, S_EVAL)
        with torch.no_grad():
            forward(m, idx, rad_hook)
            n_pos += sc[BLOCKS[0]]["g"].shape[0]
            for l in BLOCKS:
                dev = sc[l]["mw"] - bases[l]["mu"]; U = bases[l]["U"]
                for k in KS:
                    delta = dev if k == 0 else dev - (dev @ U[:, :k]) @ U[:, :k].T
                    cert1[l][k] += float((sc[l]["g"] * delta).sum())
                    cert2[l][k] += float(0.5 * (torch.einsum("snd,nd->sn", sc[l]["s"], delta) ** 2).mean(0).sum())
        print(json.dumps({"stage": "certificate", "docs": i + CH}), flush=True)
    pred = {str(l): {str(k): (cert1[l][k] + cert2[l][k]) / n_pos for k in KS} for l in BLOCKS}
    first_share = {str(l): {str(k): (cert1[l][k] / n_pos) / pred[str(l)][str(k)] if pred[str(l)][str(k)] else None for k in KS} for l in BLOCKS}
    ratio = {str(l): {str(k): (prior["ladder"][str(l)][str(k)] / pred[str(l)][str(k)] if pred[str(l)][str(k)] else None) for k in KS} for l in BLOCKS}
    radial = {str(l): radial[l] / n_pos for l in radial}; radial_u1 = {str(l): radial_u1[l] / n_pos for l in radial_u1}
    # ---- Fisher-whitened basis arms (CE ADDED on eval half)
    ce0 = ce_of(m, h1)
    fisher_arm = {}; whitened_spec = {}
    for l, k in [(17, 8), (16, 8), (17, 32)]:
        Pi, ev = fisher_basis(G[l], C[l], bases[l]["mu"], k)
        whitened_spec[str(l)] = {"eff_rank": spec_eff_rank(ev), "top8_share": float(ev[:8].clamp_min(0).sum() / ev.clamp_min(0).sum())}
        fisher_arm[f"{l}_k{k}"] = ce_of(m, h1, make_fisher_patch(bases, {l: Pi})) - ce0
        print(json.dumps({"stage": "fisher_arm", "block": l, "k": k, "ce_added": fisher_arm[f"{l}_k{k}"]}), flush=True)
    Pi_full, _ = fisher_basis(G[17], C[17], bases[17]["mu"], D)
    ident = ce_of(m, h1[:4], make_fisher_patch(bases, {17: Pi_full})) - ce_of(m, h1[:4])
    # ---- shared dictionary test (exact weights)
    diag = {}
    for l in BLOCKS:
        lam, vec, _ = Q.forms(m, l, bases[l]["U"][:, :8])
        Qs = [(vec[j] * lam[j]) @ vec[j].T for j in range(8)]
        Bm = sum(q @ q for q in Qs); _, Bv = torch.linalg.eigh(Bm)
        fr = [float(((Bv.T @ q @ Bv).diagonal() ** 2).sum() / (q ** 2).sum()) for q in Qs]
        diag[str(l)] = {"per_form": fr, "mean": sum(fr) / 8}
    inst = {"baseline_ce": ce0, "prior_baseline": prior["baseline"], "abs_diff": abs(ce0 - prior["baseline"]) if not smoke else 0.0, "identity_arm_abs": abs(ident)}
    r17 = {k: ratio["17"][str(k)] for k in CERT_KS}
    preds = {"pred_a_instrument": bool(inst["abs_diff"] <= BARS["ce_tol"] and inst["identity_arm_abs"] <= BARS["ce_tol"]),
             "pred_b_radial_gauge": bool(radial["17"] >= BARS["radial_min"]),
             "pred_c_certificate_mlp17": bool(not smoke and all(r is not None and BARS["ratio_lo"] <= r <= BARS["ratio_hi"] for r in r17.values())),
             "pred_d_fisher_basis_mlp17_k8": bool(fisher_arm["17_k8"] <= BARS["fisher17_k8_max"]),
             "pred_e_no_shared_dictionary": bool(all(diag[str(l)]["mean"] <= BARS["diag_max"] for l in BLOCKS))}
    nulls = {"b_null_radial_le_.2": bool(radial["17"] <= NULLS["radial_max"]),
             "c_null_any_ratio_outside_[.25,4]": bool(smoke or any(r is None or r < NULLS["ratio_lo"] or r > NULLS["ratio_hi"] for r in r17.values())),
             "d_null_fisher17_k8_ge_.075": bool(fisher_arm["17_k8"] >= NULLS["fisher17_k8_min"]),
             "e_null_diag_ge_.5_either": bool(any(diag[str(l)]["mean"] >= NULLS["diag_min"] for l in BLOCKS))}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke, "sign_convention": "CE ADDED above the real model on held-out docs 96-191; LOWER IS BETTER",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "instrument": inst, "n_eval_positions": n_pos,
           "certificate_pred": pred, "certificate_first_order_share": first_share, "measured_prior_ladder": prior["ladder"], "ratio_measured_over_pred": ratio,
           "radial_fraction_by_block": radial, "radial_fraction_u1": radial_u1, "fisher_metric_eff_rank": g_spec, "whitened_cov_spectrum": whitened_spec,
           "fisher_arm_ce_added": fisher_arm, "shared_dictionary_diag_fraction": diag,
           "price": {"gpu_forwards": 0, "cpu_doc_forward_equivalents": int(h0.shape[0]) * (2 + S_FIT) + int(h1.shape[0]) * (3 + S_EVAL) + int(h1.shape[0]) * 3 + 8,
                     "cpu_seconds": time.time() - started},
           "hashes": {str(p): h for p, h in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "certificate_pred", "radial_fraction_by_block", "fisher_arm_ce_added", "shared_dictionary_diag_fraction", "fisher_metric_eff_rank", "price")}, indent=1, default=str)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ratio17": r17, "radial17": radial["17"], "fisher_arm": fisher_arm, "diag": {l: diag[l]["mean"] for l in diag}, "price": out["price"]}, indent=1))


if __name__ == "__main__":
    main()
