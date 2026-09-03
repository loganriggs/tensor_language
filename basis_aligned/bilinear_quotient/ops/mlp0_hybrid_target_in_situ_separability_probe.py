#!/usr/bin/env python3
"""R536 hybrid targets under the REAL block-0 context: how separable are they in situ? (CPU probe, 0 full forwards)

# BQGATE: EXPERIMENT
# pred_a_instrument
# pred_b_token_stream_dominates_mlp0_input
# pred_c_token_target_near_separable_in_situ
# pred_d_context_target_near_separable_in_situ

Exact block-0 attention pass (tt_model semantics) on the frozen copy-induction v2 row caches gives MLP0's real input
xhat = s (p + q); hybrid pairs are built on it and the general Wiener / reduced-rank bounds are computed in W_D metric.
LOWER residual = more separable. Preregistration: polynomial_causal/MLP0_HYBRID_TARGET_IN_SITU_SEPARABILITY_PROBE_PREREGISTRATION.md
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "MLP0_HYBRID_TARGET_IN_SITU_SEPARABILITY_PROBE_PREREGISTRATION.md"
BLOB = Path("/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/"
            "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin")
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
CODE = ROOT / ".rowcache_terminal_copy_induction_v2/ood_code.pt"
OUT = ROOT / "mlp0_hybrid_target_in_situ_separability_probe_results.json"
HASHES = {
    PREREG: "86a630f78e6e82b0d01ff63dc00f80fc8b0cffef99f8ddfb3a5438ed6f58a8d8",
    BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1",
    CODE: "6cf514e75dfd03399f223a9ba5f6ebe5f4b1315bcb839a515e1c19e7b5474bd9",
}
RUNG = "mlp0_hybrid_target_in_situ_separability_probe"
D, NH, HD, T = 1152, 9, 128, 257
POS = list(range(1, 256, 2))          # 128 sampled positions per doc
KS = [3, 8, 32, 128, 512, 1152]
BUCKETS = {"1-4": (1, 4), "5-24": (5, 24), "25-124": (25, 124), "125-255": (125, 255)}
CHUNK = 2048
SEED = 20260903
torch.set_num_threads(max(1, os.cpu_count() or 1))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_hashes():
    for p, e in HASHES.items():
        if not p.is_file() or sha256(p) != e:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def load_weights():
    sd = torch.load(BLOB, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    g = lambda k: sd[k].float()
    W = {"wte": g("transformer.wte.weight"), "lambdas": g("transformer.h.0.lambdas"), "lamb": g("transformer.h.0.attn.lamb"),
         "L": g("transformer.h.0.mlp.Left.weight"), "R": g("transformer.h.0.mlp.Right.weight"), "Dn": g("transformer.h.0.mlp.Down.weight")}
    for n in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
        W[n] = g(f"transformer.h.0.attn.{n}.weight")
    return W, sd


def rope(Tn):
    inv = 1.0 / (10000 ** (torch.arange(0, HD, 2).float() / HD))
    fr = torch.outer(torch.arange(Tn).float(), inv)
    return fr.cos().bfloat16().float()[None, :, None, :], fr.sin().bfloat16().float()[None, :, None, :]


def rot(x, c, s):
    d = x.shape[-1] // 2
    return torch.cat([x[..., :d] * c + x[..., d:] * s, -x[..., :d] * s + x[..., d:] * c], -1)


@torch.no_grad()
def attn0(W, x0):
    """x0: (B,T,D) rms-normalised embeddings. Returns block-0 attention output q (B,T,D), tt_model semantics."""
    B, Tn, _ = x0.shape
    cos, sin = rope(Tn)
    def proj(w):
        return F.rms_norm((x0 @ w.T).view(B, Tn, NH, HD), (HD,))
    q1, k1, q2, k2 = (rot(proj(W[n]), cos, sin) for n in ("c_q", "c_k", "c_q2", "c_k2"))
    v = (x0 @ W["c_v"].T).view(B, Tn, NH, HD)
    v = (1 - W["lamb"]) * v + W["lamb"] * v            # v1 == v in block 0
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
    pat = (torch.einsum("bqhd,bkhd->bhqk", q1, k1) / HD) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HD)
    pat = pat.masked_fill(~mask, 0.0)
    y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D)
    return y @ W["c_proj"].T


@torch.no_grad()
def module_attn0(sd, x0):
    sys.path.insert(0, "/workspace/tensor_language")
    import jacclust.tt_model as TT
    cfg = TT.GPTConfig(vocab_size=50304, n_layer=1, n_head=NH, n_embd=D, bilinear=True, squared_attn=True, bilinear_attn=True)
    blk = TT.Block(cfg).float().eval()
    st = {k[len("transformer.h.0."):]: v.float() for k, v in sd.items() if k.startswith("transformer.h.0.")}
    blk.load_state_dict(st, strict=True)
    y, _ = blk.attn(x0, None)
    return y


@torch.no_grad()
def block0_states(W, rows):
    """Per doc: p (B,T,D) token stream entering the attention residual add, q (B,T,D) attention output."""
    lam = float(W["lambdas"][0] + W["lambdas"][1])
    P, Q = [], []
    for i in range(0, rows.shape[0], 24):
        idx = rows[i:i + 24, :T]
        x0 = F.rms_norm(W["wte"][idx], (D,))
        P.append(lam * x0); Q.append(attn0(W, x0))
    return torch.cat(P), torch.cat(Q)


def unigram(rows):
    return torch.bincount(rows.reshape(-1), minlength=50304).float()


class Cov:
    """Accumulates centred covariance blocks S_tt, C_to, C_oo in float64 from streamed (tgt, obs) rows."""
    def __init__(self, n):
        self.n = n
        self.Stt = torch.zeros(n, n, dtype=torch.float64); self.Cto = torch.zeros(n, n, dtype=torch.float64)
        self.Coo = torch.zeros(n, n, dtype=torch.float64)
        self.mt = torch.zeros(n, dtype=torch.float64); self.mo = torch.zeros(n, dtype=torch.float64); self.cnt = 0
    def add(self, t, o):
        t = t.double(); o = o.double()
        self.Stt += t.T @ t; self.Cto += t.T @ o; self.Coo += o.T @ o
        self.mt += t.sum(0); self.mo += o.sum(0); self.cnt += t.shape[0]
    def finish(self):
        n = self.cnt; mt = self.mt / n; mo = self.mo / n
        return (self.Stt / n - torch.outer(mt, mt), self.Cto / n - torch.outer(mt, mo), self.Coo / n - torch.outer(mo, mo))


def wiener_general(Stt, Cto, Coo, Dn):
    Dn = Dn.double()
    tr = lambda M: float(torch.trace(Dn @ M @ Dn.T))
    e_t = tr(Stt)
    ridge = 1e-8 * float(torch.trace(Coo)) / Coo.shape[0]
    Coo_r = Coo + ridge * torch.eye(Coo.shape[0], dtype=torch.float64)
    X = torch.linalg.solve(Coo_r, Cto.T)                       # Coo^{-1} C_ot
    res_full = (e_t - tr(Cto @ X)) / e_t
    ev, U = torch.linalg.eigh(Coo_r)
    ev = ev.clamp_min(ridge)
    Coo_mh = (U * ev.rsqrt()) @ U.T
    sv = torch.linalg.svdvals(Dn @ Cto @ Coo_mh)
    cs = torch.cumsum(sv ** 2, 0)
    ladder = {str(k): float((e_t - cs[min(k, len(cs)) - 1]) / e_t) for k in KS}
    ev_t = torch.linalg.eigvalsh(Dn @ Stt @ Dn.T).clamp_min(0)
    ev_t = ev_t.flip(0); ct = torch.cumsum(ev_t, 0)
    pure = {str(k): float(1 - ct[min(k, len(ct)) - 1] / ct[-1]) for k in KS}
    pr = ev_t / ev_t.sum(); eff = float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))
    psd = {"Stt": float(torch.linalg.eigvalsh(Stt).min() / torch.linalg.eigvalsh(Stt).max()),
           "Coo": float(torch.linalg.eigvalsh(Coo).min() / torch.linalg.eigvalsh(Coo).max())}
    return {"residual_any_rank": res_full, "rank_ladder": ladder, "pure_target_ladder": pure,
            "pure_target_eff_rank": eff, "target_energy_output_metric": e_t, "psd_min_over_max": psd}


@torch.no_grad()
def hybrid_stats(W, p, q, rows, gen, corpus):
    """p,q: (B,T,D). Returns per-target covariance blocks, leak fractions, decomposition identity error, rho stats."""
    L, R = W["L"], W["R"]
    B = p.shape[0]
    pos = torch.tensor(POS)
    ps = p[:, pos].reshape(-1, D); qs = q[:, pos].reshape(-1, D)        # (N,D), N = B*128
    N = ps.shape[0]
    rho = qs.norm(dim=1) / ps.norm(dim=1)
    posid = pos.repeat(B)
    rho_stats = {"median": float(rho.median()), "q25": float(rho.quantile(0.25)), "q75": float(rho.quantile(0.75)),
                 "mean": float(rho.mean()),
                 "by_bucket_median": {k: float(rho[(posid >= a) & (posid <= b)].median()) for k, (a, b) in BUCKETS.items()}}
    # replacement tokens ~ corpus unigram; donor contexts = random other sampled positions
    uni = unigram(rows)
    lam = float(W["lambdas"][0] + W["lambdas"][1])
    rep_tok = torch.multinomial(uni, N, replacement=True, generator=gen)
    p_rep = lam * F.rms_norm(W["wte"][rep_tok], (D,))
    perm = torch.randperm(N, generator=gen)
    perm = torch.where(perm == torch.arange(N), (perm + 1) % N, perm)
    q_don = qs[perm]

    def gT(a): return (a @ L.T) * (a @ R.T)
    def gI(a, b): return (a @ L.T) * (b @ R.T) + (b @ L.T) * (a @ R.T)
    def s_of(x): return 1.0 / x.pow(2).mean(1, keepdim=True).sqrt()

    cov_tok, cov_ctx = Cov(4608), Cov(4608)
    leak_tok = leak_ctx = 0.0; en_tok = en_ctx = 0.0; ident_err = None
    for i in range(0, N, CHUNK):
        P_, Q_, Pr, Qd = ps[i:i + CHUNK], qs[i:i + CHUNK], p_rep[i:i + CHUNK], q_don[i:i + CHUNK]
        s = s_of(P_ + Q_); s2 = s ** 2
        g_base = gT(s * (P_ + Q_))
        gT_b, gI_b, gC_b = gT(P_), gI(P_, Q_), gT(Q_)
        if ident_err is None:
            recon = s2 * (gT_b + gI_b + gC_b)
            ident_err = float((g_base - recon).norm() / g_base.norm())
        # TOKEN target: swap token, hold context
        s_t = s_of(Pr + Q_); s2t = s_t ** 2
        D_obs = gT(s_t * (Pr + Q_)) - g_base
        D_tgt = s2t * gT(Pr) - s2 * gT_b
        cov_tok.add(D_tgt, D_obs)
        leak = (s2t - s2) * gC_b
        leak_tok += float(leak.pow(2).sum()); en_tok += float(D_obs.pow(2).sum())
        # CONTEXT target: swap context, hold token
        s_c = s_of(P_ + Qd); s2c = s_c ** 2
        D_obs = gT(s_c * (P_ + Qd)) - g_base
        D_tgt = s2c * gI(P_, Qd) - s2 * gI_b
        cov_ctx.add(D_tgt, D_obs)
        leak = (s2c - s2) * gT_b
        leak_ctx += float(leak.pow(2).sum()); en_ctx += float(D_obs.pow(2).sum())
    return {"n_samples": N, "rho": rho_stats, "decomposition_identity_rel_err": ident_err,
            "rms_leak_energy_fraction": {"token": leak_tok / en_tok, "context": leak_ctx / en_ctx},
            "cov_tok": cov_tok.finish(), "cov_ctx": cov_ctx.finish()}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    check_hashes()
    W, sd = load_weights()
    # instrument (ii): manual attention vs the model's own module on 4 natural docs
    nat = torch.load(NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat)[:, :T].long()
    code = torch.load(CODE, map_location="cpu"); code = (code["rows"] if isinstance(code, dict) else code)[:, :T].long()
    x0 = F.rms_norm(W["wte"][nat[:4]], (D,))
    y_manual = attn0(W, x0); y_module = module_attn0(sd, x0)
    attn_rel_err = float((y_manual - y_module).abs().max() / y_module.pow(2).mean().sqrt())
    gen = torch.Generator().manual_seed(SEED)
    per = {}
    for name, rows in (("natural", nat), ("code", code)):
        p, q = block0_states(W, rows)
        st = hybrid_stats(W, p, q, rows, gen, name)
        per[name] = {"n_samples": st["n_samples"], "rho": st["rho"], "decomposition_identity_rel_err": st["decomposition_identity_rel_err"],
                     "rms_leak_energy_fraction": st["rms_leak_energy_fraction"],
                     "token_target": wiener_general(*st["cov_tok"], W["Dn"]), "context_target": wiener_general(*st["cov_ctx"], W["Dn"])}
        del p, q, st
    nat_r = per["natural"]
    psd_ok = all(per[c][t]["psd_min_over_max"][m] >= -1e-8 for c in per for t in ("token_target", "context_target") for m in ("Stt", "Coo"))
    pred_a = bool(attn_rel_err <= 1e-3 and all(per[c]["decomposition_identity_rel_err"] <= 1e-4 for c in per) and psd_ok
                  and all(per[c]["n_samples"] >= 20000 for c in per))
    pred_b = bool(pred_a and nat_r["rho"]["median"] <= 0.5)
    pred_c = bool(pred_a and nat_r["token_target"]["residual_any_rank"] <= 0.15)
    pred_d = bool(pred_a and nat_r["context_target"]["residual_any_rank"] <= 0.15)
    strong_null = bool(not (pred_a and pred_b and pred_c and pred_d))
    if not pred_a:
        verdict = "instrument_invalid"
    elif pred_b and pred_c and pred_d:
        verdict = "in_situ_targets_near_separable_rho_scan_rho1_rows_retired_as_operative_reference"
    elif not pred_b:
        verdict = "context_not_small_relative_to_token_stream_rho1_rows_operative"
    else:
        verdict = "small_rho_but_anisotropic_context_defeats_isotropic_scan_instrument_correction_to_2686_2687"
    result = {
        "status": "complete", "rung": RUNG, "owner_lane": "claude_parallel_probe",
        "claim_level": "exact_block0_in_situ_linear_regime_bound_no_circuit_claim",
        "source_hashes": {str(k): v for k, v in HASHES.items()},
        "attention0_manual_vs_module_max_abs_over_rms": attn_rel_err, "sampled_positions": POS[:3] + ["..."] + POS[-2:],
        "per_corpus": per,
        "bars": {"attn_tol": 1e-3, "identity_tol": 1e-4, "min_samples": 20000, "rho_median_max": 0.5, "residual_max": 0.15,
                 "nulls": {"rho_median": 1.0, "residual": 0.30}},
        'pred_a_instrument': pred_a,
        'pred_b_token_stream_dominates_mlp0_input': pred_b,
        'pred_c_token_target_near_separable_in_situ': pred_c,
        'pred_d_context_target_near_separable_in_situ': pred_d,
        "strong_null": strong_null, "verdict": verdict,
        "execution_price": {"full_model_forwards": 0, "block0_attention_passes_docs": int(nat.shape[0] + code.shape[0]),
                            "backwards": 0, "deployed_parameters": 0, "cpu_only": True},
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    brief = {c: {"rho_median": per[c]["rho"]["median"], "tok": per[c]["token_target"]["residual_any_rank"],
                 "ctx": per[c]["context_target"]["residual_any_rank"]} for c in per}
    print(json.dumps({"verdict": verdict, "strong_null": strong_null, "attn_err": attn_rel_err, "brief": brief,
                      **{k: v for k, v in result.items() if k.startswith("pred_")}, "runtime_s": result["runtime_s"]}, indent=1))


if __name__ == "__main__":
    main()
