#!/usr/bin/env python
"""late_pool_surrogate_probe -- one surrogate for the mlp11-15 pool (§2721): per-block linear maps of each block's own input applied in
sequence, the same plus the token, and a one-shot linear map of the block-11 input producing the SUM of the five writes. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_POOL_SURROGATE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_pool_surrogate_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_pool_surrogate_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_POOL_SURROGATE_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_mlp_subset_lattice_probe_results.json"   # §2721
OUT = ROOT / "late_pool_surrogate_probe_results.json"
HASHES = {PREREG: "d61399aea42b4e715f1e8b26aef69191ff7add9c62d06fdb75cb438cbbdc2b5a", PRIOR: "884a0bba4acbb92638b6b9112ef5f287784dbe1f1dcc08958bc1d97f4ff610b0",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_pool_surrogate_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_POOL = 0.724; LAM = 1e-2
POOL = [("mlp", l) for l in range(11, 16)]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.02, "b_min": 0.50, "c_mult": 0.80, "d_slack": 0.05, "e_r2_min": 0.50}
NULLS = {"b_max": 0.20, "c_mult": 0.50, "d_slack": 0.15, "e_r2_max": 0.25}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


class GAcc:
    """float64 second-moment accumulator on DEV."""
    def __init__(self):
        self.S = torch.zeros(D, D, dtype=torch.float64, device=DEV); self.mu = torch.zeros(D, dtype=torch.float64, device=DEV); self.cnt = 0
    def add(self, w):
        X = w.reshape(-1, D).double(); self.S += X.T @ X; self.mu += X.sum(0); self.cnt += X.shape[0]
    def cov(self):
        mu = self.mu / self.cnt; return mu, (self.S / self.cnt - torch.outer(mu, mu)).cpu()
    def finish(self):
        mu, C = self.cov()
        ev, U = torch.linalg.eigh(C); U = U.flip(1)
        return {"mu": mu.float(), "U": U.float().to(DEV), "spec": R.spectrum(C), "C": C}


def pooled_basis(bs):
    """One basis from the equal-weight mean of the per-site centred covariances (= PCA of the pooled per-site-centred writes)."""
    C = sum(b["C"] for b in bs) / len(bs)
    ev, U = torch.linalg.eigh(C); U = U.flip(1)
    return {"U": U.float().to(DEV), "spec": R.spectrum(C), "C": C}


def captured(b, U, k):
    """Fraction of site b's write variance inside span(U[:, :k])."""
    Uk = U[:, :k].cpu().double(); return float(torch.trace(Uk.T @ b["C"] @ Uk) / torch.trace(b["C"]))


@torch.no_grad()
def forward(m, idx, patch=None, collect=None):
    """tt_model semantics; patch: dict site -> fn(w) applied to that site's write; collect(site, w)."""
    B, Tn = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = (t.to(DEV) for t in R.rope(Tn))
    mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
    def apply(s, w, x):
        if collect is not None:
            collect(s, w)
        return patch[s](w, x) if patch and s in patch else w
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
        aw = a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
        x = x + apply(("attn", l), aw, x)
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        if collect is not None:
            collect(("mlpin", l), xhat)
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        x = x + apply(("mlp", l), mw, x) + mlp.Down_bias
    if collect is not None:
        collect(("final", -1), x)
    if collect is not None:
        collect(("final", -1), x)
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows, patch=None, hook=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
        if hook is not None:
            hook(idx)
        lg = forward(m, idx, patch)
        tot += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum")); n += tgt.numel()
    return tot / n


def fit(m, rows, sites):
    acc = {s: GAcc() for s in sites}
    for i in range(0, rows.shape[0], CH):
        forward(m, rows[i:i + CH, :TI], collect=lambda s, w: acc[s].add(w) if s in acc else None)
    return {s: acc[s].finish() for s in sites}


def pooled(bs, normalise):
    C = sum((b["C"] / torch.trace(b["C"]) if normalise else b["C"]) for b in bs) / len(bs)
    ev, U = torch.linalg.eigh(C); U = U.flip(1)
    return {"U": U.double(), "spec": R.spectrum(C)}


def emb(m, idx):
    return F.rms_norm(m.transformer.wte(idx), (D,))


IU = torch.triu_indices(KM, KM)


def feats(mode, e, xh, Um):
    Um = Um.to(xh.device, xh.dtype); IU_ = IU.to(xh.device)
    """Feature rows [N, F] for one chunk. e = token embedding rows, xh = rms-normed MLP input rows, Um = [D, KM] core basis."""
    if mode == "CUR":
        return e
    if mode == "FULLIN":
        return torch.cat([e, xh], 1)
    if mode == "IN":
        return xh
    if mode == "INTOK":
        return torch.cat([xh, e], 1)
    c = xh @ Um; q = (c[:, IU_[0]] * c[:, IU_[1]])
    return torch.cat([e, c, q], 1)          # COREIN: 1152 + 16 + 136


class Surrogate:
    """w -> mu + P (A phi) with phi built from the token (hook) and the pre-write residual x."""
    def __init__(self, m, mode, fit, Pm, Um):
        self.m, self.mode, self.Pm, self.Um = m, mode, (None if Pm is None else Pm.to(DEV).float()), Um.to(DEV).float()
        self.A = fit["A"].to(DEV).float(); self.mphi = fit["mphi"].to(DEV).float(); self.mu = fit["mu"].to(DEV).float(); self.e = None
    def hook(self, idx):
        self.e = emb(self.m, idx)
    def __call__(self, w, x):
        B, T, _ = w.shape
        phi = feats(self.mode, self.e.reshape(-1, D), F.rms_norm(x, (D,)).reshape(-1, D), self.Um)
        y = ((phi - self.mphi) @ self.A).view(B, T, D)
        if self.Pm is None:
            return self.mu + y
        return self.mu + (y @ self.Pm) @ self.Pm.T


def collect_pass(m, rows, sites):
    """Yields per chunk: idx, {site: write rows}, {site: rms-normed MLP input rows}."""
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; box = {}
        def col(s_, w):
            if s_ in sites or (s_[0] == "mlpin" and ("mlp", s_[1]) in sites):
                box[s_] = w
        forward(m, idx, collect=col)
        yield idx, {s_: box[s_].reshape(-1, D) for s_ in sites}, {s_: F.rms_norm(box[("mlpin", s_[1])], (D,)).reshape(-1, D) for s_ in sites}


def ridge_fit(m, rows, sites, mode, Um):
    G = None; B = {}; sp = None; sw = {}; ss = {}; n = 0
    for idx, W_, X_ in collect_pass(m, rows, sites):
        e = emb(m, idx).reshape(-1, D)
        for s_ in sites:
            phi = feats(mode, e, X_[s_], Um).double(); Y = W_[s_].double()
            if G is None:
                nf = phi.shape[1]; G = torch.zeros(nf, nf, dtype=torch.float64, device=DEV); sp = torch.zeros(nf, dtype=torch.float64, device=DEV)
                for t in sites:
                    B[t] = torch.zeros(nf, D, dtype=torch.float64, device=DEV); sw[t] = torch.zeros(D, dtype=torch.float64, device=DEV); ss[t] = 0.0
            if s_ == sites[0]:
                G += phi.T @ phi; sp += phi.sum(0); n += phi.shape[0]
            B[s_] += phi.T @ Y; sw[s_] += Y.sum(0); ss[s_] += float((Y ** 2).sum())
    out = {}
    for s_ in sites:
        # features differ per site only through the input rows; G above uses the first site's features -> refit G per site for COREIN/FULLIN
        pass
    # exact per-site Gram (features depend on the site's own input rows)
    for s_ in sites:
        G = None; sp = None; n = 0
        for idx, W_, X_ in collect_pass(m, rows, [s_]):
            phi = feats(mode, emb(m, idx).reshape(-1, D), X_[s_], Um).double()
            if G is None:
                nf = phi.shape[1]; G = torch.zeros(nf, nf, dtype=torch.float64, device=DEV); sp = torch.zeros(nf, dtype=torch.float64, device=DEV)
            G += phi.T @ phi; sp += phi.sum(0); n += phi.shape[0]
        mp = sp / n; Gc = G / n - torch.outer(mp, mp); mw = sw[s_] / n; Bc = B[s_] / n - torch.outer(mp, mw)
        A = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / Gc.shape[0] * torch.eye(Gc.shape[0], dtype=torch.float64, device=DEV), Bc)
        out[s_] = {"mu": mw.float(), "A": A.float(), "mphi": mp.float(), "r2_fit": float(torch.trace(Bc.T @ A)) / (ss[s_] / n - float((mw ** 2).sum()))}
    return out


def r2_core_heldout(m, rows, fits, mode, Um):
    """Held-out R^2 of the CORE-PROJECTED write."""
    num = {s: 0.0 for s in fits}; den = {s: 0.0 for s in fits}; Umf = Um.to(DEV).float()
    for idx, W_, X_ in collect_pass(m, rows, list(fits)):
        e = emb(m, idx).reshape(-1, D)
        for s_ in fits:
            phi = feats(mode, e, X_[s_], Um); pred = fits[s_]["mu"] + (phi - fits[s_]["mphi"]) @ fits[s_]["A"]
            num[s_] += float((((W_[s_] - pred) @ Umf) ** 2).sum()); den[s_] += float((((W_[s_] - fits[s_]["mu"]) @ Umf) ** 2).sum())
    return {s: 1 - num[s] / den[s] for s in fits}


def meanpatch(mu):
    mu = mu.to(DEV); return lambda w, x: mu.expand_as(w)


def oracle_core(mu, P):
    mu = mu.to(DEV); P = P.to(DEV).float(); return lambda w, x: mu + ((w - mu) @ P) @ P.T


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    m = R.load_model().to(DEV)
    if smoke:
        g = torch.Generator().manual_seed(0)
        nat = torch.randint(0, 50000, (8, 257), generator=g); fit_rows, ev = nat[:4].to(DEV), nat[4:].to(DEV)
    else:
        check_hashes()
        nat = torch.load(R.NAT, map_location="cpu"); nat = (nat["rows"] if isinstance(nat, dict) else nat).long()
        fit_rows, ev = nat[FIT[0]:FIT[1]].to(DEV), nat[EVAL[0]:EVAL[1]].to(DEV)
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    bases = fit(m, fit_rows, LATE7); log(stage="fit_cov")
    core = pooled([bases[s] for s in LATE7], False); P = core["U"][:, :KM]
    fits = {mode: ridge_fit(m, fit_rows, POOL, mode, P) for mode in ("IN", "INTOK")}
    for mode in fits:
        for s_ in POOL:
            fits[mode][s_]["mu"] = bases[s_]["mu"].to(DEV)
    # one-shot: target = SUM of the five real writes, features = x_hat at block 11
    G = None; Bm = None; sp = None; sw = None; n = 0; ss = 0.0
    for idx, W_, X_ in collect_pass(m, fit_rows, POOL):
        phi = X_[("mlp", 11)].double(); Y = sum(W_[s_] for s_ in POOL).double()
        if G is None:
            G = torch.zeros(D, D, dtype=torch.float64, device=DEV); Bm = torch.zeros(D, D, dtype=torch.float64, device=DEV); sp = torch.zeros(D, dtype=torch.float64, device=DEV); sw = torch.zeros(D, dtype=torch.float64, device=DEV)
        G += phi.T @ phi; Bm += phi.T @ Y; sp += phi.sum(0); sw += Y.sum(0); n += phi.shape[0]; ss += float((Y ** 2).sum())
    mp = sp / n; mw = sw / n; Gc = G / n - torch.outer(mp, mp); Bc = Bm / n - torch.outer(mp, mw)
    A1 = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / D * torch.eye(D, dtype=torch.float64, device=DEV), Bc)
    oneshot = {"mu": mw.float(), "A": A1.float(), "mphi": mp.float(), "r2_fit": float(torch.trace(Bc.T @ A1)) / (ss / n - float((mw ** 2).sum()))}
    r2 = {mode: {"fit": {f"{s[0]}{s[1]}": fits[mode][s]["r2_fit"] for s in POOL}, "heldout_full": {}} for mode in fits}
    # held-out full-write R^2 per block (real-stream inputs)
    for mode in fits:
        num = {s: 0.0 for s in POOL}; den = {s: 0.0 for s in POOL}
        for idx, W_, X_ in collect_pass(m, ev, POOL):
            e = emb(m, idx).reshape(-1, D)
            for s_ in POOL:
                phi = feats(mode, e, X_[s_], P); pred = fits[mode][s_]["mu"] + (phi - fits[mode][s_]["mphi"]) @ fits[mode][s_]["A"]
                num[s_] += float(((W_[s_] - pred) ** 2).sum()); den[s_] += float(((W_[s_] - fits[mode][s_]["mu"]) ** 2).sum())
        r2[mode]["heldout_full"] = {f"{s[0]}{s[1]}": 1 - num[s] / den[s] for s in POOL}
    r2["ONESHOT"] = {"fit": oneshot["r2_fit"]}
    log(stage="ridge", r2=r2)
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    def sur(mode):
        objs = {s: Surrogate(m, mode, fits[mode][s], None, P) for s in POOL}
        def hook(idx):
            for o in objs.values():
                o.hook(idx)
        return objs, hook
    arms = {}
    arms["POOL_MEAN"] = ce_of(m, ev, {s: meanpatch(bases[s]["mu"]) for s in POOL}) - ce0
    o, h = sur("IN"); arms["SEQ_LIN"] = ce_of(m, ev, o, h) - ce0
    o, h = sur("INTOK"); arms["SEQ_LIN_TOK"] = ce_of(m, ev, o, h) - ce0
    one = Surrogate(m, "IN", oneshot, None, P)
    zero = lambda w, x: torch.zeros_like(w)
    patch = {("mlp", 11): one, **{s: zero for s in POOL[1:]}}
    arms["ONESHOT_LIN"] = ce_of(m, ev, patch, one.hook) - ce0
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    rec = {k: 1 - arms[k] / arms["POOL_MEAN"] for k in arms if k != "POOL_MEAN"} if arms["POOL_MEAN"] > 0 else {k: float("nan") for k in arms}
    r2h = sorted(r2["IN"]["heldout_full"].values()); med_r2 = r2h[len(r2h) // 2]
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["POOL_MEAN"] - PRIOR_POOL) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_linear_in_own_input_recovers_half': bool(rec["SEQ_LIN"] >= BARS["b_min"]),
        'pred_c_chain_matters_little': bool(rec["ONESHOT_LIN"] >= BARS["c_mult"] * rec["SEQ_LIN"]),
        'pred_d_token_adds_little': bool(rec["SEQ_LIN_TOK"] <= rec["SEQ_LIN"] + BARS["d_slack"]),
        'pred_e_writes_are_linearly_predictable_from_input': bool(med_r2 >= BARS["e_r2_min"]),
    }
    nulls = {"b_null_rec_seq_le_.20": bool(rec["SEQ_LIN"] <= NULLS["b_max"]), "c_null_oneshot_le_.5_seq": bool(rec["ONESHOT_LIN"] <= NULLS["c_mult"] * rec["SEQ_LIN"]),
             "d_null_tok_adds_ge_.15": bool(rec["SEQ_LIN_TOK"] >= rec["SEQ_LIN"] + NULLS["d_slack"]), "e_null_med_r2_le_.25": bool(med_r2 <= NULLS["e_r2_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); rec = 1 - CE(arm)/CE(POOL_MEAN) (HIGHER = better)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "ridge_lambda_rel": LAM,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "pool_mean": arms["POOL_MEAN"], "prior_pool_mean": PRIOR_POOL},
           "ce_added": arms, "recovery": rec, "ridge_r2": r2, "median_heldout_r2_seq_lin": med_r2,
           "price": {"gpu_doc_forwards": 5 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (2 + 1 + 4), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "recovery", "ridge_r2", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "recovery": rec, "ridge_r2": r2}, indent=1))


if __name__ == "__main__":
    main()
