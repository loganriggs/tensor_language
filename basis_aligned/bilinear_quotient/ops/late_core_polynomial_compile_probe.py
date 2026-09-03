#!/usr/bin/env python
"""late_core_polynomial_compile_probe -- mlp16/17 core program compiled EXACTLY into 16 token-free 16x16 quadratic forms + a per-token 16x16
linear read + a per-token 16-vector offset (no hidden width); term ablations say which carries the value. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_quadratic_term_matters pred_c_linear_term_matters pred_d_token_enters_through_the_offset
#                     pred_e_quadratic_forms_are_simple

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_CORE_POLYNOMIAL_COMPILE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_core_polynomial_compile_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_core_polynomial_compile_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_CORE_POLYNOMIAL_COMPILE_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_mlp_weights_on_core_input_probe_results.json"   # §2720
OUT = ROOT / "late_core_polynomial_compile_probe_results.json"
HASHES = {PREREG: "a2e9b133e64d339652a4440c3d621fa43ddaed9932e21a2453f05fbe00861119", PRIOR: "577724584361261327348db32b41dda18aaff8eed6877089a995b9055fa0e55c",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_core_polynomial_compile_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_TOKFILL_M = 0.233; LAM = 1e-2
KR = 4
BARS = {"ce_tol": 1e-4, "exact_tol": 0.003, "repro_tol": 0.02, "b_min": 0.15, "c_min": 0.15, "d_max": 0.04, "e_max": 0.03}
NULLS = {"b_max": 0.03, "c_max": 0.03, "d_min": 0.10, "e_min": 0.10}


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


def fit_filler(m, rows, sites, Pm):
    """Per site: mean of the non-core input x_perp and a ridge map e(t) -> x_perp (both on the fit set)."""
    Pm = Pm.to(DEV).float(); out = {}
    for s_ in sites:
        G = None
        for idx, W_, X_ in collect_pass(m, rows, [s_]):
            e = emb(m, idx).reshape(-1, D).double(); xh = X_[s_].double(); xp = xh - (xh @ Pm.double()) @ Pm.double().T
            if G is None:
                G = torch.zeros(D, D, dtype=torch.float64, device=DEV); Bm = torch.zeros(D, D, dtype=torch.float64, device=DEV)
                se = torch.zeros(D, dtype=torch.float64, device=DEV); sx = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
            G += e.T @ e; Bm += e.T @ xp; se += e.sum(0); sx += xp.sum(0); n += e.shape[0]
        me = se / n; mx = sx / n; Gc = G / n - torch.outer(me, me); Bc = Bm / n - torch.outer(me, mx)
        A = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / D * torch.eye(D, dtype=torch.float64, device=DEV), Bc)
        out[s_] = {"xbar_perp": mx.float(), "A_fill": A.float(), "me": me.float()}
    return out


class WeightsArm:
    """The block's OWN mlp applied to x' = P x_hat + filler; filler in {MEAN, TOK, RAND}; optional core output restriction."""
    def __init__(self, m, l, fill, mode, mu, Pm, restrict):
        self.mlp = m.transformer.h[l].mlp; self.fill = fill; self.mode = mode; self.mu = mu.to(DEV); self.Pm = Pm.to(DEV).float(); self.restrict = restrict; self.e = None
    def hook(self, idx):
        self.idx = idx
    def __call__(self, w, x):
        B, T, _ = w.shape
        xh = F.rms_norm(x, (D,)); core = (xh @ self.Pm) @ self.Pm.T
        if self.mode == "MEAN":
            fl = self.fill["xbar_perp"]
        elif self.mode == "TOK":
            e = F.rms_norm(self.mlp_wte(self.idx), (D,))
            fl = self.fill["xbar_perp"] + (e - self.fill["me"]) @ self.fill["A_fill"]
        else:  # RAND: x_perp of a random other position in the same chunk (fixed permutation per call, seeded)
            xp = xh - core; flat = xp.reshape(-1, D)
            g = torch.Generator(device="cpu").manual_seed(int(self.idx[0, 0]) + 7 * T)
            perm = torch.randperm(flat.shape[0], generator=g).to(DEV)
            fl = flat[perm].view(B, T, D)
        xp_ = core + fl
        wr = self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
        if self.restrict:
            wr = self.mu + ((wr - self.mu) @ self.Pm) @ self.Pm.T
        return wr


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
    core = pooled([bases[s] for s in LATE7], False); P = core["U"][:, :KM].to(DEV).float()
    fills = fit_filler(m, fit_rows, LAST2, P); log(stage="fits_done")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    class Compiled:
        """y = c^T A c + b(t)^T c + d(t) in the 16-dim core; write = mu + P (y - P^T mu)."""
        def __init__(self, l, fill, mu, tok, use_quad, use_cross, use_off, cross_tok, rank):
            mlp = m.transformer.h[l].mlp; self.fill = fill; self.mu = mu.to(DEV).float(); self.tok = tok
            self.use = (use_quad, use_cross, use_off); self.cross_tok = cross_tok
            Lc = mlp.Left.weight.float() @ P; Rc = mlp.Right.weight.float() @ P            # 4608 x 16
            Dc = (P.T @ mlp.Down.weight.float())                                            # 16 x 4608
            A = torch.einsum("hi,kh,hj->kij", Lc, Dc, Rc)                                   # 16 x 16 x 16
            if rank is not None:
                U_, S_, Vh_ = torch.linalg.svd(A); A = (U_[:, :, :rank] * S_[:, None, :rank]) @ Vh_[:, :rank, :]
            self.A = A; self.Lc, self.Rc, self.Dc = Lc, Rc, Dc; self.Left, self.Right = mlp.Left, mlp.Right
            self.pmu = self.mu @ P
            self.l0 = self.Left(fill["xbar_perp"]).float(); self.r0 = self.Right(fill["xbar_perp"]).float()
        def hook(self, idx):
            self.idx = idx
        def filler(self, B, T):
            if not self.tok:
                return None
            e = F.rms_norm(m.transformer.wte(self.idx), (D,))
            return self.fill["xbar_perp"] + (e - self.fill["me"]) @ self.fill["A_fill"]
        def __call__(self, w, x):
            B, T, _ = w.shape
            c = F.rms_norm(x, (D,)) @ P                                                     # B T 16
            f = self.filler(B, T)
            if f is None:
                l_, r_ = self.l0, self.r0
            else:
                l_, r_ = self.Left(f).float(), self.Right(f).float()                        # B T 4608
            lc_, rc_ = (self.l0, self.r0) if not self.cross_tok else (l_, r_)
            y = torch.zeros(B, T, KM, device=DEV)
            if self.use[0]:
                y = y + torch.einsum("bti,kij,btj->btk", c, self.A, c)
            if self.use[1]:
                cL = c @ self.Lc.T; cR = c @ self.Rc.T                                        # B T 4608
                y = y + (cL * rc_ + lc_ * cR) @ self.Dc.T
            if self.use[2]:
                y = y + (l_ * r_) @ self.Dc.T
            return self.mu + (y - self.pmu) @ P.T
    def pair(**kw):
        objs = [Compiled(l, fills[("mlp", l)], bases[("mlp", l)]["mu"], **kw) for l in (16, 17)]
        patch = {("mlp", 16): objs[0], ("mlp", 17): objs[1]}
        def hook(idx):
            for o in objs:
                o.hook(idx)
        return ce_of(m, ev, patch, hook) - ce0
    def wpair(mode):
        objs = []
        for s_ in LAST2:
            o = WeightsArm(m, s_[1], fills[s_], mode, bases[s_]["mu"], P, True); o.mlp_wte = m.transformer.wte; objs.append(o)
        patch = {LAST2[0]: objs[0], LAST2[1]: objs[1]}
        def hook(idx):
            for o in objs:
                o.hook(idx)
        return ce_of(m, ev, patch, hook) - ce0
    arms = {}
    arms["MEAN_16_17"] = ce_of(m, ev, {s: meanpatch(bases[s]["mu"]) for s in LAST2}) - ce0
    arms["W_TOKFILL_M"] = wpair("TOK"); arms["W_MEANFILL_M"] = wpair("MEAN")
    base = dict(tok=True, use_quad=True, use_cross=True, use_off=True, cross_tok=True, rank=None)
    arms["COMPILED_TOK"] = pair(**base)
    arms["COMPILED_MEAN"] = pair(**{**base, "tok": False})
    arms["NO_QUAD"] = pair(**{**base, "use_quad": False})
    arms["NO_CROSS"] = pair(**{**base, "use_cross": False})
    arms["NO_OFFSET"] = pair(**{**base, "use_off": False})
    arms["CROSS_TOKFREE"] = pair(**{**base, "cross_tok": False})
    arms[f"QUAD_RANK{KR}"] = pair(**{**base, "rank": KR})
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    ct = arms["COMPILED_TOK"]
    deltas = {k: arms[k] - ct for k in ("NO_QUAD", "NO_CROSS", "NO_OFFSET", "CROSS_TOKFREE", f"QUAD_RANK{KR}")}
    exact = {"tok": abs(arms["COMPILED_TOK"] - arms["W_TOKFILL_M"]), "mean": abs(arms["COMPILED_MEAN"] - arms["W_MEANFILL_M"])}
    rec = {k: 1 - v / arms["MEAN_16_17"] for k, v in arms.items()}
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and exact["tok"] <= BARS["exact_tol"] and exact["mean"] <= BARS["exact_tol"] and abs(arms["W_TOKFILL_M"] - PRIOR_TOKFILL_M) <= BARS["repro_tol"]) if not smoke else bool(exact["tok"] <= BARS["exact_tol"] and exact["mean"] <= BARS["exact_tol"])
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_quadratic_term_matters': bool(deltas["NO_QUAD"] >= BARS["b_min"]),
        'pred_c_linear_term_matters': bool(deltas["NO_CROSS"] >= BARS["c_min"]),
        'pred_d_token_enters_through_the_offset': bool(deltas["CROSS_TOKFREE"] <= BARS["d_max"]),
        'pred_e_quadratic_forms_are_simple': bool(deltas[f"QUAD_RANK{KR}"] <= BARS["e_max"]),
    }
    nulls = {"b_null_quad_le_.03": bool(deltas["NO_QUAD"] <= NULLS["b_max"]), "c_null_cross_le_.03": bool(deltas["NO_CROSS"] <= NULLS["c_max"]),
             "d_null_cross_token_ge_.10": bool(deltas["CROSS_TOKFREE"] >= NULLS["d_min"]), "e_null_rank4_ge_.10": bool(deltas[f"QUAD_RANK{KR}"] >= NULLS["e_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); rec = 1 - CE/CE(MEAN_16_17); deltas = CE(ablation) - CE(COMPILED_TOK)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "k_core": KM, "quad_rank": KR,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "w_tokfill_m": arms["W_TOKFILL_M"], "prior_w_tokfill_m": PRIOR_TOKFILL_M, "exactness_abs_diff": exact},
           "ce_added": arms, "recovery_of_mean_16_17": rec, "ablation_deltas": deltas,
           "program_numbers": {"quadratic_forms": 2 * KM * KM * KM, "core_basis": D * KM, "per_token_linear_read_table": V * KM * KM * 2, "per_token_offset_table": V * KM * 2,
                               "note": "b(t), d(t) are affine/quadratic in e_t through A_fill and the block weights; as tables: 16x16 and 16 numbers per token per block"},
           "price": {"gpu_doc_forwards": 2 * int(fit_rows.shape[0]) + int(ev.shape[0]) * 11, "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "ablation_deltas", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "deltas": deltas}, indent=1))


if __name__ == "__main__":
    main()
