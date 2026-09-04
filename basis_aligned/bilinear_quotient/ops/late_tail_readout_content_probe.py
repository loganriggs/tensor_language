#!/usr/bin/env python
"""late_tail_readout_content_probe -- WHAT the unembedding reads from the late MLPs' tail writes. §2797: withholding the late-MLP tail
writes (blocks 8-17, the 384 dims outside the bus-768 core, centred, lambda-propagated in one accumulator A) from the FINAL READOUT alone
costs .1130; §2798/§2799: that channel is high-rank in every frame and has no small gate. This rung scores the same FINAL_ONLY arm PER
TOKEN and asks which tokens pay: by target class (induction / repeat / novel -- ops/target_token_classes.py), by fit-corpus unigram
frequency, by the real model's own loss on the token, by position; and whether the three largest writers (blocks 15/16/17) hurt the SAME
tokens (token-level Pearson of their per-token deltas) or different ones. CUDA lane-1.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_damage_concentrated_on_few_tokens pred_c_novel_targets_over_induction_targets
#                     pred_d_confident_tokens_carry_a_large_share pred_e_rare_targets_over_common_targets pred_f_writers_hurt_the_same_tokens
#                     pred_g_early_positions_carry_less

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER; per-token deltas are
CE added per token under the patch. Descriptive; nothing installs into the §312 frontier; bases are data covariances scored by CE only
(§2118 stays closed).
Preregistration: polynomial_causal/LATE_TAIL_READOUT_CONTENT_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_tail_readout_content_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R
from target_token_classes import target_token_classes

if not torch.cuda.is_available():
    raise RuntimeError("late_tail_readout_content_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_TAIL_READOUT_CONTENT_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_tail_write_consumer_probe_results.json"   # §2797
OUT = ROOT / "late_tail_readout_content_probe_results.json"
HASHES = {PREREG: "2b1961a042a19429dcd0e523ade4796413eb1c971d20e2fa9b0e90c039bbb405", PRIOR: "002f99919ef998efc44471bb1ff61ed681c0bd847cdb07d80d74ea5f9f9f1017",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_tail_readout_content_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_SPLIT8_1024 = 0.0374; PRIOR_LATE_MLP_768 = 0.1249; PRIOR_EARLY_TAIL_ONLY = 0.0711; PRIOR_ALL_TAILOUT = 0.1459; PRIOR_READERS_ONLY = 0.0404; PRIOR_FINAL_ONLY = 0.1130; LAM = 1e-2
K = 1024; SPLIT = 8; EARLY = list(range(0, 11)); LATE = list(range(11, 18))
WRITERS = (15, 16, 17); FREQ_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256]; LOSS_EDGES = [0.5, 1.0, 2.0, 4.0, 8.0]; RARE_MAX = 3; COMMON_MIN = 64; CONF_LOSS = 0.5
SITES = [("mlp", l) for l in range(NL)]; ASITES = [("attn", l) for l in range(NL)]; FINAL = ("final", -1)
ESITES = [(kd, l) for l in EARLY for kd in ("attn", "mlp")]; LSITES = [("mlp", l) for l in LATE] + [("attn", l) for l in LATE]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "none_tol": 1e-3, "b_top10": 0.6, "c_ratio": 1.2, "d_conf": 0.35, "e_ratio": 2.0, "f_r": 0.25, "g_ratio": 0.6}
NULLS = {"b_top10": 0.35, "c_ratio": 0.8, "d_conf": 0.15, "e_ratio": 1.2, "f_r": 0.1, "g_ratio": 0.9}


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
    if patch and ("final", -1) in patch:
        x = patch[("final", -1)](x, x)
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
    st = {s_: {"Gx": torch.zeros(D, D, dtype=torch.float64, device=DEV), "sx": torch.zeros(D, dtype=torch.float64, device=DEV)} for s_ in SITES + ASITES}
    wst = {s_: {"Gw": torch.zeros(D, D, dtype=torch.float64, device=DEV), "sw": torch.zeros(D, dtype=torch.float64, device=DEV)} for s_ in SITES + ASITES}
    class Rec:
        def __init__(self, s_): self.s = s_
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)).reshape(-1, D).double(); st[self.s]["Gx"] += xh.T @ xh; st[self.s]["sx"] += xh.sum(0); return w
    n = 0
    for i in range(0, fit_rows.shape[0], CH):
        idx = fit_rows[i:i + CH, :TI]; box = {}
        forward(m, idx, patch={s_: Rec(s_) for s_ in ASITES}, collect=lambda s_, w: box.__setitem__(s_, w) if (s_[0] == "mlpin" and ("mlp", s_[1]) in SITES) or s_ in SITES + ASITES else None)
        for s_ in SITES:
            xh = F.rms_norm(box[("mlpin", s_[1])], (D,)).reshape(-1, D).double(); st[s_]["Gx"] += xh.T @ xh; st[s_]["sx"] += xh.sum(0)
        for s_ in SITES + ASITES:
            ww = box[s_].reshape(-1, D).double(); wst[s_]["Gw"] += ww.T @ ww; wst[s_]["sw"] += ww.sum(0)
        n += idx.numel()
    me = None
    heads = {}
    for s_ in SITES + ASITES:
        mx = st[s_]["sx"] / n; Cx = st[s_]["Gx"] / n - torch.outer(mx, mx); evx, Ux = torch.linalg.eigh(Cx); Ux = Ux.flip(1)
        heads[s_] = {"U": Ux.float(), "mx": mx.float(), "Afull": None, "Cx": Cx, "eff_rank": R.spectrum(Cx.cpu())["eff_rank"]}
    wheads = {}
    for s_ in SITES + ASITES:
        mw = wst[s_]["sw"] / n; Cw = wst[s_]["Gw"] / n - torch.outer(mw, mw); _, Uw = torch.linalg.eigh(Cw); Uw = Uw.flip(1)
        wheads[s_] = {"U": Uw.float(), "mu": mw.float(), "Cw": Cw, "eff_rank": R.spectrum(Cw.cpu())["eff_rank"]}
    del st, wst
    log(stage="heads_done")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    class OwnHead:
        """Block l with its own weights on a k-dim input subspace (its own top-k input PCs, or a supplied basis U_); the rest of the input
        is the fit-set constant ('const') or the constant plus the ridge token read ('tok', §2730 recipe); output unrestricted."""
        def __init__(self, l, k, fill, U_=None):
            h = heads[("mlp", l)]; self.mlp = m.transformer.h[l].mlp; self.U = h["U"][:, :k] if U_ is None else U_; self.mx = h["mx"]; self.fill = fill
            self.A = h["Afull"]
        def hook(self, idx):
            self.idx = idx
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); core_ = (xh @ self.U) @ self.U.T
            if self.fill == "tok":
                e = F.rms_norm(m.transformer.wte(self.idx), (D,)); fb = self.mx + (e - me.float()) @ self.A
            else:
                fb = self.mx
            xp_ = core_ + fb - (fb @ self.U) @ self.U.T
            return self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
    class AttnHead:
        """Attention block l recomputed from a projected input h = mx + U U^T (xhat - mx) (or the exact xhat when full); own weights;
        the block-0 value residual v1 is recomputed exactly from the token embedding."""
        def __init__(self, l, k, U_=None, full=False):
            self.a = m.transformer.h[l].attn; h = heads[("attn", l)]; self.U = (h["U"] if U_ is None else U_)[:, :k]; self.mx = h["mx"]; self.full = full; self.l = l
        def hook(self, idx):
            B, Tn = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); b0 = m.transformer.h[0]
            self.v1 = b0.attn.c_v(F.rms_norm(b0.lambdas[0] * x0 + b0.lambdas[1] * x0, (D,))).view(B, Tn, NH, HD)
            self.cos, self.sin = (t.to(DEV) for t in R.rope(Tn)); self.mask = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=DEV))
        def __call__(self, w, x):
            B, Tn = x.shape[:2]; xh = F.rms_norm(x, (D,)); a = self.a
            h = xh if self.full else self.mx + ((xh - self.mx) @ self.U) @ self.U.T
            def pr(lin):
                return R.rot(F.rms_norm(lin(h).view(B, Tn, NH, HD), (HD,)), self.cos, self.sin)
            v = a.c_v(h).view(B, Tn, NH, HD)
            if self.l != 0:
                v = (1 - a.lamb) * v + a.lamb * self.v1.view_as(v)
            pat = (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q), pr(a.c_k)) / HD) * (torch.einsum("bqhd,bkhd->bhqk", pr(a.c_q2), pr(a.c_k2)) / HD)
            pat = pat.masked_fill(~self.mask, 0.0)
            return a.c_proj(torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, Tn, D))
    class Buf:
        def __init__(self): self.acc = None
        def hook(self, idx): self.acc = None
        def add(self, t): self.acc = t.clone() if self.acc is None else self.acc + t
    class Route:
        """Split site s's write into its part in the read core and its remainder; mode 'delete' drops the remainder, 'readout' moves
        it to the buffer (added before the final norm), 'hidden' keeps it in the stream and books it for subtraction at the end."""
        def __init__(self, s_, k, U_, mode, buf):
            self.U = U_[:, :k]; self.mu = wheads[s_]["mu"]; self.mode = mode; self.buf = buf
        def hook(self, idx):
            pass
        def __call__(self, w, x):
            c = w - self.mu; inside = (c @ self.U) @ self.U.T; out = c - inside
            if self.mode == "delete":
                return self.mu + inside
            self.buf.add(out)
            return self.mu + inside if self.mode == "readout" else w
    class FinalAdd:
        def __init__(self, buf, sign): self.buf = buf; self.sign = sign
        def hook(self, idx):
            pass
        def __call__(self, x, _):
            return x if self.buf.acc is None else x + self.sign * self.buf.acc
    class Both:
        def __init__(self, reader, writer): self.r = reader; self.w = writer
        def hook(self, idx):
            self.r.hook(idx); self.w.hook(idx)
        def __call__(self, w, x):
            return self.w(self.r(w, x), x)
    def core_of(sites):
        C = sum(heads[s_]["Cx"] for s_ in sites) / len(sites); _, U_ = torch.linalg.eigh(C); return U_.flip(1).float()
    U_late = core_of(LSITES)
    def run(patch, extra=()):
        objs = list(patch.values()) + list(extra)
        def hook(idx):
            for o in objs:
                o.hook(idx)
        return ce_of(m, ev, patch, hook) - ce0
    def head(s_, k, U_=None):
        return OwnHead(s_[1], k, "const", None if U_ is None else U_[:, :k]) if s_[0] == "mlp" else AttnHead(s_[1], k, U_)
    def blocks(ls):
        return [(kd, l) for l in ls for kd in ("attn", "mlp") if 0 <= l < NL]
    OWN8 = blocks(range(0, SPLIT)); SET8 = blocks(range(SPLIT, NL)); U_8 = core_of(SET8)
    def split8(k):
        patch = {s_: head(s_, k) for s_ in OWN8}
        for s_ in SET8:
            patch[s_] = head(s_, k, U_8)
        return run(patch)
    LB = list(range(SPLIT, NL))
    Uk = U_8[:, :768]
    def perp(z):
        return z - (z @ Uk) @ Uk.T
    class Box:
        """One lambda-propagated accumulator A of the late MLPs' centred TAIL writes (perp(write) - fit mean, blocks in `writers`), in stream
        units. Each consumer in `mlps` / `attns` / `final` computes from x - A instead of x; the stream itself keeps every real write, so
        the consumers NOT named see the tail exactly as the real model does. With all consumers named, x - A is the write-site drop of §2796."""
        def __init__(self, writers, mlps=(), attns=(), final=False):
            self.writers = set(writers); self.mlps = set(mlps); self.attns = set(attns); self.final = final; self.A = None
        def hook(self, idx):
            self.A = None
        def see(self, x, on):
            return x - self.A if (on and self.A is not None) else x
    class CAttn(AttnHead):
        """Late attention l: propagates A by lambda0 at block entry (Tracker semantics), then recomputes attention exactly from x - A if it is a consumer."""
        def __init__(self, l, box):
            super().__init__(l, D, full=True); self.box = box; self.l = l
        def __call__(self, w, x):
            b = self.box
            if b.A is not None and self.l != SPLIT:
                b.A = m.transformer.h[self.l].lambdas[0] * b.A
            return super().__call__(w, b.see(x, self.l in b.attns))
    class CMLP(OwnHead):
        """Late MLP l: computes from x - A if it is a consumer; if it is a writer, adds its centred tail write to A. rec: fit-pass sums."""
        def __init__(self, l, box, mu=None, rec=None):
            super().__init__(l, 768, "const", Uk); self.l = l; self.box = box; self.mu = mu; self.rec = rec
        def __call__(self, w, x):
            b = self.box; xh = F.rms_norm(b.see(x, self.l in b.mlps), (D,)); out = self.mlp.Down(self.mlp.Left(xh) * self.mlp.Right(xh))
            if self.rec is not None:
                r = self.rec; pm = perp(out).reshape(-1, D); r[("sum", self.l)] += pm.double().sum(0); r[("n", self.l)] += pm.shape[0]
            if self.l in b.writers:
                d = perp(out) - self.mu[self.l]; b.A = d if b.A is None else b.A + d
            return out
    class CFinal:
        def __init__(self, box): self.box = box
        def hook(self, idx):
            pass
        def __call__(self, x, _):
            return self.box.see(x, self.box.final)
    rec = {}
    for l in LB:
        rec[("sum", l)] = torch.zeros(D, dtype=torch.float64, device=DEV); rec[("n", l)] = 0
    for i in range(0, fit_rows.shape[0], CH):
        forward(m, fit_rows[i:i + CH, :TI], patch={("mlp", l): CMLP(l, Box(()), rec=rec) for l in LB})
    mu = {l: (rec[("sum", l)] / rec[("n", l)]).float() for l in LB}
    log(stage="fit", n_fit_tokens=rec[("n", SPLIT)], mu_norm_17=round(float(mu[17].norm()), 3))
    def consumer(writers, mlps=(), attns=(), final=False):
        box = Box(writers, mlps, attns, final)
        patch = {("mlp", l): CMLP(l, box, mu=mu) for l in LB}; patch.update({("attn", l): CAttn(l, box) for l in LB}); patch[("final", -1)] = CFinal(box)
        return run(patch, extra=(box,))
    def tok_loss(patch=None, extra=()):
        """Per-token held-out CE [n_eval_docs, TI] (double) under a patch; same forward as ce_of."""
        objs = (list(patch.values()) if patch else []) + list(extra); outs = []
        for i in range(0, ev.shape[0], CH):
            idx = ev[i:i + CH, :TI]; tgt = ev[i:i + CH, 1:TI + 1]
            for o in objs:
                o.hook(idx)
            lg = forward(m, idx, patch)
            outs.append(F.cross_entropy(lg.reshape(-1, V).float(), tgt.reshape(-1), reduction="none").view(idx.shape[0], TI))
        return torch.cat(outs, 0).double()
    def final_tok(writers):
        box = Box(writers, final=True)
        patch = {("mlp", l): CMLP(l, box, mu=mu) for l in LB}; patch.update({("attn", l): CAttn(l, box) for l in LB}); patch[("final", -1)] = CFinal(box)
        return tok_loss(patch, extra=(box,))
    arms = {"SPLIT8_1024": split8(1024), "NONE": consumer(LB)}
    base = tok_loss()
    tok = {"FINAL_ONLY": final_tok(LB)}
    tok.update({f"FINAL_W{j}": final_tok((j,)) for j in WRITERS})
    delta = {k: (v - base) for k, v in tok.items()}                       # per-token CE ADDED (LOWER IS BETTER)
    arms.update({k: float(v.mean()) for k, v in delta.items()})
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    # --- token axes (all computed from the eval rows and the FIT corpus only) ---
    tgt = ev[:, 1:TI + 1]; inp = ev[:, :TI]
    cls = target_token_classes(inp, tgt)                                  # induction / repeat / novel (Codex's audited module)
    cnt = torch.bincount(fit_rows.reshape(-1), minlength=V)               # fit-corpus unigram counts (docs 96-191)
    fc = cnt[tgt]
    fbin = torch.bucketize(fc, torch.tensor(FREQ_EDGES, device=DEV))      # 0: unseen, 1: 1, 2: 2-3, ..., 9: >= 256
    lbin = torch.bucketize(base, torch.tensor(LOSS_EDGES, device=DEV, dtype=torch.float64))
    pos = torch.arange(TI, device=DEV).expand_as(tgt)
    dF = delta["FINAL_ONLY"]; total = float(dF.sum()); ntok = dF.numel()
    def share(mask):
        n = int(mask.sum()); s = float(dF[mask].sum())
        return {"n": n, "tok_share": n / ntok, "damage_share": s / total if abs(total) > 1e-12 else None, "per_token": s / n if n else None}
    def per_tok(mask):
        n = int(mask.sum()); return float(dF[mask].sum()) / n if n else None
    def sdiv(a, b):
        return (a / b) if (a is not None and b is not None and abs(b) > 1e-12) else None
    flat = dF.reshape(-1); k10 = max(1, ntok // 10)
    top10 = float(flat.topk(k10).values.sum()) / total if abs(total) > 1e-12 else None
    by_class = {c: share(cls[c]) for c in ("induction", "repeat", "novel")}
    by_freq = {f"f{b}": share(fbin == b) for b in range(len(FREQ_EDGES) + 1)}
    by_loss = {f"l{b}": share(lbin == b) for b in range(len(LOSS_EDGES) + 1)}
    by_pos = {f"p{q}": share((pos >= q * 32) & (pos < (q + 1) * 32)) for q in range(TI // 32)}
    rare = per_tok(fc <= RARE_MAX); common = per_tok(fc >= COMMON_MIN)
    early = per_tok(pos < 32); late = per_tok(pos >= 128)
    conf = share(base < CONF_LOSS)["damage_share"]
    def pearson(a, b):
        a = a.reshape(-1) - a.mean(); b = b.reshape(-1) - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))
    rw = {f"r_W17_W{j}": pearson(delta["FINAL_W17"], delta[f"FINAL_W{j}"]) for j in WRITERS if j != 17}
    rw["r_W15_W16"] = pearson(delta["FINAL_W15"], delta["FINAL_W16"])
    r_base = pearson(dF, base)
    neg_frac = float((dF < 0).double().mean())
    writer_by_class = {f"W{j}": {c: sdiv(float(delta[f"FINAL_W{j}"][cls[c]].sum()), float(cls[c].sum())) for c in ("induction", "repeat", "novel")} for j in WRITERS}
    summ = {"total_damage_per_token": total / ntok, "n_eval_tokens": ntok, "top10_share": top10, "neg_frac": neg_frac, "r_delta_vs_baseline_loss": r_base,
            "by_class": by_class, "novel_over_induction": sdiv(by_class["novel"]["per_token"], by_class["induction"]["per_token"]),
            "by_freq_bin": by_freq, "rare_per_token": rare, "common_per_token": common, "rare_over_common": sdiv(rare, common),
            "by_loss_bin": by_loss, "confident_damage_share": conf,
            "by_pos_32": by_pos, "early_per_token": early, "late_per_token": late, "early_over_late": sdiv(early, late),
            "writer_token_pearson": rw, "writer_by_class_per_token": writer_by_class,
            "prior_final_only_2797": PRIOR_FINAL_ONLY, "n_fit_tokens": rec[("n", SPLIT)]}
    def ge(v, b):
        return bool(v is not None and v >= b)
    def le(v, b):
        return bool(v is not None and v <= b)
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["SPLIT8_1024"] - PRIOR_SPLIT8_1024) <= BARS["repro_tol"] and abs(arms["NONE"]) <= BARS["none_tol"]
                   and abs(arms["FINAL_ONLY"] - PRIOR_FINAL_ONLY) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_damage_concentrated_on_few_tokens': ge(top10, BARS["b_top10"]),
        'pred_c_novel_targets_over_induction_targets': ge(summ["novel_over_induction"], BARS["c_ratio"]),
        'pred_d_confident_tokens_carry_a_large_share': ge(conf, BARS["d_conf"]),
        'pred_e_rare_targets_over_common_targets': ge(summ["rare_over_common"], BARS["e_ratio"]),
        'pred_f_writers_hurt_the_same_tokens': bool(ge(rw["r_W17_W15"], BARS["f_r"]) and ge(rw["r_W17_W16"], BARS["f_r"])),
        'pred_g_early_positions_carry_less': le(summ["early_over_late"], BARS["g_ratio"]),
    }
    nulls = {"b_null_top10_le_.35": le(top10, NULLS["b_top10"]), "c_null_novel_over_induction_le_.8": le(summ["novel_over_induction"], NULLS["c_ratio"]),
             "d_null_confident_share_le_.15": le(conf, NULLS["d_conf"]), "e_null_rare_over_common_le_1.2": le(summ["rare_over_common"], NULLS["e_ratio"]),
             "f_null_min_writer_r_le_.1": bool(min(rw["r_W17_W15"], rw["r_W17_W16"]) <= NULLS["f_r"]), "g_null_early_over_late_ge_.9": ge(summ["early_over_late"], NULLS["g_ratio"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); per-token deltas are CE added per token",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "split": SPLIT,
           "program": "the §2797 consumer instrument (late-MLP centred tail writes in one lambda-propagated A, withheld from the FINAL READOUT only) scored PER TOKEN and decomposed by target class (induction/repeat/novel, ops/target_token_classes.py), fit-corpus unigram frequency bin, baseline-loss bin and position; writer arms FINAL_W15/16/17 compared token by token",
           "freq_edges": FREQ_EDGES, "loss_edges": LOSS_EDGES, "rare_max": RARE_MAX, "common_min": COMMON_MIN, "conf_loss": CONF_LOSS,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "split8_1024": arms["SPLIT8_1024"], "prior_split8_1024": PRIOR_SPLIT8_1024, "none": arms["NONE"], "final_only": arms["FINAL_ONLY"], "prior_final_only": PRIOR_FINAL_ONLY,
                          "tok_mean_baseline": float(base.mean())},
           "ce_added": arms, "summary": summ,
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (2 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "summary", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
