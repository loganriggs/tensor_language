#!/usr/bin/env python
"""late_stack_composition_penalty_anatomy_probe -- where the §2732 composition penalty (pi = .180 between pool OWN_32_TOK and the mlp16/17
shared-square program) lives: truncation vs interface (FULL compile), core-borne vs perpendicular (GUARD / COREERR), per pool block, co-adaptation.
CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_penalty_is_interface_not_truncation pred_c_guarding_the_core_removes_it
#                     pred_d_core_error_alone_carries_it pred_e_coadaptation_does_not_repair_it

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_STACK_COMPOSITION_PENALTY_ANATOMY_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_stack_composition_penalty_anatomy_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_stack_composition_penalty_anatomy_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_STACK_COMPOSITION_PENALTY_ANATOMY_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_stack_extracted_program_probe_results.json"   # §2732
OUT = ROOT / "late_stack_composition_penalty_anatomy_probe_results.json"
HASHES = {PREREG: "b84fba3dc96682a125a9a9db6a61da4c2d6e69187271629787190501e781783f", PRIOR: "a790b0f5e093e5625c690c19dbda9a4e36ab64a3ba41c4306b8d2de8e00dd68c",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_stack_composition_penalty_anatomy_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_MEAN7 = 1.885; PRIOR_POOL32TOK = 0.319; PRIOR_PROG = 0.246; PRIOR_COMB = 0.745; PRIOR_FULL = 0.233; LAM = 1e-2
POOL = [("mlp", l) for l in range(11, 16)]
KIN = 32; RSQ = 8; RB = 8
BARS = {"ce_tol": 1e-4, "repro_tol7": 0.03, "repro_tol": 0.02, "repro_tol_full": 0.03, "b_min": 0.12, "c_max": 0.06, "d_min": 0.10, "e_max": 0.03}
NULLS = {"b_max": 0.06, "c_min": 0.14, "d_max": 0.04, "e_min": 0.08}


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
    fills = fit_filler(m, fit_rows, LAST2, P)
    st = {s_: {"Gx": torch.zeros(D, D, dtype=torch.float64, device=DEV), "sx": torch.zeros(D, dtype=torch.float64, device=DEV),
               "Bex": torch.zeros(D, D, dtype=torch.float64, device=DEV)} for s_ in POOL}
    Ge = torch.zeros(D, D, dtype=torch.float64, device=DEV); se = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
    for idx, W_, X_ in collect_pass(m, fit_rows, POOL):
        e = emb(m, idx).reshape(-1, D).double(); Ge += e.T @ e; se += e.sum(0); n += e.shape[0]
        for s_ in POOL:
            xh = X_[s_].double(); st[s_]["Gx"] += xh.T @ xh; st[s_]["sx"] += xh.sum(0); st[s_]["Bex"] += e.T @ xh
    me = se / n; Gec = Ge / n - torch.outer(me, me)
    ev_, Ue = torch.linalg.eigh(Gec); ev_ = ev_.clamp_min(1e-12)
    Ceh = ((Ue * ev_.sqrt()) @ Ue.T).float(); Ceih = ((Ue * ev_.rsqrt()) @ Ue.T).float()
    heads = {}
    for s_ in POOL:
        mx = st[s_]["sx"] / n; Cx = st[s_]["Gx"] / n - torch.outer(mx, mx); evx, Ux = torch.linalg.eigh(Cx); Ux = Ux.flip(1)
        Bc = st[s_]["Bex"] / n - torch.outer(me, mx)
        Afull = torch.linalg.solve(Gec + LAM * torch.trace(Gec) / D * torch.eye(D, dtype=torch.float64, device=DEV), Bc)
        heads[s_] = {"U": Ux.float(), "mx": mx.float(), "Afull": Afull.float()}
    log(stage="heads_done")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    PP = P @ P.T
    class OwnHead:
        """Pool block l with its own weights on its top-k input PCs + token filler; mode: None | 'guard' (core component of the
        write taken from the REAL write) | 'coreerr' (only the core component taken from the truncated write)."""
        def __init__(self, l, k, mode=None):
            h = heads[("mlp", l)]; self.mlp = m.transformer.h[l].mlp; self.U = h["U"][:, :k]; self.mx = h["mx"]; self.A = h["Afull"]; self.mode = mode
        def hook(self, idx):
            self.idx = idx
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); core_ = (xh @ self.U) @ self.U.T
            e = F.rms_norm(m.transformer.wte(self.idx), (D,)); fb = self.mx + (e - me.float()) @ self.A
            xp_ = core_ + fb - (fb @ self.U) @ self.U.T
            wr = self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
            if self.mode == "guard":
                wr = wr + (w - wr) @ PP
            elif self.mode == "coreerr":
                wr = w + (wr - w) @ PP
            return wr
    class Full:
        """mlp16/17 own weights on the 16-dim core + token filler, output restricted to the core (= the exact 16-dim compile, §2727)."""
        def __init__(self, l, fill):
            self.mlp = m.transformer.h[l].mlp; self.fill = fill; self.mu = bases[("mlp", l)]["mu"].to(DEV).float()
        def hook(self, idx):
            self.idx = idx
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); e = F.rms_norm(m.transformer.wte(self.idx), (D,))
            fl = self.fill["xbar_perp"] + (e - self.fill["me"]) @ self.fill["A_fill"]
            xp_ = xh @ PP + fl
            wr = self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
            return self.mu + (wr - self.mu) @ PP
    # mlp16/17: exact compile, quadratic core on the SHARED top-RSQ square space, rank-RB token read, no offset (as §2732)
    def compile_block(l, fill):
        mlp = m.transformer.h[l].mlp
        Lc = mlp.Left.weight.float() @ P; Rc = mlp.Right.weight.float() @ P; Dc = P.T @ mlp.Down.weight.float()
        A = torch.einsum("hi,kh,hj->kij", Lc, Dc, Rc); Sy = 0.5 * (A + A.transpose(1, 2)); lam, Q = torch.linalg.eigh(Sy)
        S = torch.einsum("kr,kir,kjr->ij", lam.abs(), Q, Q)
        l0 = mlp.Left(fill["xbar_perp"]).float(); r0 = mlp.Right(fill["xbar_perp"]).float()
        b0 = torch.einsum("hi,kh,h->ki", Lc, Dc, r0) + torch.einsum("hi,kh,h->ki", Rc, Dc, l0)
        RA = fill["A_fill"].float() @ mlp.Right.weight.float().T; LA = fill["A_fill"].float() @ mlp.Left.weight.float().T
        B = (torch.einsum("hi,kh,dh->kid", Lc, Dc, RA) + torch.einsum("hi,kh,dh->kid", Rc, Dc, LA)).reshape(KM * KM, D)
        Mw = B @ Ceh; U_, S_, Vh_ = torch.linalg.svd(Mw, full_matrices=False)
        B = ((U_[:, :RB] * S_[:RB]) @ Vh_[:RB]) @ Ceih
        return {"A": A, "S": S, "b0": b0, "B": B, "me": fill["me"], "mu": bases[("mlp", l)]["mu"].to(DEV).float()}
    comp = {l: compile_block(l, fills[("mlp", l)]) for l in (16, 17)}
    _, SUsh = torch.linalg.eigh(comp[16]["S"] + comp[17]["S"]); SUsh = SUsh.flip(1); Pi = SUsh[:, :RSQ] @ SUsh[:, :RSQ].T
    class Prog:
        def __init__(self, cb):
            self.A = Pi @ cb["A"] @ Pi; self.b0, self.B, self.me_, self.mu = cb["b0"], cb["B"], cb["me"], cb["mu"]; self.pmu = self.mu @ P
        def hook(self, idx):
            self.idx = idx
        def __call__(self, w, x):
            B_, T, _ = w.shape
            c = F.rms_norm(x, (D,)) @ P
            de = F.rms_norm(m.transformer.wte(self.idx), (D,)) - self.me_
            bt = self.b0 + (de @ self.B.T).view(B_, T, KM, KM)
            y = torch.einsum("bti,kij,btj->btk", c, self.A, c) + torch.einsum("btki,bti->btk", bt, c)
            return self.mu + (y - self.pmu) @ P.T
    def run(parts):
        patch = {}; objs = []
        for p_ in parts:
            patch.update(p_); objs += list(p_.values())
        def hook(idx):
            for o in objs:
                o.hook(idx)
        return ce_of(m, ev, patch, hook) - ce0
    pool_part = lambda mode=None, blocks=range(11, 16): {("mlp", l): OwnHead(l, KIN, mode) for l in blocks}
    prog_part = lambda cp=comp: {("mlp", 16): Prog(cp[16]), ("mlp", 17): Prog(cp[17])}
    full_part = lambda: {("mlp", 16): Full(16, fills[("mlp", 16)]), ("mlp", 17): Full(17, fills[("mlp", 17)])}
    # co-adaptation: refit the mlp16/17 fillers on the POOL-perturbed stream (same ridge, same fit docs)
    def fit_filler_perturbed():
        pp = pool_part(); objs = list(pp.values()); Pm = P; out = {}
        acc = {s_: None for s_ in LAST2}
        for i in range(0, fit_rows.shape[0], CH):
            idx = fit_rows[i:i + CH, :TI]; box = {}
            for o in objs:
                o.hook(idx)
            forward(m, idx, patch=pp, collect=lambda s_, w: box.__setitem__(s_, w) if s_[0] == "mlpin" and ("mlp", s_[1]) in LAST2 else None)
            e = emb(m, idx).reshape(-1, D).double()
            for s_ in LAST2:
                xh = F.rms_norm(box[("mlpin", s_[1])], (D,)).reshape(-1, D).double(); xp = xh - (xh @ Pm.double()) @ Pm.double().T
                if acc[s_] is None:
                    acc[s_] = {"G": torch.zeros(D, D, dtype=torch.float64, device=DEV), "B": torch.zeros(D, D, dtype=torch.float64, device=DEV),
                               "se": torch.zeros(D, dtype=torch.float64, device=DEV), "sx": torch.zeros(D, dtype=torch.float64, device=DEV), "n": 0}
                a_ = acc[s_]; a_["G"] += e.T @ e; a_["B"] += e.T @ xp; a_["se"] += e.sum(0); a_["sx"] += xp.sum(0); a_["n"] += e.shape[0]
        for s_ in LAST2:
            a_ = acc[s_]; me_ = a_["se"] / a_["n"]; mx_ = a_["sx"] / a_["n"]; Gc = a_["G"] / a_["n"] - torch.outer(me_, me_); Bc = a_["B"] / a_["n"] - torch.outer(me_, mx_)
            A_ = torch.linalg.solve(Gc + LAM * torch.trace(Gc) / D * torch.eye(D, dtype=torch.float64, device=DEV), Bc)
            out[s_] = {"xbar_perp": mx_.float(), "A_fill": A_.float(), "me": me_.float()}
        return out
    fills_co = fit_filler_perturbed(); comp_co = {l: compile_block(l, fills_co[("mlp", l)]) for l in (16, 17)}; log(stage="coadapt_fit")
    arms = {"MEAN7": ce_of(m, ev, {s: meanpatch(bases[s]["mu"]) for s in LATE7}) - ce0}
    arms["POOL"] = run([pool_part()]); arms["PROG"] = run([prog_part()]); arms["FULL"] = run([full_part()])
    arms["COMBINED"] = run([pool_part(), prog_part()]); arms["COMBINED_FULL"] = run([pool_part(), full_part()])
    arms["POOL_GUARD"] = run([pool_part("guard")]); arms["POOL_GUARD+PROG"] = run([pool_part("guard"), prog_part()])
    arms["POOL_COREERR"] = run([pool_part("coreerr")]); arms["POOL_COREERR+PROG"] = run([pool_part("coreerr"), prog_part()])
    for l in range(11, 16):
        arms[f"POOL_{l}"] = run([pool_part(None, [l])]); arms[f"POOL_{l}+PROG"] = run([pool_part(None, [l]), prog_part()])
    arms["COMBINED_COADAPT"] = run([pool_part(), prog_part(comp_co)])
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    pis = {"pi": arms["COMBINED"] - arms["POOL"] - arms["PROG"], "pi_full": arms["COMBINED_FULL"] - arms["POOL"] - arms["FULL"],
           "pi_guard": arms["POOL_GUARD+PROG"] - arms["POOL_GUARD"] - arms["PROG"], "pi_coreerr": arms["POOL_COREERR+PROG"] - arms["POOL_COREERR"] - arms["PROG"]}
    pis.update({f"pi_{l}": arms[f"POOL_{l}+PROG"] - arms[f"POOL_{l}"] - arms["PROG"] for l in range(11, 16)})
    pis["sum_pi_l"] = sum(pis[f"pi_{l}"] for l in range(11, 16)); pis["coadapt_gain"] = arms["COMBINED"] - arms["COMBINED_COADAPT"]
    rec7 = {k: 1 - v / arms["MEAN7"] for k, v in arms.items()}
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["MEAN7"] - PRIOR_MEAN7) <= BARS["repro_tol7"] and abs(arms["POOL"] - PRIOR_POOL32TOK) <= BARS["repro_tol"]
                   and abs(arms["PROG"] - PRIOR_PROG) <= BARS["repro_tol"] and abs(arms["COMBINED"] - PRIOR_COMB) <= BARS["repro_tol"] and abs(arms["FULL"] - PRIOR_FULL) <= BARS["repro_tol_full"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_penalty_is_interface_not_truncation': bool(pis["pi_full"] >= BARS["b_min"]),
        'pred_c_guarding_the_core_removes_it': bool(pis["pi_guard"] <= BARS["c_max"]),
        'pred_d_core_error_alone_carries_it': bool(pis["pi_coreerr"] >= BARS["d_min"]),
        'pred_e_coadaptation_does_not_repair_it': bool(pis["coadapt_gain"] <= BARS["e_max"]),
    }
    nulls = {"b_null_pi_full_le_.06": bool(pis["pi_full"] <= NULLS["b_max"]), "c_null_pi_guard_ge_.14": bool(pis["pi_guard"] >= NULLS["c_min"]),
             "d_null_pi_coreerr_le_.04": bool(pis["pi_coreerr"] <= NULLS["d_max"]), "e_null_coadapt_gain_ge_.08": bool(pis["coadapt_gain"] >= NULLS["e_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); pi(X,Y) = CE(X+Y) - CE(X) - CE(Y)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "k_in_pool": KIN, "square_space_rank": RSQ, "token_read_rank": RB,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "mean7": arms["MEAN7"], "pool": arms["POOL"], "prog": arms["PROG"], "combined": arms["COMBINED"], "full": arms["FULL"],
                          "priors": {"mean7": PRIOR_MEAN7, "pool": PRIOR_POOL32TOK, "prog": PRIOR_PROG, "combined": PRIOR_COMB, "full": PRIOR_FULL}},
           "ce_added": arms, "recovery_of_mean7": rec7, "penalties": pis,
           "price": {"gpu_doc_forwards": 4 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "penalties", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "penalties": pis}, indent=1))


if __name__ == "__main__":
    main()
