#!/usr/bin/env python
"""late_square_direction_mean_control_probe -- is q1's 2.0-nat cost (§2736) the loss of a CONSTANT (mean of x^.q1, a linear-term carrier
for the bilinear form) or of per-token INFORMATION? Mean-preserving ablations (coordinate pinned to its fit-set mean) vs the zeroing
anchor vs mean-only removal, on the real mlp16/17 and on the compiled program. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_constant_carries_most pred_c_q1_is_mean_dominated pred_d_q1_still_informative
#                     pred_e_five_beat_random_mean_preserved

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_SQUARE_DIRECTION_MEAN_CONTROL_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_square_direction_mean_control_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_square_direction_mean_control_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_SQUARE_DIRECTION_MEAN_CONTROL_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_stack_extracted_program_probe_results.json"   # §2732
PRIOR36 = ROOT / "late_square_directions_ablation_probe_results.json"   # §2736
OUT = ROOT / "late_square_direction_mean_control_probe_results.json"
HASHES = {PREREG: "e27c805c4cd67de9eb6050bcded5f2bd21ed6a5a0111369f96b4b7182d6feb7f", PRIOR: "a790b0f5e093e5625c690c19dbda9a4e36ab64a3ba41c4306b8d2de8e00dd68c", PRIOR36: "78f58a79099a39924498c5f6e4d8e9784e98f8302398ee8d6bf3c06c661a2cc7",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_square_direction_mean_control_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_PROG = 0.246; PRIOR_ZERO_Q1 = 2.003; LAM = 1e-2
POOL = [("mlp", l) for l in range(11, 16)]
KIN = 32; RSQ = 8; RB = 8; NTOP = 5; SEEDS = [0, 1, 2]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.02, "anchor_tol": 0.05, "b_max": 1.0, "c_min": 2.0, "d_min": 0.10, "e_min": 2.0}
NULLS = {"b_min": 1.6, "c_max": 1.0, "d_max": 0.03, "e_max": 1.2}


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
            collect(("mlpin", l), xhat); collect(("xpre", l), x)
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
    Ge = torch.zeros(D, D, dtype=torch.float64, device=DEV); se = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
    for i in range(0, fit_rows.shape[0], CH):
        e = emb(m, fit_rows[i:i + CH, :TI]).reshape(-1, D).double(); Ge += e.T @ e; se += e.sum(0); n += e.shape[0]
    me = se / n; Gec = Ge / n - torch.outer(me, me)
    ev_, Ue = torch.linalg.eigh(Gec); ev_ = ev_.clamp_min(1e-12)
    Ceh = ((Ue * ev_.sqrt()) @ Ue.T).float(); Ceih = ((Ue * ev_.rsqrt()) @ Ue.T).float()
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
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
    _, SUsh = torch.linalg.eigh(comp[16]["S"] + comp[17]["S"]); SUsh = SUsh.flip(1); Ush = SUsh[:, :RSQ]   # u_1..u_8 in core coords
    Q = P @ Ush   # stream directions q_j (D x 8), orthonormal
    # fit-set statistics of the core coordinates of each block's rms-normed input
    cs = {l: {"s": torch.zeros(KM, dtype=torch.float64, device=DEV), "ss": torch.zeros(KM, KM, dtype=torch.float64, device=DEV), "n": 0} for l in (16, 17)}
    for idx, W_, X_ in collect_pass(m, fit_rows, LAST2):
        for l in (16, 17):
            c = (X_[("mlp", l)] @ P).double(); cs[l]["s"] += c.sum(0); cs[l]["ss"] += c.T @ c; cs[l]["n"] += c.shape[0]
    cmean = {}; cvar = {}; stats = {}
    for l in (16, 17):
        mu_c = cs[l]["s"] / cs[l]["n"]; C_c = cs[l]["ss"] / cs[l]["n"] - torch.outer(mu_c, mu_c)
        cmean[l] = mu_c.float(); cvar[l] = C_c.diag().float()
        mq = (Ush.double().T @ mu_c); vq = torch.einsum("ij,jk,ki->i", Ush.double().T, C_c, Ush.double())
        stats[l] = {"mean_q": mq.tolist(), "std_q": vq.sqrt().tolist(), "mean_dominance_q": (mq ** 2 / vq).tolist(),
                    "core_mean_sq_over_var": float((mu_c ** 2).sum() / torch.trace(C_c))}
    mq_l = {l: (Ush.T @ cmean[l]) for l in (16, 17)}   # mean of x^.q_j per block (float)
    log(stage="stats", **{f"b{l}": {"mean_q": [round(v, 3) for v in stats[l]["mean_q"][:NTOP]], "std_q": [round(v, 3) for v in stats[l]["std_q"][:NTOP]]} for l in (16, 17)})
    class RealAbl:
        """Real block; on its rms-normed input the coordinates along the columns of Qs (orthonormal, stream coords) are replaced:
        mode 'zero' -> 0; 'pin' -> their fit-set means (variation removed, mean kept); 'meanonly' -> mean subtracted (variation kept)."""
        def __init__(self, l, Qs, mode, means=None):
            self.mlp = m.transformer.h[l].mlp; self.Qs = Qs; self.mode = mode; self.means = means
        def hook(self, idx):
            pass
        def __call__(self, w, x):
            xh = F.rms_norm(x, (D,)); cq = xh @ self.Qs
            if self.mode == "zero": xp_ = xh - cq @ self.Qs.T
            elif self.mode == "pin": xp_ = xh + (self.means - cq) @ self.Qs.T
            else: xp_ = xh - self.means @ self.Qs.T
            return self.mlp.Down(self.mlp.Left(xp_) * self.mlp.Right(xp_))
    class Prog:
        def __init__(self, cb, pin_u=None, pin_val=None):
            self.A, self.b0, self.B, self.me_, self.mu = cb["A"], cb["b0"], cb["B"], cb["me"], cb["mu"]; self.pmu = self.mu @ P
            self.pin_u, self.pin_val = pin_u, pin_val
        def hook(self, idx):
            self.idx = idx
        def __call__(self, w, x):
            B_, T, _ = w.shape
            c = F.rms_norm(x, (D,)) @ P
            if self.pin_u is not None:
                c = c + torch.outer((self.pin_val - c @ self.pin_u).reshape(-1), self.pin_u).view(B_, T, KM)
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
    def real_part(Um, mode):   # Um: KM x r in core coords; Qs = P @ Um in stream coords; means per block = Um^T cmean_l
        Qs = P @ Um; return {("mlp", l): RealAbl(l, Qs, mode, Um.T @ cmean[l]) for l in (16, 17)}
    def prog_part(j=None):
        return {("mlp", l): Prog(comp[l], None if j is None else Ush[:, j], None if j is None else float(mq_l[l][j])) for l in (16, 17)}
    arms = {"REAL_ZERO_q1": run([real_part(Ush[:, :1], "zero")]), "REAL_MEANONLY_q1": run([real_part(Ush[:, :1], "meanonly")])}
    for j in range(NTOP):
        arms[f"REAL_MEANABL_q{j + 1}"] = run([real_part(Ush[:, j:j + 1], "pin")])
    arms["REAL_MEANABL_top5"] = run([real_part(Ush[:, :NTOP], "pin")]); arms["REAL_MEANABL_core16"] = run([real_part(torch.eye(KM, device=DEV), "pin")])
    rnd = []
    for sd in SEEDS:
        g = torch.Generator().manual_seed(sd); Rm = torch.linalg.qr(torch.randn(KM, NTOP, generator=g))[0].to(DEV)
        rnd.append(run([real_part(Rm, "pin")]))
    arms.update({f"REAL_MEANABL_rand5_s{sd}": v for sd, v in zip(SEEDS, rnd)})
    arms["PROG_SHARED8"] = run([prog_part()])
    for j in range(NTOP):
        arms[f"PROG_SHARED8_meanpin_u{j + 1}"] = run([prog_part(j)])
    log(stage="arms", **{k: round(v, 4) for k, v in arms.items()})
    rand_med = float(torch.tensor(rnd).median()); zero1 = arms["REAL_ZERO_q1"]
    f1 = arms["REAL_MEANONLY_q1"] / zero1 if zero1 > 0 else float("inf"); g1 = arms["REAL_MEANABL_q1"] / zero1 if zero1 > 0 else float("inf")
    e_ratio = arms["REAL_MEANABL_top5"] / rand_med if rand_med > 0 else float("inf")
    dom1 = [stats[l]["mean_dominance_q"][0] for l in (16, 17)]
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(arms["PROG_SHARED8"] - PRIOR_PROG) <= BARS["repro_tol"] and abs(zero1 - PRIOR_ZERO_Q1) <= BARS["anchor_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_constant_carries_most': bool(arms["REAL_MEANABL_q1"] <= BARS["b_max"]),
        'pred_c_q1_is_mean_dominated': bool(min(dom1) >= BARS["c_min"]),
        'pred_d_q1_still_informative': bool(arms["REAL_MEANABL_q1"] >= BARS["d_min"]),
        'pred_e_five_beat_random_mean_preserved': bool(e_ratio >= BARS["e_min"]),
    }
    nulls = {"b_null_meanabl_q1_ge_1.6": bool(arms["REAL_MEANABL_q1"] >= NULLS["b_min"]), "c_null_dominance_le_1_either_block": bool(min(dom1) <= NULLS["c_max"]),
             "d_null_meanabl_q1_le_.03": bool(arms["REAL_MEANABL_q1"] <= NULLS["d_max"]), "e_null_ratio_le_1.2": bool(e_ratio <= NULLS["e_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "square_space_rank": RSQ, "token_read_rank": RB, "n_top": NTOP,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "prog_shared8": arms["PROG_SHARED8"], "prior_prog": PRIOR_PROG, "real_zero_q1": zero1, "prior_zero_q1": PRIOR_ZERO_Q1},
           "ce_added": arms, "coord_stats_fit": {str(l): stats[l] for l in (16, 17)}, "mean_fraction_q1": f1, "info_fraction_q1": g1, "mean_dominance_q1": dom1,
           "random5_median": rand_med, "top5_over_random": e_ratio,
           "price": {"gpu_doc_forwards": 3 * int(fit_rows.shape[0]) + int(ev.shape[0]) * (1 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "ce_added", "mean_fraction_q1", "info_fraction_q1", "mean_dominance_q1", "top5_over_random", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "ce_added": arms, "mean_fraction_q1": f1, "info_fraction_q1": g1, "mean_dominance_q1": dom1}, indent=1))


if __name__ == "__main__":
    main()
