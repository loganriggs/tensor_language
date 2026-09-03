#!/usr/bin/env python
"""late_mlp_subset_lattice_probe -- the full 2^5 mean-ablation lattice over mlp11..15, with mlp16/17 intact and with mlp16/17 mean-ablated
(64 arms): is the late stack's superadditive ~1 nat (§2719) redundancy among mlp11-15 or an interaction between them and mlp16/17? CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_MLP_SUBSET_LATTICE_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_mlp_subset_lattice_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_mlp_subset_lattice_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_MLP_SUBSET_LATTICE_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_stack_token_lookup_map_probe_results.json"   # §2719
OUT = ROOT / "late_mlp_subset_lattice_probe_results.json"
HASHES = {PREREG: "a60d5f4a1f112cb7e9db91583b04a39278ce1989bd076104536e3fdc7ab4aa94", PRIOR: "c0aaa202346dde9753d34fbac0873ec234abf0d4d6f5618631a8f36f5920e4bd",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_mlp_subset_lattice_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_MEAN2 = 0.848; PRIOR_MEAN7 = 1.885; LAM = 1e-2   # LAM unused here (inherited ridge helpers)
PRIOR_SINGLE = {11: 0.0474, 12: 0.0436, 13: 0.0403, 14: 0.0326, 15: 0.0386}   # §2719
FIVE = [11, 12, 13, 14, 15]
BARS = {"ce_tol": 1e-4, "repro_tol": 0.02, "repro_tol7": 0.03, "b_min": 0.50, "b_mult": 2.0, "c_max": 0.40, "d_mult": 3.0, "e_max": 0.02}
NULLS = {"b_max": 0.15, "c_min": 0.80, "d_mult": 1.5, "e_min": 0.05}


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
    c = xh @ Um; q = (c[:, IU_[0]] * c[:, IU_[1]])
    return torch.cat([e, c, q], 1)          # COREIN: 1152 + 16 + 136


class Surrogate:
    """w -> mu + P (A phi) with phi built from the token (hook) and the pre-write residual x."""
    def __init__(self, m, mode, fit, Pm, Um):
        self.m, self.mode, self.Pm, self.Um = m, mode, Pm.to(DEV).float(), Um.to(DEV).float()
        self.A = fit["A"].to(DEV).float(); self.mphi = fit["mphi"].to(DEV).float(); self.mu = fit["mu"].to(DEV).float(); self.e = None
    def hook(self, idx):
        self.e = emb(self.m, idx)
    def __call__(self, w, x):
        B, T, _ = w.shape
        phi = feats(self.mode, self.e.reshape(-1, D), F.rms_norm(x, (D,)).reshape(-1, D), self.Um)
        y = ((phi - self.mphi) @ self.A).view(B, T, D)
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
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    import itertools
    lat = {}
    for r in range(0, 6):
        for S in itertools.combinations(FIVE, r):
            for drop in (False, True):
                sites = [("mlp", l) for l in S] + ([("mlp", 16), ("mlp", 17)] if drop else [])
                key = "S=" + ",".join(map(str, S)) + ("|16,17" if drop else "")
                lat[key] = (ce_of(m, ev, {s_: meanpatch(bases[s_]["mu"]) for s_ in sites}) - ce0) if sites else 0.0
        log(stage="lattice", size=r, n=len(lat))
    def M(S, drop=False):
        return lat["S=" + ",".join(map(str, S)) + ("|16,17" if drop else "")]
    single = {l: M((l,)) for l in FIVE}; five = M(tuple(FIVE)); two = M((), True); seven = M(tuple(FIVE), True)
    within5 = five - sum(single.values()); between = seven - five - two
    pairs = {f"{i},{j}": M((i, j)) - single[i] - single[j] for i, j in itertools.combinations(FIVE, 2)}
    marg = {l: M((l,), True) - two for l in FIVE}
    med = lambda xs: sorted(xs)[len(xs) // 2]
    med_single = med(list(single.values())); med_marg = med(list(marg.values())); med_pair = med(list(pairs.values()))
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(two - PRIOR_MEAN2) <= BARS["repro_tol"] and abs(seven - PRIOR_MEAN7) <= BARS["repro_tol7"]
                   and all(abs(single[l] - PRIOR_SINGLE[l]) <= BARS["repro_tol"] for l in FIVE)) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok),
        'pred_b_between_group_interaction_dominates': bool(between >= BARS["b_min"] and between >= BARS["b_mult"] * within5),
        'pred_c_five_jointly_cheap_with_16_17_intact': bool(five <= BARS["c_max"]),
        'pred_d_marginals_grow_without_16_17': bool(med_marg >= BARS["d_mult"] * med_single),
        'pred_e_pairwise_interactions_small': bool(med_pair <= BARS["e_max"]),
    }
    nulls = {"b_null_between_le_.15": bool(between <= NULLS["b_max"]), "c_null_five_ge_.80": bool(five >= NULLS["c_min"]),
             "d_null_marg_le_1.5x": bool(med_marg <= NULLS["d_mult"] * med_single), "e_null_pair_ge_.05": bool(med_pair >= NULLS["e_min"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); interactions = CE(joint) - sum of parts (positive = superadditive)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda",
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "mean_16_17": two, "prior_mean_16_17": PRIOR_MEAN2, "mean_all7": seven, "prior_mean_all7": PRIOR_MEAN7, "singles": single, "prior_singles": PRIOR_SINGLE},
           "lattice": lat, "five_jointly_16_17_intact": five, "within5": within5, "between": between, "pairs": pairs, "marginal_without_16_17": marg,
           "medians": {"single": med_single, "marginal_without_16_17": med_marg, "pair": med_pair},
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * 65, "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "within5", "between", "medians", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "five": five, "within5": within5, "between": between, "medians": med_pair and {"single": med_single, "marg": med_marg, "pair": med_pair}}, indent=1))


if __name__ == "__main__":
    main()
