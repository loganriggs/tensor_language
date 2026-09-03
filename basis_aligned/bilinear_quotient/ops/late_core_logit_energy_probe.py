#!/usr/bin/env python
"""late_core_logit_energy_probe -- how does lm_head read the late shared write core (§2710/§2713/§2714)? Logit-energy fraction of the
vocab-centred logits through P_M x_hat, activation fraction of x_hat in M, weight read-energy ratio of lm_head on M (vs X_16, early core,
random), plus clean mean-ablation references for mlp16/mlp17/late7 against the §2714 drop-core patch. CUDA lane-1 script.

# BQGATE: EXPERIMENT  pred_a_instrument pred_b_shared_dictionary_cheap pred_c_shared_512_converges pred_d_energy_overlap
#                     pred_e_adjacent_pairs_share

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 -- LOWER IS BETTER. Descriptive;
nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
Preregistration: polynomial_causal/LATE_CORE_LOGIT_ENERGY_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/late_core_logit_energy_probe.py")
sys.path.insert(0, str(SELF.parent))
import mlp_in_situ_usage_rank_map_probe as R

if not torch.cuda.is_available():
    raise RuntimeError("late_core_logit_energy_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "LATE_CORE_LOGIT_ENERGY_PROBE_PREREGISTRATION.md"
PRIOR = ROOT / "late_core_norm_channel_probe_results.json"   # §2714
OUT = ROOT / "late_core_logit_energy_probe_results.json"
HASHES = {PREREG: "f3fd361e13c17d103f1064323726d959c781dd1a16235559e36327f467812d9a", PRIOR: "60e663ce6e9d1a6acb1f11d2c66c5b8c551376002c621c69fdab12c962ecb4a0",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
          R.NAT: "666a32015c8ab3dcbabca4a859f5a0c8a3e1b9b9cc8f0b7f7c9e5211d903e2a1"}
RUNG = "late_core_logit_energy_probe"
D, NH, HD, NL, V = R.D, R.NH, R.HD, R.NL, R.V
TI = 256; FIT = (96, 192); EVAL = (0, 64); CH = 8
KM = 16
LATE7 = [("mlp", l) for l in range(11, 18)]; EARLY7 = [("mlp", l) for l in range(0, 7)]; LAST2 = [("mlp", 16), ("mlp", 17)]
PRIOR_BASE = 3.0322401; PRIOR_TW_EFF = 10.004; PRIOR_PLAIN16 = 6.1496
BARS = {"ce_tol": 1e-4, "eff_tol": 0.5, "repro_tol": 0.02, "er_rnd_lo": 0.9, "er_rnd_hi": 1.1, "q_rnd_max": 0.03, "b_min": 0.50, "c_min": 1.5, "d_ratio": 0.80, "e_min": 1.0}
NULLS = {"b_max": 0.20, "c_max": 1.1, "d_ratio": 1.0, "e_max": 0.30}


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
        mw = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat))
        x = x + apply(("mlp", l), mw, x) + mlp.Down_bias
    if collect is not None:
        collect(("final", -1), x)
    if collect is not None:
        collect(("final", -1), x)
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def ce_of(m, rows, patch=None):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], CH):
        idx = rows[i:i + CH, :TI]; tgt = rows[i:i + CH, 1:TI + 1]
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


def dropcore(U, k):
    Uk = U[:, :k].to(DEV).float()
    return lambda w, x: w - (w @ Uk) @ Uk.T


def meanpatch(b):
    mu = b["mu"].to(DEV)
    return lambda w, x: mu.expand_as(w)


def logit_energy(m, rows, subs, Wc):
    """Returns per-subspace {q: logit-energy fraction through P_U x_hat, p: activation fraction of x_hat in U} over all tokens."""
    num_q = {n: 0.0 for n in subs}; num_p = {n: 0.0 for n in subs}; den_q = 0.0; den_p = 0.0
    for i in range(0, rows.shape[0], CH):
        box = {}
        forward(m, rows[i:i + CH, :TI], collect=lambda s, w: box.__setitem__(s, w) if s == ("final", -1) else None)
        xh = F.rms_norm(box[("final", -1)], (D,)).reshape(-1, D)
        den_q += float(((xh @ Wc.T) ** 2).sum()); den_p += float((xh ** 2).sum())
        for n, U in subs.items():
            Uk = U[:, :KM].to(DEV).float(); pj = (xh @ Uk) @ Uk.T
            num_q[n] += float(((pj @ Wc.T) ** 2).sum()); num_p[n] += float((pj ** 2).sum())
    return {n: {"q": num_q[n] / den_q, "p": num_p[n] / den_p} for n in subs}


def er_lm(Wc, U):
    Uk = U[:, :KM].to(DEV).float(); return float((((Wc @ Uk) ** 2).sum() / KM) / ((Wc ** 2).sum() / D))


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
    bases = fit(m, fit_rows, LATE7 + EARLY7 + [("final", -1)]); log(stage="fit")
    ce0 = ce_of(m, ev); log(stage="baseline", ce=ce0)
    core = pooled([bases[s] for s in LATE7], False); early = pooled([bases[s] for s in EARLY7], False)
    g = torch.Generator().manual_seed(0); rnd, _ = torch.linalg.qr(torch.randn(D, KM, generator=g, dtype=torch.float64))
    subs = {"M": core["U"], "X": bases[("final", -1)]["U"].double(), "E": early["U"], "R": rnd}
    W = m.lm_head.weight.detach().float(); Wc = W - W.mean(0, keepdim=True)
    le = logit_energy(m, ev, subs, Wc); er = {n: er_lm(Wc, U) for n, U in subs.items()}; er_raw = {n: er_lm(W, U) for n, U in subs.items()}
    log(stage="logit_energy", le={n: {k: round(v, 4) for k, v in d.items()} for n, d in le.items()}, er={n: round(v, 3) for n, v in er.items()})
    arms = {"DROPCORE16_late7": {s: dropcore(core["U"], KM) for s in LATE7}, "DROPCORE16_last2": {s: dropcore(core["U"], KM) for s in LAST2},
            "MEAN_mlp16": {("mlp", 16): meanpatch(bases[("mlp", 16)])}, "MEAN_mlp17": {("mlp", 17): meanpatch(bases[("mlp", 17)])},
            "MEAN_last2": {s: meanpatch(bases[s]) for s in LAST2}, "MEAN_late7": {s: meanpatch(bases[s]) for s in LATE7}}
    ce = {}
    for n, p in arms.items():
        ce[n] = ce_of(m, ev, p) - ce0; log(stage="arm", arm=n, ce_added=round(ce[n], 4))
    d_ratio = ce["MEAN_last2"] / ce["DROPCORE16_last2"] if ce["DROPCORE16_last2"] > 0 else float("inf")
    inst_ok = bool(abs(ce0 - PRIOR_BASE) <= BARS["ce_tol"] and abs(core["spec"]["eff_rank"] - PRIOR_TW_EFF) <= BARS["eff_tol"] and abs(ce["DROPCORE16_late7"] - PRIOR_PLAIN16) <= BARS["repro_tol"]) if not smoke else True
    preds = {
        'pred_a_instrument': bool(inst_ok and BARS["er_rnd_lo"] <= er["R"] <= BARS["er_rnd_hi"] and le["R"]["q"] <= BARS["q_rnd_max"]),
        'pred_b_core_carries_logit_energy': bool(le["M"]["q"] >= BARS["b_min"]),
        'pred_c_lm_reads_core_above_isotropic': bool(er["M"] >= BARS["c_min"]),
        'pred_d_dropping_only_the_core_is_worse_than_mean_ablation': bool(d_ratio <= BARS["d_ratio"]),
        'pred_e_last_mlp_is_essential': bool(ce["MEAN_mlp17"] >= BARS["e_min"]),
    }
    nulls = {"b_null_q_le_.20": bool(le["M"]["q"] <= NULLS["b_max"]), "c_null_er_le_1.1": bool(er["M"] <= NULLS["c_max"]),
             "d_null_mean_ge_dropcore": bool(d_ratio >= NULLS["d_ratio"]), "e_null_mlp17_le_.30": bool(ce["MEAN_mlp17"] <= NULLS["e_max"])}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "CE numbers are CE ADDED above the real model on held-out docs 0-63 (FRESH split; LOWER IS BETTER); q/p are energy fractions and ER read-energy ratios (HIGHER = more read)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda", "k_core": KM,
           "instrument": {"baseline_ce": ce0, "prior_baseline": PRIOR_BASE, "core_tw_eff_rank": core["spec"]["eff_rank"], "prior_core_tw_eff_rank": PRIOR_TW_EFF,
                          "dropcore16_late7": ce["DROPCORE16_late7"], "prior_plain16": PRIOR_PLAIN16, "er_random": er["R"], "q_random": le["R"]["q"]},
           "logit_energy": le, "er_lm_centred": er, "er_lm_raw": er_raw, "ce_added": ce, "d_ratio_mean_over_dropcore_last2": d_ratio,
           "price": {"gpu_doc_forwards": int(fit_rows.shape[0]) + int(ev.shape[0]) * (2 + len(arms)), "cpu_doc_forwards": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "instrument", "logit_energy", "er_lm_centred", "ce_added", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "logit_energy": le, "er": er, "ce_added": ce}, indent=1))


if __name__ == "__main__":
    main()
