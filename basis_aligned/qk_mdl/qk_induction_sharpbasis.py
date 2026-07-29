"""Close the induction-symbol caveat: WHAT content does induction need that the 16-d symbol basis
dropped? Re-measure symbol-driven induction retention under richer bases:
  baseline  : rank-16 per-head PCA, no augment            (prior result: full-sym 64.3%, min-sym 27.8%)
  +prevtok  : add PREVIOUS-token embedding to codes (induction keys on the previous token)
  rank64    : per-head PCA rank 64 instead of 16
  both      : prevtok + rank64
If +prevtok recovers induction, the loss was token-identity carriage (mechanistic); if only rank64
helps, it was basis capacity. Headline metric = full-model symbol-driven pattern retention (layers
2..16); minimal-circuit retention for the winning config.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
wte = m.transformer.wte.weight.detach().float().to(DEV)
GENL = list(range(2, 17)); MAXL = 16

mh = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV); pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
PA = {}
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4; Dv = bb[f'h{hh}_v_Dm'].to(DEV); Dv = Dv / Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        Pp = mh[f'h{hh}']; Dn_ = Pp['Dm'].to(DEV); Dn_ = Dn_ / Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dn_[:, 2*HD:].T @ Pp['U'].to(DEV)[:, :16]
    if Vdir.shape[1] < 16: Vdir = torch.cat([Vdir, torch.zeros(HD, 16-Vdir.shape[1], device=DEV)], 1)
    PA[hh] = (Vdir / Vdir.norm(dim=0, keepdim=True).clamp_min(1e-9)).contiguous()
EPC96 = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0), full_matrices=False).Vh[:96].T.contiguous()
EW = F.rms_norm(wte, (D,)) @ EPC96   # (V,96) embedding codes per token


@torch.no_grad()
def run_cooc(idx):
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool)); yh = {}; xin = {}
    for li in range(MAXL+1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; xin[li] = x
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v); yh[li] = yh4
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return yh, xin

# accumulate per-head covariance once (max rank 64), derive rank-R bases by slicing
cov = {l: torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for l in range(MAXL)}
mu = {l: torch.zeros(NH, HD, device=DEV, dtype=torch.float64) for l in range(MAXL)}
nt = 0
for i in range(0, 256, 4):
    yh, _ = run_cooc(COOC[i:i+4].to(DEV)[:, :-1])
    for l in range(MAXL):
        f = yh[l].reshape(-1, NH, HD).double(); mu[l] += f.sum(0); cov[l] += torch.einsum('nhd,nhe->hde', f, f)
    nt += yh[0].reshape(-1, NH, HD).shape[0]
for l in range(MAXL):
    mu[l] /= nt; cov[l] = cov[l]/nt - torch.einsum('hd,he->hde', mu[l], mu[l])
EVEC = {}
for l in range(1, MAXL):
    EVEC[l] = {}
    for h in range(NH):
        ev, evec = torch.linalg.eigh(cov[l][h]); EVEC[l][h] = evec[:, ev.argsort(descending=True)[:64]].float().contiguous()
print('covariance ready', flush=True)

# induction eval + minimal circuit
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]; EV = torch.cat([pref, pref], 1).to(DEV)
SEC = torch.arange(P, 2*P-1, device=DEV); FIR = torch.arange(1, P-1, device=DEV)
KEEP = set(eval(c) for c in json.load(open(f'{QK}/qk_induction_minimal.json'))['minimal_components'])


def make_codes_fn(R, prevtok):
    PB = {l: {h: EVEC[l][h][:, :R] for h in range(NH)} for l in range(1, MAXL)}
    def codes(L, idx, yh):
        idxf = idx.reshape(-1); ce = EW[idxf]
        parts = [ce]
        if prevtok:
            prev = torch.roll(idx, 1, dims=1); prev[:, 0] = idx[:, 0]  # prev token per position
            parts.append(EW[prev.reshape(-1)])
        parts.append(torch.cat([(yh[0][..., h, :].reshape(-1, HD) - mu[0][h].float()) @ PA[h] for h in range(NH)], 1))
        for l in range(1, L):
            parts.append(torch.cat([(yh[l][..., h, :].reshape(-1, HD) - mu[l][h].float()) @ PB[l][h] for h in range(NH)], 1))
        return torch.cat(parts, 1)
    return codes


def fit_generators(codes_fn):
    W = {}
    for L in GENL:
        # discover dim from a probe
        b0 = COOC[:4].to(DEV)[:, :-1]; yh0, _ = run_cooc(b0); dim = codes_fn(L, b0, yh0).shape[1] + 1
        AtA = torch.zeros(dim, dim, device=DEV, dtype=torch.float64); AtY = torch.zeros(dim, D, device=DEV, dtype=torch.float64)
        for i in range(0, 384, 4):
            b = COOC[i:i+4].to(DEV)[:, :-1]; yh, xin = run_cooc(b)
            Cd = codes_fn(L, b, yh).double(); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV, dtype=torch.float64)], 1)
            Y = xin[L].reshape(-1, D).double(); AtA += Cd.T @ Cd; AtY += Cd.T @ Y
        W[L] = torch.linalg.solve(AtA + 10.0*torch.eye(dim, device=DEV, dtype=torch.float64), AtY).float().cpu()
        del AtA, AtY; torch.cuda.empty_cache()
    return W


@torch.no_grad()
def forward(codes_fn, W, keep, symbolic, collect_mean=False):
    idx = EV[:, :-1]; B, T = idx.shape
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); yh = {}; means = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if symbolic and li in GENL and keep is not None and any((('h', li, h) in keep) for h in range(NH)):
            Cd = codes_fn(li, idx, yh); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
            hsym = F.rms_norm((Cd @ W[li].to(DEV)).view(B, T, D), (D,)).to(hcur.dtype)
        else:
            hsym = hcur
        def rope(src): return lambda lin: apply_rot(F.rms_norm(lin(src).view(B, T, NH, HD), (HD,)), cosb, sinb)
        sr, qr = rope(hcur), rope(hsym)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        def patf(rp):
            q, k, q2, k2 = rp(a.c_q), rp(a.c_k), rp(a.c_q2), rp(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            return (s1*s2).masked_fill(~mask, 0.0)
        pat = patf(sr)
        if symbolic and li in GENL:
            ps = patf(qr)
            for h in range(NH):
                if keep is not None and ('h', li, h) in keep: pat[:, h] = ps[:, h]
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect_mean: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); yh[li] = yh4
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))
        if keep is not None and ('m', li) not in keep: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float(); tgt = EV[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(B, T)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item(), means

ALLSET = {('h', li, h) for li in range(NL) for h in range(NH)} | {('m', li) for li in range(NL)}
# clean means/full/none (use a trivial codes_fn; not used when symbolic=False)
dummy = make_codes_fn(16, False)
_, MEAN = forward(dummy, {}, None, False, True)
adv_full, _ = forward(dummy, {}, None, False)
adv_none, _ = forward(dummy, {}, set(), False)
DENOM = adv_full - adv_none
def ret(a): return (a - adv_none) / DENOM
print(f"full {adv_full:+.4f} none {adv_none:+.4f}", flush=True)

CONFIGS = [('baseline_R16', 16, False), ('prevtok_R16', 16, True), ('R64', 64, False), ('prevtok_R64', 64, True)]
res = {'adv_full': round(adv_full, 4), 'adv_none': round(adv_none, 4)}
for name, R, pv in CONFIGS:
    cf = make_codes_fn(R, pv); W = fit_generators(cf)
    a_full = forward(cf, W, ALLSET, True)[0]
    a_min = forward(cf, W, KEEP, True)[0]
    res[name] = {'full_symbolic': round(a_full, 4), 'full_ret': round(ret(a_full), 4),
                 'min_symbolic': round(a_min, 4), 'min_ret': round(ret(a_min), 4)}
    print(f"{name}: full-sym {a_full:+.4f} ({ret(a_full):.1%}) | min-sym {a_min:+.4f} ({ret(a_min):.1%})", flush=True)
    del W; torch.cuda.empty_cache()
json.dump(res, open(f'{QK}/qk_induction_sharpbasis.json', 'w'), indent=2)
print('QK INDUCTION SHARPBASIS DONE', flush=True)
