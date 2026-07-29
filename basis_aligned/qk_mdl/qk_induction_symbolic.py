"""SYNTHESIS: the minimal induction circuit expressed END-TO-END in preceding-layer symbols.
Fit per-layer symbol generators on the natural cooc corpus (layers 2..16). Then on the induction
eval, run a forward that (a) keeps ONLY the minimal 45-component circuit (mean-ablate the rest) and
(b) drives every kept head in layers >=2 by its SYMBOL-generated QK input (codes of preceding
layers), leaving layers 0-1 and kept MLPs raw. If the induction advantage survives, the circuit is
interpretable end-to-end: symbols -> pattern -> copy -> readout, not raw activations.
Baselines: minimal circuit with RAW inputs (the 90.7% reference) and symbol-driven FULL model.
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
GENL = list(range(2, 17))  # symbol generators for layers 2..16
MAXL = 16

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
EW = F.rms_norm(wte, (D,)) @ EPC96


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

# bases layers 1..MAXL-1
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
PB = {}
for l in range(1, MAXL):
    PB[l] = {}
    for h in range(NH):
        ev, evec = torch.linalg.eigh(cov[l][h]); PB[l][h] = evec[:, ev.argsort(descending=True)[:16]].float().contiguous()
print('bases ready', flush=True)


def codes(L, idx, yh):
    ce = EW[idx.reshape(-1)]
    c = [ce, torch.cat([(yh[0][..., h, :].reshape(-1, HD) - mu[0][h].float()) @ PA[h] for h in range(NH)], 1)]
    for l in range(1, L):
        c.append(torch.cat([(yh[l][..., h, :].reshape(-1, HD) - mu[l][h].float()) @ PB[l][h] for h in range(NH)], 1))
    return torch.cat(c, 1)

# fit generators on cooc (CPU offload)
W = {}
for L in GENL:
    dim = 96+144*L+1; AtA = torch.zeros(dim, dim, device=DEV, dtype=torch.float64); AtY = torch.zeros(dim, D, device=DEV, dtype=torch.float64)
    for i in range(0, 384, 4):
        b = COOC[i:i+4].to(DEV)[:, :-1]; yh, xin = run_cooc(b)
        Cd = codes(L, b, yh).double(); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV, dtype=torch.float64)], 1)
        Y = xin[L].reshape(-1, D).double(); AtA += Cd.T @ Cd; AtY += Cd.T @ Y
    W[L] = torch.linalg.solve(AtA + 10.0*torch.eye(dim, device=DEV, dtype=torch.float64), AtY).float().cpu()
    del AtA, AtY; torch.cuda.empty_cache()
print('generators fit', flush=True)

# induction eval + minimal circuit
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]; EV = torch.cat([pref, pref], 1).to(DEV)
SEC = torch.arange(P, 2*P-1, device=DEV); FIR = torch.arange(1, P-1, device=DEV)
MIN = json.load(open(f'{QK}/qk_induction_minimal.json'))['minimal_components']
KEEP = set(eval(c) for c in MIN)   # {('h',li,h) / ('m',li)}


@torch.no_grad()
def forward(keep=None, symbolic=False, collect_mean=False):
    idx = EV[:, :-1]; B, T = idx.shape
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); yh = {}; means = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        # symbolic QK source for this layer's kept heads (layers 2..16 with a generator)
        if symbolic and li in GENL and any((('h', li, h) in keep) for h in range(NH)):
            Cd = codes(li, idx, yh); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
            xh = (Cd @ W[li].to(DEV)).view(B, T, D); hsym = F.rms_norm(xh, (D,)).to(hcur.dtype)
        else:
            hsym = hcur
        def qk(src):
            return lambda lin: apply_rot(F.rms_norm(lin(src).view(B, T, NH, HD), (HD,)), cosb, sinb)
        qr, sr = qk(hsym), qk(hcur)  # rope helpers for symbolic vs raw
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        # compute per-head pattern; kept heads in GENL use symbolic source, others raw
        def pat_for(src_q):
            q, k, q2, k2 = src_q(a.c_q), src_q(a.c_k), src_q(a.c_q2), src_q(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            return (s1*s2).masked_fill(~mask, 0.0)
        pat_raw = pat_for(sr)
        if symbolic and li in GENL:
            pat_sym = pat_for(qr)
            pat = pat_raw.clone()
            for h in range(NH):
                if keep is not None and ('h', li, h) in keep: pat[:, h] = pat_sym[:, h]
        else:
            pat = pat_raw
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

_, MEAN = forward(None, False, True)
adv_full, _ = forward(None, False)
adv_none, _ = forward(set(), False)
def ret(a): return (a - adv_none) / (adv_full - adv_none)
ALLSET = {('h', li, h) for li in range(NL) for h in range(NH)} | {('m', li) for li in range(NL)}
adv_min_raw, _ = forward(KEEP, False)
adv_full_sym, _ = forward(ALLSET, True)   # nothing ablated; all layer-2..16 patterns symbol-driven
adv_min_sym, _ = forward(KEEP, True)
res = {'adv_full': round(adv_full, 4), 'adv_none': round(adv_none, 4),
       'minimal_raw': round(adv_min_raw, 4), 'minimal_raw_ret': round(ret(adv_min_raw), 4),
       'minimal_symbolic': round(adv_min_sym, 4), 'minimal_symbolic_ret': round(ret(adv_min_sym), 4),
       'full_symbolic_patterns': round(adv_full_sym, 4), 'full_symbolic_ret': round(ret(adv_full_sym), 4)}
for k, v in res.items(): print(f"  {k}: {v}", flush=True)
json.dump(res, open(f'{QK}/qk_induction_symbolic.json', 'w'), indent=2)
print('QK INDUCTION SYMBOLIC DONE', flush=True)
