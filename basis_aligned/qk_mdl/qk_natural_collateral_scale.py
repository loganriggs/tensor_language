"""Completes §37d: the REACH-vs-COLLATERAL tradeoff in the wild. §37d showed natural reach recovers
with amplitude (scale 10->160); this measures what that recovering amplitude COSTS. For the well-
powered moderate trigger (447), at each scale, fire the conditional redirect on natural text and report
natural-text cross-entropy change split three ways: (a) whole-slice dCE (overall cost), (b) NON-trigger
positions = pure collateral (should stay small, bounded by trigger frequency), (c) trigger positions =
the intended redirect cost. Memory-safe: chunked lm_head + CE. Pairs with §37d's reach curve to give
the honest in-the-wild operating curve.
"""
import json, sys, ast
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
MINCOMP = json.load(open(f'{QK}/qk_understanding_props.json'))['minimality']['locally_minimal_components']
SUBST = sorted({(li, h) for (t, li, h) in [ast.literal_eval(c) for c in MINCOMP if c.startswith("('h'")] if 2 <= li <= 10})
SIDX = {lh: i for i, lh in enumerate(SUBST)}
P_C = 1; TRIGGER = 447; SCALES = [0, 10, 20, 40, 80, 160]

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefBase = FINEWEB[:NSEQ, 1:1+P]; EVbase = torch.cat([prefBase, prefBase], 1).to(DEV)
AINIT = torch.zeros(len(SUBST), device=DEV)
@torch.no_grad()
def read_a(EV):
    idx = EV[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); MM = match_matrix(idx)
    for li in range(11):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        for h in range(NH):
            if (li, h) in SIDX:
                Pt = pat[:, h]; Tm = Pt.mean(0); mb = mask.expand(B, T, T)
                Xf = torch.stack([MM[mb], Tm.unsqueeze(0).expand(B, T, T)[mb], torch.ones_like(MM[mb])], 1)
                AINIT[SIDX[(li, h)]] = torch.linalg.lstsq(Xf, Pt[mb].unsqueeze(1)).solution[0, 0]
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
read_a(EVbase)

@torch.no_grad()
def forward_resid(idx, gate, scale):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    if gate is not None and scale != 0:
        MMn = match_matrix(idx)
        MMr = torch.zeros_like(MMn); MMr[:, :, P_C] = gate.float(); MMr = MMr * mask.float()
        DELTA = MMr - MMn * gate.unsqueeze(-1).float()
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if gate is not None and scale != 0:
            pat = torch.stack([pat[:, h] + (scale*AINIT[SIDX[(li, h)]]*DELTA if (li, h) in SIDX else 0.0) for h in range(NH)], 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return F.rms_norm(x, (D,))

SL = FINEWEB[64:320, :128].to(DEV)
idxf = SL[:, :-1]; tgt = SL[:, 1:]
active_f = match_matrix(idxf).sum(-1) > 0
gate = active_f & (idxf == TRIGGER)
n_fire = int(gate.sum())

@torch.no_grad()
def ce_split(scale):
    """per-token CE over the slice at `scale`, chunked; return (all, nontrigger, trigger) means."""
    ce_all = []
    for s in range(0, idxf.shape[0], 32):
        ci = idxf[s:s+32]; cg = gate[s:s+32]; ct = tgt[s:s+32]
        r = forward_resid(ci, cg, scale)
        lg = 30*torch.tanh(m.lm_head(r)/30).float()
        ce_all.append(F.cross_entropy(lg.reshape(-1, V), ct.reshape(-1), reduction='none').view(ci.shape))
    ce = torch.cat(ce_all, 0)                       # (B,T)
    return ce
ce0 = ce_split(0)
res = {'trigger': TRIGGER, 'n_fire': n_fire, 'base_CE_slice': round(float(ce0.mean()), 4), 'sweep': {}}
for sc in SCALES:
    if sc == 0: continue
    ce = ce_split(sc); d = ce - ce0
    res['sweep'][f'scale={sc}'] = {
        'whole_slice_dCE': round(float(d.mean()), 5),
        'nontrigger_dCE': round(float(d[~gate].mean()), 5),     # pure collateral
        'trigger_dCE': round(float(d[gate].mean()), 4)}         # intended redirect cost
print("REACH-vs-COLLATERAL (natural, trigger 447):", json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_natural_collateral_scale.json', 'w'), indent=2)
print("QK NATURAL COLLATERAL SCALE DONE", flush=True)
