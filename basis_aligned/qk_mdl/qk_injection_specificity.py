"""DECISIVE control for §37d (red-team F3): is the high-amplitude natural "reach recovery" a genuine
INDUCTION repoint, or generic BRUTE-FORCE logit injection? Compares, at matched amplitude, the redirect
fired in three places for the same moderate trigger 447 with payload = token@1:
  (A) INDUCTION heads, induction-ACTIVE trigger queries      -- the real edit (repoint should work)
  (B) INDUCTION heads, NON-active same-token queries         -- NO induction match to repoint
  (C) NON-induction heads (matched count & amplitude), active trigger queries
If reach in (B) or (C) climbs like (A), the recovery is generic injection of scale*A*v_payload, not
induction repointing, and §37d's 'in-the-wild capability' collapses. Clean-repoint prediction: only (A)
climbs; (B) and (C) stay near baseline. Memory-safe chunked gated lm_head.
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
# matched NON-induction head set: same layers, heads not in SUBST, matched count
NONIND = [(li, h) for li in range(2, 11) for h in range(NH) if (li, h) not in SIDX][:len(SUBST)]
NIDX = {lh: i for i, lh in enumerate(NONIND)}
P_C = 1; TRIGGER = 447; SCALES = [10, 40, 160]

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
AMEAN = float(AINIT.abs().mean())   # reference amplitude for the non-induction-head control

@torch.no_grad()
def forward_resid(idx, gate, scale, where):
    """where: 'ind'=inject at SUBST heads (coeff AINIT) | 'nonind'=inject at NONIND heads (coeff AMEAN)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
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
        pats = []
        for h in range(NH):
            if where == 'ind' and (li, h) in SIDX:
                pats.append(pat[:, h] + scale*AINIT[SIDX[(li, h)]]*DELTA)
            elif where == 'nonind' and (li, h) in NIDX:
                pats.append(pat[:, h] + scale*AMEAN*DELTA)
            else:
                pats.append(pat[:, h])
        pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return F.rms_norm(x, (D,))

SL = FINEWEB[64:320, :128].to(DEV)
idxf = SL[:, :-1]; active_f = match_matrix(idxf).sum(-1) > 0
payload = idxf[:, P_C]
CONDS = {
    'A_ind_active':    (active_f & (idxf == TRIGGER), 'ind'),
    'B_ind_nonactive': ((~active_f) & (idxf == TRIGGER), 'ind'),
    'C_nonind_active': (active_f & (idxf == TRIGGER), 'nonind'),
}

@torch.no_grad()
def reach(gate, where, scale):
    outs = []
    for s in range(0, idxf.shape[0], 64):
        ci = idxf[s:s+64]; cg = gate[s:s+64]
        if cg.sum() == 0: continue
        r = forward_resid(ci, cg, scale, where)
        lg = 30*torch.tanh(m.lm_head(r[cg])/30).float()
        pay = payload[s:s+64][:, None].expand_as(ci)[cg]
        outs.append((lg.softmax(-1).gather(-1, pay[:, None]).squeeze(-1), (lg.argmax(-1) == pay).float()))
    P = torch.cat([o[0] for o in outs]); C = torch.cat([o[1] for o in outs]); n = P.numel()
    return {'n': int(n), 'P_payload': round(float(P.mean()), 4), 'P_SE': round(float(P.std()/np.sqrt(n)), 4),
            'argmax_capture': round(float(C.mean()), 4)}

res = {'trigger': TRIGGER, 'AMEAN': round(AMEAN, 5), 'conditions': {}}
for nm, (gate, where) in CONDS.items():
    res['conditions'][nm] = {'n_gate': int(gate.sum()), 'sweep': {f'scale={sc}': reach(gate, where, sc) for sc in SCALES}}
    print(nm, res['conditions'][nm]['sweep'], flush=True)
json.dump(res, open(f'{QK}/qk_injection_specificity.json', 'w'), indent=2)
print("QK INJECTION SPECIFICITY DONE", flush=True)
