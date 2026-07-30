"""CONFIRMS THE SURVIVING CLAIM: is the planted §37 result (0.83/0.96 at scale 10) a GENUINE induction
repoint, or also brute-force? Runs the §37e specificity control in the PLANTED repeated-prefix setting.
Conditions (payload=token@1):
  A = induction heads, the planted second-copy trigger query (induction-ACTIVE, clean single-source)
  B = induction heads, the first-copy trigger position (NON-active: first occurrence, nothing to repoint)
  C = non-induction heads (matched amplitude), the active trigger query
Prediction if §37 is a genuine repoint: at scale 10, A ~0.83 while B ~0 and C ~0 (injection is too weak
at scale 10 to brute-force, as the natural B@10=0.0 showed) -- so the planted reach comes from
repointing the strong clean induction match, not from injection. If B climbs at scale 10 too, even §37
is brute-force and the whole capstone collapses.
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
NONIND = [(li, h) for li in range(2, 11) for h in range(NH) if (li, h) not in SIDX][:len(SUBST)]
NIDX = {lh: i for i, lh in enumerate(NONIND)}
P_C = 1; TRIG_POS = 20; SCALES = [10, 40, 160]

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P].clone()
TRIGGER = int(prefN[0, TRIG_POS]); prefN[:, TRIG_POS] = TRIGGER
EVN = torch.cat([prefN, prefN], 1).to(DEV)
idx = EVN[:, :-1]; B0, T0 = idx.shape
MMglobal = match_matrix(idx); active = MMglobal.sum(-1) > 0

# read-off AINIT on this planted eval (same as §37)
AINIT = torch.zeros(len(SUBST), device=DEV)
@torch.no_grad()
def read_a():
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T0, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T0, T0, device=DEV, dtype=torch.bool)); MM = MMglobal
    for li in range(11):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B0, T0, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B0, T0, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        for h in range(NH):
            if (li, h) in SIDX:
                Pt = pat[:, h]; Tm = Pt.mean(0); mb = mask.expand(B0, T0, T0)
                Xf = torch.stack([MM[mb], Tm.unsqueeze(0).expand(B0, T0, T0)[mb], torch.ones_like(MM[mb])], 1)
                AINIT[SIDX[(li, h)]] = torch.linalg.lstsq(Xf, Pt[mb].unsqueeze(1)).solution[0, 0]
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B0, T0, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
read_a()
AMEAN = float(AINIT.abs().mean())
payload = idx[:, P_C]

# gates: A = active trigger query (2nd copy), B = non-active trigger position (1st copy), C = A but non-ind heads
gA = torch.zeros(B0, T0, dtype=torch.bool, device=DEV); gA[:, P+TRIG_POS] = True; gA &= active
gB = torch.zeros(B0, T0, dtype=torch.bool, device=DEV); gB[:, TRIG_POS] = True; gB &= ~active

@torch.no_grad()
def forward_reach(gate, where, scale):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T0, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T0, T0, device=DEV, dtype=torch.bool))
    MMr = torch.zeros_like(MMglobal); MMr[:, :, P_C] = gate.float(); MMr = MMr * mask.float()
    DELTA = MMr - MMglobal * gate.unsqueeze(-1).float()
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B0, T0, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B0, T0, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        pats = []
        for h in range(NH):
            if where == 'ind' and (li, h) in SIDX: pats.append(pat[:, h] + scale*AINIT[SIDX[(li, h)]]*DELTA)
            elif where == 'nonind' and (li, h) in NIDX: pats.append(pat[:, h] + scale*AMEAN*DELTA)
            else: pats.append(pat[:, h])
        pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B0, T0, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    r = F.rms_norm(x, (D,))
    lg = 30*torch.tanh(m.lm_head(r[gate])/30).float()
    pay = payload[:, None].expand(B0, T0)[gate]
    pr = lg.softmax(-1); n = int(gate.sum())
    return {'n': n, 'P_payload': round(float(pr.gather(-1, pay[:, None]).mean()), 4),
            'argmax_capture': round(float((lg.argmax(-1) == pay).float().mean()), 4)}

CONDS = {'A_ind_active': (gA, 'ind'), 'B_ind_nonactive': (gB, 'ind'), 'C_nonind_active': (gA, 'nonind')}
res = {'TRIGGER': TRIGGER, 'conditions': {}}
for nm, (g, w) in CONDS.items():
    res['conditions'][nm] = {f'scale={sc}': forward_reach(g, w, sc) for sc in SCALES}
    print(nm, res['conditions'][nm], flush=True)
json.dump(res, open(f'{QK}/qk_planted_specificity.json', 'w'), indent=2)
print("QK PLANTED SPECIFICITY DONE", flush=True)
