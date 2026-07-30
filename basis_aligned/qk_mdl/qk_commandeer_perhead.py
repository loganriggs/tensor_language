"""DECISIVE test of §37f's 'copy-head commandeering' claim: in condition B (induction heads pointed at a
source, NON-active query, NO natural match), does the head copy WHATEVER SOURCE we point it at (aim@p ->
token@p, a double dissociation) -- confirming it is the copy function being commandeered -- or does it
always emit one fixed token regardless of aim (-> fixed-direction injection, and §37f is wrong)?
Point condition-B redirect at column A1=1 vs A2=30; measure P(token@1) and P(token@30) under each aim.
Commandeered-copy prediction: aim@1 raises token@1 not token@30; aim@30 the reverse.
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
TRIG_POS = 20; A1, A2 = 1, 10; SCALES = [10, 40]  # both source cols BEFORE the query (causal); query at pos 20

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

QPOS_SWEEP = [20, 35, 50]

@torch.no_grad()
def reach_aim(aim_col, scale, gB, only_head=None):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T0, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T0, T0, device=DEV, dtype=torch.bool))
    MMr = torch.zeros_like(MMglobal); MMr[:, :, aim_col] = gB.float(); MMr = MMr * mask.float()
    DELTA = MMr - MMglobal * gB.unsqueeze(-1).float()
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B0, T0, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B0, T0, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        pat = torch.stack([pat[:, h] + (scale*AINIT[SIDX[(li, h)]]*DELTA if ((li, h) in SIDX and (only_head is None or (li,h)==only_head)) else 0.0) for h in range(NH)], 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B0, T0, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,))[gB])/30).float(); pr = lg.softmax(-1)
    t1 = idx[:, A1][:, None].expand(B0, T0)[gB]; t2 = idx[:, A2][:, None].expand(B0, T0)[gB]
    p1 = pr.gather(-1, t1[:, None]).squeeze(-1); p2 = pr.gather(-1, t2[:, None]).squeeze(-1); n = int(gB.sum())
    return {'P_tok@A1': round(float(p1.mean()), 4), 'SE@A1': round(float(p1.std()/max(n,1)**0.5), 4),
            'P_tok@A2': round(float(p2.mean()), 4), 'SE@A2': round(float(p2.std()/max(n,1)**0.5), 4)}

gB = torch.zeros(B0, T0, dtype=torch.bool, device=DEV); gB[:, 35] = True; gB &= ~active
res = {'note': 'which single SUBST head carries commandeering (aim@1, qpos35, scale40)', 'n': int(gB.sum()), 'all_heads': None, 'per_head': {}}
res['all_heads'] = reach_aim(A1, 40, gB, None)['P_tok@A1']
for lh in SUBST:
    res['per_head'][str(lh)] = reach_aim(A1, 40, gB, lh)['P_tok@A1']
top = sorted(res['per_head'].items(), key=lambda kv: -kv[1])[:6]
print('all-heads P_tok@1:', res['all_heads'], flush=True)
print('top single heads:', top, flush=True)
json.dump(res, open(f'{QK}/qk_commandeer_perhead.json', 'w'), indent=2)
print("QK COMMANDEER PERHEAD DONE", flush=True)
