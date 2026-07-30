"""ITEM 4 control (final review): is the NATURAL high-amplitude leg AIMED (aim@p -> token@p) or a
fixed-vector saturation? Runs the §37g double dissociation in NATURAL text. Gate = trigger 447's natural
occurrences at position >= 10 (so source columns 1 and 5 are causally valid for every gated query); aim
the induction heads at column 1 vs column 5; measure P(token@1) and P(token@5) at scales 40/160.
Aimed-commandeering prediction: aim@1 raises token@1 not token@5; aim@5 the reverse. Fixed-vector
prediction: same token regardless of aim. Built on the working natural forward from qk_injection_specificity.
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
TRIGGER = 447; A1, A2 = 1, 5; SCALES = [40, 160]

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

SL = FINEWEB[64:320, :128].to(DEV)
idx = SL[:, :-1]; B0, T0 = idx.shape
MMg = match_matrix(idx)
pos = torch.arange(T0, device=DEV)[None, :].expand(B0, T0)
gate = (idx == TRIGGER) & (pos >= 10)                    # natural trigger occurrences, causal room for cols 1,5
NG = int(gate.sum())

@torch.no_grad()
def forward_aim(aim_col, scale):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T0, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T0, T0, device=DEV, dtype=torch.bool))
    MMr = torch.zeros_like(MMg); MMr[:, :, aim_col] = gate.float(); MMr = MMr * mask.float()
    DELTA = MMr - MMg * gate.unsqueeze(-1).float()
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B0, T0, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B0, T0, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        pat = torch.stack([pat[:, h] + (scale*AINIT[SIDX[(li, h)]]*DELTA if (li, h) in SIDX else 0.0) for h in range(NH)], 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B0, T0, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,))[gate])/30).float(); pr = lg.softmax(-1)
    t1 = idx[:, A1][:, None].expand(B0, T0)[gate]; t2 = idx[:, A2][:, None].expand(B0, T0)[gate]
    p1 = pr.gather(-1, t1[:, None]).squeeze(-1); p2 = pr.gather(-1, t2[:, None]).squeeze(-1)
    return {'P_tok@1': round(float(p1.mean()), 4), 'SE@1': round(float(p1.std()/max(NG,1)**0.5), 4),
            'P_tok@5': round(float(p2.mean()), 4), 'SE@5': round(float(p2.std()/max(NG,1)**0.5), 4)}

res = {'trigger': TRIGGER, 'n_gate': NG, 'sweep': {}}
for sc in SCALES:
    res['sweep'][f'scale={sc}'] = {'aim@col1': forward_aim(A1, sc), 'aim@col5': forward_aim(A2, sc)}
    print(f"scale {sc}:", res['sweep'][f'scale={sc}'], flush=True)
json.dump(res, open(f'{QK}/qk_natural_aimability.json', 'w'), indent=2)
print("QK NATURAL AIMABILITY DONE", flush=True)
