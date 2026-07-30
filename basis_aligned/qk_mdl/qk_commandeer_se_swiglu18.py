"""ARCHITECTURE-GENERALITY of the editing capstone: does copy-head COMMANDEERING replicate on swiglu18
(a FOURTH family -- softmax + swiglu MLP, 18L/9H twin of bilin18)? Because
bilin12's pattern is a normalized distribution over keys, commandeering is direct: overwrite a copy
head's attention row to a one-hot at a chosen source column, and the head copies v[source]. Target the
census-identified bilin12 MATCH_same (induction-ish) heads [(2,1),(2,3)]. Planted repeated prefix; gate
on a non-active query; aimability double dissociation (aim@col1 vs aim@col10) -> if it copies the
pointed source's token in bilin12 too, the commandeering primitive is architecture-general.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('swiglu18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
CENSUS2 = [(2,1),(2,3)]  # census bilin12 MATCH_same heads
ALL_L2 = [(li,h) for li in range(2,NL) for h in range(NH)]
ALL = [(li,h) for li in range(NL) for h in range(NH)]
A1, A2 = 1, 10; TRIG_POS = 20

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P].clone()
EVN = torch.cat([prefN, prefN], 1).to(DEV)
idx = EVN[:, :-1]; B0, T0 = idx.shape
active = match_matrix(idx).sum(-1) > 0
gate = torch.zeros(B0, T0, dtype=torch.bool, device=DEV); gate[:, TRIG_POS] = True; gate &= ~active
NG = int(gate.sum())

@torch.no_grad()
def forward(aim_col, HEADS):
    """bilin12 forward (normalized squared attention); if aim_col is not None, overwrite COPY_HEADS'
    attention row at gated queries to one-hot at aim_col."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        b = m.transformer.h[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        if hasattr(a, 'lamb'): v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
        pat = F.softmax(sc, -1)
        if aim_col is not None:
            oneh = torch.zeros(T, device=DEV); oneh[aim_col] = 1.0
            for (cl, ch) in HEADS:
                if cl == li:
                    ph = pat[:, ch].clone(); ph[gate] = oneh                     # overwrite gated-query rows
                    pat = pat.clone(); pat[:, ch] = ph
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + b.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def reach(aim_col, HEADS):
    lg = forward(aim_col, HEADS).float()[gate]; pr = lg.softmax(-1)
    t1 = idx[:, A1][:, None].expand(B0, T0)[gate]; t2 = idx[:, A2][:, None].expand(B0, T0)[gate]
    p1 = pr.gather(-1, t1[:, None]).squeeze(-1); p2 = pr.gather(-1, t2[:, None]).squeeze(-1); n = max(NG, 1)
    return {'P_tok@1': round(float(p1.mean()), 4), 'SE@1': round(float(p1.std()/n**0.5), 4),
            'P_tok@10': round(float(p2.mean()), 4), 'SE@10': round(float(p2.std()/n**0.5), 4)}

res = {'model': 'swiglu18', 'n_gate': NG, 'sets': {}}
for nm, HS in [('census2', CENSUS2), ('all_L2plus', ALL_L2), ('all_heads', ALL)]:
    res['sets'][nm] = {'n_heads': len(HS), 'no_edit': reach(None, HS), 'aim@col1': reach(A1, HS), 'aim@col10': reach(A2, HS)}
    print(nm, res['sets'][nm], flush=True)
json.dump(res, open(f'{QK}/qk_commandeer_se_swiglu18.json', 'w'), indent=2)
print("QK COMMANDEER SE SWIGLU18 DONE", flush=True)
