"""TRIGGER-FREQUENCY SWEEP -- quantifies §37's main caveat: how does conditional-redirect collateral
scale with how often the trigger token appears? Picks trigger tokens spanning rare->common base rates
in the natural slice, and for each measures (a) REACH at the planted clean trigger query (should stay
high -- reach is a property of the clean match, not the token frequency) and (b) COLLATERAL on natural
text (should rise roughly with base_rate x induction_rate). Establishes that a conditional redirect's
cost is bounded by trigger frequency and always << the unconditional edit. Method = §37's scaled linear
repoint (x10) on the census induction heads, gated on the trigger token.
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
P_C = 1; TRIG_POS = 20; SCALE = 10.0

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefBase = FINEWEB[:NSEQ, 1:1+P].clone()               # unplanted, for a trigger-independent read-off
COLL = FINEWEB[64:128, :128].to(DEV)                    # natural collateral slice

# read-off AINIT once on unplanted repeated eval
EVbase = torch.cat([prefBase, prefBase], 1).to(DEV)
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
def forward(idx, trigger):
    """trigger=None: model. else: conditional scaled-linear redirect gated on `trigger` token id."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    if trigger is not None:
        MMn = match_matrix(idx); active = MMn.sum(-1) > 0
        gate = active & (idx == trigger)
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
        if trigger is not None:
            pat = torch.stack([pat[:, h] + (SCALE*AINIT[SIDX[(li, h)]]*DELTA if (li, h) in SIDX else 0.0) for h in range(NH)], 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# choose triggers spanning base-rate range from the collateral slice
counts = torch.bincount(COLL.reshape(-1).cpu(), minlength=V).float()
rate = counts / counts.sum()
present = torch.where(counts > 0)[0]
order = present[rate[present].argsort()]                   # ascending frequency
picks = {'rare': order[max(len(order)//20, 1)], 'low': order[len(order)//4],
         'mid': order[len(order)//2], 'high': order[int(len(order)*0.9)], 'common': order[-2]}
base_coll = F.cross_entropy(forward(COLL[:, :-1], None).float().reshape(-1, V), COLL[:, 1:].reshape(-1)).item()

@torch.no_grad()
def eval_trigger(tk):
    tk = int(tk)
    # reach on planted eval
    pref = prefBase.clone(); pref[:, TRIG_POS] = tk; EV = torch.cat([pref, pref], 1).to(DEV)
    idx = EV[:, :-1]; SEC = torch.arange(P, 2*P-1, device=DEV)
    C = idx[:, P_C]; tn = EV[:, SEC+1]
    trigm = (idx[:, SEC] == tk) & (match_matrix(idx).sum(-1)[:, SEC] > 0)
    sub = forward(idx, tk).float().softmax(-1)[:, SEC]
    pC = sub.gather(-1, C[:, None, None].expand(-1, SEC.numel(), 1)).squeeze(-1)
    reach = round(float(pC[trigm].mean()), 4) if trigm.any() else None
    argm = sub.argmax(-1); cap = round(float((argm == C[:, None])[trigm].float().mean()), 4) if trigm.any() else None
    # collateral on natural slice
    ce = F.cross_entropy(forward(COLL[:, :-1], tk).float().reshape(-1, V), COLL[:, 1:].reshape(-1)).item()
    return {'token': tk, 'base_rate': round(float(rate[tk]), 6), 'reach_P_chosen': reach,
            'reach_argmax_capture': cap, 'collateral_dCE': round(ce - base_coll, 4)}

res = {'model_coll_CE': round(base_coll, 4), 'sweep': {k: eval_trigger(v) for k, v in picks.items()}}
print("FREQ SWEEP:", json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_redirect_freq_sweep.json', 'w'), indent=2)
print("QK REDIRECT FREQ SWEEP DONE", flush=True)
