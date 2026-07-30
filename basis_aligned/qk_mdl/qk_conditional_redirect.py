"""CONDITIONAL (trigger-gated) REDIRECT -- the precision follow-up to §36.
§36's unconditional redirect hit EVERY induction query, so it cost +0.316-0.588 natural-CE collateral
(it damaged the genuine induction natural text uses). The honest precision primitive is a redirect
CONDITIONED on a trigger token: "at queries whose current token == TRIGGER (and only those), repoint
the induction match to a chosen source column P_C; leave all other induction untouched." Uses the
now-headline method from the §36 correction -- the SCALED LINEAR repoint on the named match channel
(scale x10), NOT the hard overwrite. Predictions: (1) REACH at trigger queries -- chosen token rises,
true-next collapses; (2) SPECIFICITY -- induction at NON-trigger queries is preserved (~model); (3)
COLLATERAL on natural text is MUCH smaller than the unconditional edit (only the trigger token's
induction is perturbed). Edit touches ONLY the census induction heads' match channel.
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
P_C = 1        # source column: redirect copies the token at position 1
TRIG_POS = 20  # trigger planted at this prefix position; TRIGGER token = a real, moderately common token
SCALE = 10.0   # scaled linear repoint (the §36-corrected principled method)

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P].clone()
TRIGGER = int(prefN[0, TRIG_POS])                 # fix a real token id as the trigger
prefN[:, TRIG_POS] = TRIGGER                       # plant one guaranteed trigger occurrence per sequence
EVN = torch.cat([prefN, prefN], 1).to(DEV)
SEC = torch.arange(P, 2*P-1, device=DEV)           # second-copy induction queries
TRIGQ = P + TRIG_POS                                # the guaranteed second-copy trigger query position

# --- read off a_readoff per head (identical to §36 / the dial) ---
AINIT = torch.zeros(len(SUBST), device=DEV)
@torch.no_grad()
def read_a():
    idx = EVN[:, :-1]; B, T = idx.shape
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
read_a()
print(f"TRIGGER token id {TRIGGER}; a_readoff ready", flush=True)

@torch.no_grad()
def forward(idx, edit):
    """edit: None | 'cond' (redirect only at TRIGGER queries) | 'uncond' (all induction queries)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    if edit in ('cond', 'uncond'):
        MMn = match_matrix(idx)
        active = MMn.sum(-1) > 0                                    # (B,T) induction-active queries
        gate = active & (idx == TRIGGER) if edit == 'cond' else active
        MMr = torch.zeros_like(MMn); MMr[:, :, P_C] = gate.float()
        MMr = MMr * mask.float()
        DELTA = MMr - MMn * gate.unsqueeze(-1).float()             # only touch gated query rows
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if edit in ('cond', 'uncond'):
            pats = []
            for h in range(NH):
                if (li, h) in SIDX: pats.append(pat[:, h] + SCALE * AINIT[SIDX[(li, h)]] * DELTA)
                else: pats.append(pat[:, h])
            pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def reach_and_specificity():
    idx = EVN[:, :-1]
    C = idx[:, P_C]; true_next = EVN[:, SEC+1]
    trig_mask = (idx[:, SEC] == TRIGGER) & (match_matrix(idx)[:, :, :].sum(-1)[:, SEC] > 0)  # (B,|SEC|) trigger induction queries
    out = {'n_trigger_queries': int(trig_mask.sum()), 'n_nontrigger_active': int(((match_matrix(idx).sum(-1)[:, SEC] > 0) & ~trig_mask).sum())}
    nontrig = (match_matrix(idx).sum(-1)[:, SEC] > 0) & ~trig_mask
    for nm, edit in [('model', None), ('cond', 'cond')]:
        sub = forward(idx, edit).float().softmax(-1)[:, SEC]       # (B,|SEC|,V)
        pC = sub.gather(-1, C[:, None, None].expand(-1, SEC.numel(), 1)).squeeze(-1)
        pT = sub.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)
        argm = sub.argmax(-1)
        out[nm] = {
            'REACH_trigger': {'P_chosen': round(float(pC[trig_mask].mean()), 4), 'P_true_next': round(float(pT[trig_mask].mean()), 4),
                              'argmax_is_chosen': round(float((argm == C[:, None])[trig_mask].float().mean()), 4),
                              'argmax_is_true': round(float((argm == true_next)[trig_mask].float().mean()), 4)},
            'SPECIFICITY_nontrigger': {'P_chosen': round(float(pC[nontrig].mean()), 4), 'P_true_next': round(float(pT[nontrig].mean()), 4),
                                       'argmax_is_true': round(float((argm == true_next)[nontrig].float().mean()), 4)}}
    return out

@torch.no_grad()
def collateral():
    idxN = FINEWEB[64:128, :128].to(DEV)
    base = F.cross_entropy(forward(idxN[:, :-1], None).float().reshape(-1, V), idxN[:, 1:].reshape(-1)).item()
    trig_rate = float((idxN == TRIGGER).float().mean())
    o = {'model_CE': round(base, 4), 'trigger_base_rate': round(trig_rate, 5)}
    for nm, edit in [('cond', 'cond'), ('uncond', 'uncond')]:
        ce = F.cross_entropy(forward(idxN[:, :-1], edit).float().reshape(-1, V), idxN[:, 1:].reshape(-1)).item()
        o[f'{nm}_CE'] = round(ce, 4); o[f'collateral_dCE_{nm}'] = round(ce - base, 4)
    return o

res = {'reach_specificity': reach_and_specificity(), 'collateral': collateral(),
       'TRIGGER': TRIGGER, 'P_C': P_C, 'scale': SCALE, 'n_heads_edited': len(SUBST)}
print("REACH + SPECIFICITY:", json.dumps(res['reach_specificity'], indent=2), flush=True)
print("COLLATERAL (cond vs uncond):", json.dumps(res['collateral'], indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_conditional_redirect.json', 'w'), indent=2)
print("QK CONDITIONAL REDIRECT DONE", flush=True)
