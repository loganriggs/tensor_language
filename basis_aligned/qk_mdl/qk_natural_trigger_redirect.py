"""NATURAL-TRIGGER REDIRECT -- closes §37's last open caveat (red-team concerns 1 & 6). §37's reach
(0.833/0.958) was measured at a single PLANTED clean trigger query (one position, 48 replicates).
This fires the same conditional redirect on NATURALLY-OCCURRING trigger occurrences in real FineWeb
text -- many different positions, weaker/ambiguous matches -- and reports reach WITH standard errors
over the query set, for several trigger tokens of differing frequency. No planting: the trigger is a
real token wherever it naturally recurs with an induction match; the chosen payload is the real token
at position 1 of each sequence. Establishes the honest reach LOWER BOUND in the wild.
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
P_C = 1; SCALE = 10.0

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

# read-off AINIT on a standard repeated eval (trigger-independent), same as §37
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
def forward(idx, trigger):
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

# natural slice; pick triggers of differing frequency that actually occur with induction matches
SL = FINEWEB[64:192, :128].to(DEV)
idx = SL[:, :-1]; B, T = idx.shape
MMn = match_matrix(idx); active = MMn.sum(-1) > 0
counts = torch.bincount(idx[active].cpu(), minlength=V).float()   # counts among induction-active queries
cand = torch.where(counts >= 8)[0]                               # enough natural firings for an SE
rate_all = torch.bincount(idx.reshape(-1).cpu(), minlength=V).float() / idx.numel()
# choose 3 spanning available frequency (ascending), avoiding the very top (degenerate common tokens)
csort = cand[rate_all[cand].argsort()]
picks = {'distinctive': int(csort[max(len(csort)//10, 0)]), 'moderate': int(csort[len(csort)//2]),
         'frequent': int(csort[-1])}
payload = idx[:, P_C]                                             # chosen payload = real token at position 1

@torch.no_grad()
def natural_reach(tk):
    gate = active & (idx == tk)                                  # natural trigger induction queries
    n = int(gate.sum())
    if n == 0: return {'token': tk, 'n': 0}
    true_next = SL[:, 1:]                                         # actual next token at each position
    out = {'token': tk, 'base_rate': round(float(rate_all[tk]), 6), 'n_natural_queries': n,
           'n_distinct_positions': int(torch.unique(torch.where(gate)[1]).numel())}
    for nm, edit in [('model', None), ('cond', tk)]:
        pr = forward(idx, edit).float().softmax(-1)
        pay = pr.gather(-1, payload[:, None, None].expand(-1, T, 1)).squeeze(-1)[gate]     # P(payload) at trigger queries
        ptn = pr.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)[gate]                     # P(true-next)
        cap = (pr.argmax(-1) == payload[:, None])[gate].float()
        out[nm] = {'P_payload': round(float(pay.mean()), 4), 'P_payload_SE': round(float(pay.std()/np.sqrt(n)), 4),
                   'P_true_next': round(float(ptn.mean()), 4),
                   'argmax_is_payload': round(float(cap.mean()), 4), 'argmax_capture_SE': round(float(cap.std()/np.sqrt(n)), 4)}
    return out

res = {'triggers': {k: natural_reach(v) for k, v in picks.items()}}
print("NATURAL-TRIGGER REACH:", json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_natural_trigger_redirect.json', 'w'), indent=2)
print("QK NATURAL TRIGGER REDIRECT DONE", flush=True)
