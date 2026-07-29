"""Layer-6 MLP explicit program (bottom-up ladder; same TN-native family that cracked MLP0/1):
out ~ TokenTable[tok] + PrevTable[prev] + sum_r u_r (a_r . hin)^2, arms R in {128, 256}.
Substitution audit on the ledger subset vs this interface's mean-input floor; induction-advantage
retention checked (natural + shuffled) so the program is verified not to break verified functions.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
WANT = 6
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
LED = json.load(open(f'{QK}/qk_completeness_ledger.json'))
SUBBASE = LED['subset_base']; FLOOR = LED['mlp_floor'][str(WANT)]


@torch.no_grad()
def pairs(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(WANT + 1):
        blk = m.transformer.h[li]; a = blk.attn
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
        if li == WANT:
            prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
            return hin.reshape(-1, D), blk.mlp(hin).reshape(-1, D), idx.reshape(-1), prv.reshape(-1)
        x = x + blk.mlp(hin)

Hs, Ys, Ts, Ps = [], [], [], []
for i in range(0, 400, 8):
    h, y, t, p = pairs(COOC[i:i+8].to(DEV)[:, :128]); Hs.append(h); Ys.append(y); Ts.append(t); Ps.append(p)
H = torch.cat(Hs); Y = torch.cat(Ys); TOK = torch.cat(Ts); PRV = torch.cat(Ps)
def cond_table(keys, target):
    ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
    ts.index_add_(0, keys, target); tc.index_add_(0, keys, torch.ones_like(keys, dtype=torch.float32))
    lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0)
    return lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*target.mean(0)
TT = cond_table(TOK, Y); PT = cond_table(PRV, Y - TT[TOK])
RES = Y - TT[TOK] - PT[PRV]

def fit(R, steps=2800):
    A = torch.nn.Parameter(torch.randn(R, D, device=DEV)*0.02); U = torch.nn.Parameter(torch.randn(R, D, device=DEV)*0.02)
    opt = torch.optim.Adam([A, U], lr=3e-3); n = H.shape[0]
    for _ in range(steps):
        ii = torch.randint(0, n, (8192,), device=DEV)
        loss = (((H[ii] @ A.T)**2) @ U - RES[ii]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fvu = float((((H @ A.T)**2) @ U - RES).pow(2).sum() / (RES - RES.mean(0)).pow(2).sum())
    return A.detach(), U.detach(), fvu

ARMS = {}
for name, R in [('table_prev', 0), ('table_prev_R128', 128), ('table_prev_R256', 256)]:
    if R == 0: ARMS[name] = (None, None); continue
    A, U, fvu = fit(R); ARMS[name] = (A, U); print(f"{name}: FVU {fvu:.3f}", flush=True)
torch.save({k: v for k, v in ARMS.items() if v[0] is not None}, f'{QK}/qk_mlp6_program.pt')


def forward(idx, arm):
    B, T2 = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
    for li in range(NL):
        blk = m.transformer.h[li]; a = blk.attn
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T2, -1)); hin = F.rms_norm(x, (D,))
        if li == WANT and arm is not None:
            flat = hin.reshape(-1, D)
            mo = TT[idx.reshape(-1)] + PT[prv.reshape(-1)]
            A, U = ARMS[arm]
            if A is not None: mo = mo + ((flat @ A.T)**2) @ U
            x = x + mo.view(B, T2, D).to(x.dtype)
        else:
            x = x + blk.mlp(hin)
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def audit(arm):
    tot = 0.0; n = 0
    for i in range(0, 200, 4):
        b = FINEWEB[i:i+4].to(DEV)
        lg = forward(b[:, :-1], arm)
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P]; EVN = torch.cat([prefN, prefN], 1).to(DEV)
g = torch.Generator().manual_seed(11); prefS = prefN.clone()
for r in range(NSEQ): prefS[r] = prefS[r][torch.randperm(P, generator=g)]
EVS = torch.cat([prefS, prefS], 1).to(DEV)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)
@torch.no_grad()
def adv(EV, arm):
    lg = forward(EV[:, :-1], arm)
    ce = F.cross_entropy(lg.reshape(-1, V), EV[:, 1:].reshape(-1), reduction='none').view(NSEQ, -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

advNf, advSf = adv(EVN, None), adv(EVS, None)
res = {'floor': FLOOR}
for arm in ['table_prev', 'table_prev_R128', 'table_prev_R256']:
    d = audit(arm) - SUBBASE
    an, as_ = adv(EVN, arm), adv(EVS, arm)
    res[arm] = {'dCE': round(d, 5), 'understood': round(1 - d/FLOOR, 3),
                'ind_nat': round(an/advNf, 3), 'ind_shuf': round(as_/advSf, 3)}
    print(f"MLP{WANT} {arm}: dCE +{d:.5f} ({1-d/FLOOR:.1%}) | induction nat {an/advNf:.1%} shuf {as_/advSf:.1%}", flush=True)
json.dump(res, open(f'{QK}/qk_mlp6_program.json', 'w'), indent=2)
print(f"QK MLP{WANT} PROGRAM DONE", flush=True)
