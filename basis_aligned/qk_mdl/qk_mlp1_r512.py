"""MLP 1 explicit program (same TN-native method that cracked MLP 0) + the JOINT fully-explicit test.
MLP1 is also a bilinear MLP -> program: out ~ TokenTable[tok] + PrevTable[prev] + sum_r u_r (a_r.x)^2.
Arms: table; table+prev; table+prev+R64; table+prev+R256. Each verified TWO ways: natural dCE on the
ledger subset (floor 2.151) AND induction-advantage retention natural+shuffled (falsifiable: MLP1
carries the two-branch match service -- if the program misses it, induction collapses).
JOINT: substitute MLP0 program (table+R256, from qk_mlp0_interaction.pt) + MLP1 best program +
explicit induction patterns (72 finetuned scalars, qk_induction_finetune.pt) SIMULTANEOUSLY ->
the fully-explicit early stack, natural dCE + induction retention.
"""
import json, sys
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
AUD = FINEWEB[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
FLOOR1 = 2.15118
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']


@torch.no_grad()
def block1_pairs(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(2):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
        if li == 1:
            prev = torch.roll(idx, 1, dims=1); prev[:, 0] = idx[:, 0]
            return hin.reshape(-1, D), blk.mlp(hin).reshape(-1, D), idx.reshape(-1), prev.reshape(-1)
        x = x + blk.mlp(hin)

H, Y, TOK, PRV = [], [], [], []
for i in range(0, 400, 8):
    h, y, t, p = block1_pairs(COOC[i:i+8].to(DEV)[:, :128]); H.append(h); Y.append(y); TOK.append(t); PRV.append(p)
H = torch.cat(H); Y = torch.cat(Y); TOK = torch.cat(TOK); PRV = torch.cat(PRV)
print(f"pairs {H.shape[0]}", flush=True)

def cond_table(keys, target):
    ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
    ts.index_add_(0, keys, target); tc.index_add_(0, keys, torch.ones_like(keys, dtype=torch.float32))
    lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0)
    return lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*target.mean(0)

TT1 = cond_table(TOK, Y)
R1 = Y - TT1[TOK]
PT1 = cond_table(PRV, R1)
R2 = R1 - PT1[PRV]

def fit_interaction(target, R, steps=3500):
    A = torch.nn.Parameter(torch.randn(R, D, device=DEV)*0.02); U = torch.nn.Parameter(torch.randn(R, D, device=DEV)*0.02)
    opt = torch.optim.Adam([A, U], lr=3e-3); n = H.shape[0]
    for s in range(steps):
        ii = torch.randint(0, n, (8192,), device=DEV)
        loss = (((H[ii] @ A.T)**2) @ U - target[ii]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fvu = float((((H @ A.T)**2) @ U - target).pow(2).sum() / (target - target.mean(0)).pow(2).sum())
    return A.detach(), U.detach(), fvu

ARMS = {'table': (None, None), 'table_prev': (None, None)}
for name, R in [('table_prev_R512', 512)]:
    A, U, fvu = fit_interaction(R2, R); ARMS[name] = (A, U); print(f"{name}: FVU {fvu:.3f}", flush=True)


import torch as _t
_t.save({k: v for k, v in ARMS.items() if v[0] is not None}, f'{QK}/qk_mlp1_r512.pt')
print('fit saved', flush=True)
P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P]; EVN = torch.cat([prefN, prefN], 1).to(DEV)
g = torch.Generator().manual_seed(11); prefS = prefN.clone()
for r in range(NSEQ): prefS[r] = prefS[r][torch.randperm(P, generator=g)]
EVS = torch.cat([prefS, prefS], 1).to(DEV)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)
@torch.no_grad()
def forward(idx, arm=None, joint=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    
    prev = torch.roll(idx, 1, dims=1); prev[:, 0] = idx[:, 0]
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
        if li == 1 and arm is not None:
            use = arm if arm is not None else 'table_prev_R256'
            flat = hin.reshape(-1, D)
            mo = TT1[idx.reshape(-1)]
            if use != 'table': mo = mo + PT1[prev.reshape(-1)]
            A1, U1 = ARMS.get(use, (None, None))
            if A1 is not None: mo = mo + ((flat @ A1.T)**2) @ U1
            mo = mo.view(B, T, D).to(x.dtype)
        else:
            mo = blk.mlp(hin)
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

def adv_of(lg, EV):
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

@torch.no_grad()
def audit(arm=None, joint=False, Tcap=None):
    tot, n = 0.0, 0
    for i in range(0, len(AUD), 4):
        full = AUD[i:i+4].to(DEV)
        if Tcap is not None: full = full[:, :Tcap]
        lg = forward(full[:, :-1], arm, joint)
        ce = F.cross_entropy(lg.reshape(-1, V), full[:, 1:].reshape(-1))
        tot += ce.item()*full[:, 1:].numel(); n += full[:, 1:].numel()
    return tot/n


adv_full_n = adv_of(forward(EVN[:, :-1]), EVN); adv_full_s = adv_of(forward(EVS[:, :-1]), EVS)
res = {'floor': FLOOR1}
for arm in ['table_prev_R512']:
    d = audit(arm) - SUBBASE
    an = adv_of(forward(EVN[:, :-1], arm), EVN); as_ = adv_of(forward(EVS[:, :-1], arm), EVS)
    res[arm] = {'dCE': round(d, 5), 'understood': round(1 - d/FLOOR1, 3),
                'adv_nat_ret': round(an/adv_full_n, 3), 'adv_shuf_ret': round(as_/adv_full_s, 3)}
    print(f"{arm}: dCE +{d:.5f} ({1-d/FLOOR1:.1%}) | induction nat {an/adv_full_n:.1%} shuf {as_/adv_full_s:.1%}", flush=True)
import json as _json
_json.dump(res, open(f'{QK}/qk_mlp1_r512.json', 'w'), indent=2)
print("QK MLP1 R512 DONE", flush=True)
