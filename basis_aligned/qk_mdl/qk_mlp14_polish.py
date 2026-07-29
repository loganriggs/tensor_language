"""Gentle CE-level polish of the MLP1 program (plan arm 2): the CE-relevant part of the tail is much
smaller than its MSE share (6.4% vs 20.1%), so tune ONLY non-structural scalars -- 512 per-feature
output gains + 2 table blend scalars -- on cross-entropy through the frozen model. Train on cooc
sequences disjoint from every fitting set; held-out FineWeb substitution dCE before/after; induction
service re-verified after polish (natural + shuffled retention). Guard against the old CE-polish-
overfit failure: small scalar count, streaming data, held-out gate.
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
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
WANT = 14
A512, U512 = torch.load(f'{QK}/qk_mlp14_program.pt', map_location=DEV)['table_prev_R256']
FLOOR1 = json.load(open(f'{QK}/qk_completeness_ledger.json'))['mlp_floor'][str(14)]
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']


@torch.no_grad()
def block1_pairs(idx):  # collects at layer WANT
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(WANT + 1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
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

# tables (standard recipe, cooc 0-400)
Hs, Ys, Ts, Ps = [], [], [], []
for i in range(0, 400, 8):
    h, y, t, p = block1_pairs(COOC[i:i+8].to(DEV)[:, :128]); Ys.append(y); Ts.append(t); Ps.append(p)
Y = torch.cat(Ys); TOK = torch.cat(Ts); PRV = torch.cat(Ps)
def cond_table(keys, target):
    ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
    ts.index_add_(0, keys, target); tc.index_add_(0, keys, torch.ones_like(keys, dtype=torch.float32))
    lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0)
    return lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*target.mean(0)
TT1 = cond_table(TOK, Y); PT1 = cond_table(PRV, Y - TT1[TOK])
del Ys, Ts, Ps, Y, TOK, PRV

g = torch.nn.Parameter(torch.ones(256, device=DEV))
ab = torch.nn.Parameter(torch.ones(2, device=DEV))

def forward(idx, sub):
    B, T2 = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T2, -1)); hin = F.rms_norm(x, (D,))
        if li == WANT and sub:
            flat = hin.reshape(-1, D)
            mo = ab[0]*TT1[idx.reshape(-1)] + ab[1]*PT1[prv.reshape(-1)] + (((flat @ A512.T)**2) * g) @ U512
            x = x + mo.view(B, T2, D).to(x.dtype)
        else:
            x = x + blk.mlp(hin)
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def audit(sub):
    tot = 0.0; n = 0
    for i in range(0, 200, 4):
        b = FINEWEB[i:i+4].to(DEV)
        lg = forward(b[:, :-1], sub).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P]; EVN = torch.cat([prefN, prefN], 1).to(DEV)
gg = torch.Generator().manual_seed(11); prefS = prefN.clone()
for r in range(NSEQ): prefS[r] = prefS[r][torch.randperm(P, generator=gg)]
EVS = torch.cat([prefS, prefS], 1).to(DEV)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)
@torch.no_grad()
def adv(EV, sub):
    lg = forward(EV[:, :-1], sub).float()
    ce = F.cross_entropy(lg.reshape(-1, V), EV[:, 1:].reshape(-1), reduction='none').view(NSEQ, -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

d_pre = audit(True) - SUBBASE
advn_f, advs_f = adv(EVN, False), adv(EVS, False)
advn0, advs0 = adv(EVN, True), adv(EVS, True)
print(f"pre-polish : dCE +{d_pre:.5f} ({1-d_pre/FLOOR1:.1%}) | induction nat {advn0/advn_f:.1%} shuf {advs0/advs_f:.1%}", flush=True)

opt = torch.optim.Adam([g, ab], lr=5e-3)
for step in range(400):
    i = 2400 + np.random.randint(0, 2500); b = COOC[i:i+2].to(DEV)[:, :128]
    lg = forward(b[:, :-1], True).float()
    loss = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)

d_post = audit(True) - SUBBASE
advn1, advs1 = adv(EVN, True), adv(EVS, True)
print(f"post-polish: dCE +{d_post:.5f} ({1-d_post/FLOOR1:.1%}) | induction nat {advn1/advn_f:.1%} shuf {advs1/advs_f:.1%}", flush=True)
res = {'pre': {'dCE': round(d_pre, 5), 'understood': round(1-d_pre/FLOOR1, 3),
               'ind_nat': round(advn0/advn_f, 3), 'ind_shuf': round(advs0/advs_f, 3)},
       'post': {'dCE': round(d_post, 5), 'understood': round(1-d_post/FLOOR1, 3),
                'ind_nat': round(advn1/advn_f, 3), 'ind_shuf': round(advs1/advs_f, 3)},
       'gain_stats': {'mean': round(float(g.mean().detach()), 4), 'std': round(float(g.std().detach()), 4),
                      'table_scalars': [round(float(x), 4) for x in ab.detach()]}}
json.dump(res, open(f'{QK}/qk_mlp14_polish.json', 'w'), indent=2)
torch.save({'g': g.detach(), 'ab': ab.detach()}, f'{QK}/qk_mlp14_polish.pt')
print("QK MLP14 POLISH DONE", flush=True)
