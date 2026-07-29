"""RED-TEAM FIX #1: the all-programs JOINT substitution and JOINT floor that the 89.4% headline
extrapolated but never measured. Substitute ALL credited MLP programs simultaneously (layers
0,1,2,3,4,5,16,17 with their polish scalars where they exist) -> joint dCE; and mean-ablate all
eight interfaces simultaneously -> joint floor. Joint substitutable fraction = 1 - joint/floor.
Compare against sum-of-singles (0.4769) to quantify program-level superadditivity honestly.
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
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']
LAYERS = [0, 1, 2, 3, 4, 5, 16, 17]

PROG = {}
PROG[0] = {'AU': torch.load(f'{QK}/qk_mlp0_interaction.pt', map_location=DEV)['table_R256'], 'g': None, 'ab': None, 'prev': False}
PROG[1] = {'AU': torch.load(f'{QK}/qk_mlp1_r512.pt', map_location=DEV)['table_prev_R512'], 'prev': True}
p1 = torch.load(f'{QK}/qk_mlp1_ce_polish.pt', map_location=DEV); PROG[1]['g'], PROG[1]['ab'] = p1['g'], p1['ab']
for L in (2, 3, 4, 16, 17):
    key = 'table_prev_R512' if L == 17 else 'table_prev_R256'
    PROG[L] = {'AU': torch.load(f'{QK}/qk_mlp{L}_program.pt', map_location=DEV)[key], 'prev': True}
    pl = torch.load(f'{QK}/qk_mlp{L}_polish.pt', map_location=DEV); PROG[L]['g'], PROG[L]['ab'] = pl['g'], pl['ab']
PROG[5] = {'AU': torch.load(f'{QK}/qk_mlp5_program.pt', map_location=DEV)['table_prev_R256'], 'g': None, 'ab': None, 'prev': True}

# one collection pass over cooc: (hin, y) at every target layer -> tables + mean-input floors
acc = {L: {'ysum_tok': torch.zeros(V, D, device=DEV), 'cnt_tok': torch.zeros(V, device=DEV),
           'hsum': torch.zeros(D, device=DEV, dtype=torch.float64), 'n': 0,
           'raw': []} for L in LAYERS}
@torch.no_grad()
def collect(idx, first240):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
    out = {}
    for li in range(NL):
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
        if li in LAYERS:
            y = blk.mlp(hin)
            A = acc[li]
            use = (li != 0) or first240   # MLP0 tables replicate the 0-240 fit range
            if use:
                yf = y.reshape(-1, D); tf = idx.reshape(-1); pf = prv.reshape(-1)
                A['ysum_tok'].index_add_(0, tf, yf); A['cnt_tok'].index_add_(0, tf, torch.ones_like(tf, dtype=torch.float32))
                A['raw'].append((yf.cpu(), tf.cpu(), pf.cpu()))
            A['hsum'] += hin.reshape(-1, D).double().sum(0); A['n'] += hin.reshape(-1, D).shape[0]
            x = x + y
        else:
            x = x + blk.mlp(hin)
    return out
for i in range(0, 400, 8):
    collect(COOC[i:i+8].to(DEV)[:, :128], first240=(i < 240))
TT, PT, MUH = {}, {}, {}
for L in LAYERS:
    A = acc[L]
    gmean_cnt = A['cnt_tok'].clamp_min(1)
    lam = A['cnt_tok'].unsqueeze(1)/(A['cnt_tok'].unsqueeze(1)+3.0)
    ys = torch.cat([r[0] for r in A['raw']]).to(DEV); ts = torch.cat([r[1] for r in A['raw']]).to(DEV); ps = torch.cat([r[2] for r in A['raw']]).to(DEV)
    gmean = ys.mean(0)
    TT[L] = lam*(A['ysum_tok']/gmean_cnt.unsqueeze(1)) + (1-lam)*gmean
    if PROG[L]['prev']:
        r1 = ys - TT[L][ts]
        psum = torch.zeros(V, D, device=DEV); pcnt = torch.zeros(V, device=DEV)
        psum.index_add_(0, ps, r1); pcnt.index_add_(0, ps, torch.ones_like(ps, dtype=torch.float32))
        lamp = pcnt.unsqueeze(1)/(pcnt.unsqueeze(1)+3.0)
        PT[L] = lamp*(psum/pcnt.clamp_min(1).unsqueeze(1)) + (1-lamp)*r1.mean(0)
    MUH[L] = F.rms_norm((A['hsum']/A['n']).float(), (D,))
    A['raw'] = None
print("tables + floors ready", flush=True)


@torch.no_grad()
def audit(mode):
    """mode: 'joint' all programs | 'floor' all interfaces mean-input | None base"""
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
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
            if li in LAYERS and mode == 'joint':
                Aq, Uq = PROG[li]['AU']; flat = hin.reshape(-1, D)
                f = (flat @ Aq.T)**2
                if PROG[li].get('g') is not None: f = f * PROG[li]['g']
                mo = f @ Uq
                ab = PROG[li].get('ab')
                a0 = ab[0] if ab is not None else 1.0
                mo = mo + a0*TT[li][idx.reshape(-1)]
                if PROG[li]['prev']:
                    a1 = ab[1] if ab is not None else 1.0
                    mo = mo + a1*PT[li][prv.reshape(-1)]
                x = x + mo.view(B, T2, D).to(x.dtype)
            elif li in LAYERS and mode == 'floor':
                x = x + blk.mlp(MUH[li].expand(B, T2, D).to(x.dtype))
            else:
                x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

base = audit(None)
dj = audit('joint') - base
df = audit('floor') - base
sum_singles = 0.07914+0.08459+0.05886+0.05299+0.04424+0.04993+0.03265+0.07448
res = {'recomputed_base': round(base, 5), 'joint_dCE': round(dj, 5), 'joint_floor': round(df, 5),
       'sum_of_singles': round(sum_singles, 5),
       'superadditivity_ratio': round(dj/sum_singles, 3),
       'joint_substitutable_fraction': round(1 - dj/df, 4)}
print(f"JOINT 8-MLP stack: dCE +{dj:.5f} | sum-of-singles {sum_singles:.5f} (ratio {dj/sum_singles:.2f}x)", flush=True)
print(f"JOINT floor (all 8 mean-ablated): +{df:.5f} | joint substitutable fraction {1-dj/df:.1%}", flush=True)
json.dump(res, open(f'{QK}/qk_joint_mlp_stack.json', 'w'), indent=2)
print("QK JOINT MLP STACK DONE", flush=True)
