"""FORMAL completeness-ledger rerun with the explicit programs in place (was informal ~81%).
Re-audits the two program substitutions on the ledger subset (MLP0 table+R256; MLP1 tables+R512
polished gains), then recomputes the full 36-interface understood-share table with all credits:
attn0 exact, attn1 tables (0.99), attn2-17 symbolgen, MLP0 program, MLP1 program, MLP16 rank-16
gains, everything else at floor. Writes qk_ledger_v2.json + prints the table.
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
LED = json.load(open(f'{QK}/qk_completeness_ledger.json'))
SUBBASE = LED['subset_base']; AF = LED['attn_floor']; MF = LED['mlp_floor']
A0, U0 = torch.load(f'{QK}/qk_mlp0_interaction.pt', map_location=DEV)['table_R256']
A1w, U1w = torch.load(f'{QK}/qk_mlp1_r512.pt', map_location=DEV)['table_prev_R512']
POL = torch.load(f'{QK}/qk_mlp1_ce_polish.pt', map_location=DEV); G1, AB1 = POL['g'], POL['ab']


@torch.no_grad()
def pairs(idx, want):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(want + 1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
        if li == want:
            prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
            return hin.reshape(-1, D), blk.mlp(hin).reshape(-1, D), idx.reshape(-1), prv.reshape(-1)
        x = x + blk.mlp(hin)

def cond_table(keys, target):
    ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
    ts.index_add_(0, keys, target); tc.index_add_(0, keys, torch.ones_like(keys, dtype=torch.float32))
    lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0)
    return lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*target.mean(0)

TAB = {}
for want in (0, 1):
    Ys, Ts, Ps = [], [], []
    for i in range(0, 400 if want else 240, 8):
        _, y, t, p = pairs(COOC[i:i+8].to(DEV)[:, :128], want); Ys.append(y); Ts.append(t); Ps.append(p)
    Y = torch.cat(Ys); T = torch.cat(Ts); P = torch.cat(Ps)
    TT = cond_table(T, Y)
    TAB[want] = (TT, cond_table(P, Y - TT[T]) if want == 1 else None)

@torch.no_grad()
def audit(sub0, sub1):
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
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
            if li == 0 and sub0:
                flat = hin.reshape(-1, D)
                x = x + (TAB[0][0][idx.reshape(-1)] + ((flat @ A0.T)**2) @ U0).view(B, T2, D).to(x.dtype)
            elif li == 1 and sub1:
                flat = hin.reshape(-1, D)
                mo = AB1[0]*TAB[1][0][idx.reshape(-1)] + AB1[1]*TAB[1][1][prv.reshape(-1)] + (((flat @ A1w.T)**2) * G1) @ U1w
                x = x + mo.view(B, T2, D).to(x.dtype)
            else:
                x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

d0 = audit(True, False) - SUBBASE
d1 = audit(False, True) - SUBBASE
dboth = audit(True, True) - SUBBASE
print(f"MLP0 program: +{d0:.5f} | MLP1 program(polished): +{d1:.5f} | BOTH: +{dboth:.5f} (sum {d0+d1:.5f})", flush=True)

sym_attn = {2:.01763,3:.0188,4:.01927,5:.02978,6:.01187,7:.00994,8:.01591,9:.00946,10:.00545,11:.01159,12:.00274,13:.00494,14:.00759,15:.00216,16:.00476,17:.00965}
expl_cost = {}
for L in range(18):
    af = AF[str(L)]
    expl_cost[f'attn{L}'] = (0.0 if L == 0 else (0.01*af if L == 1 else sym_attn[L]))
for L in range(18):
    mf = MF[str(L)]
    if L == 0: expl_cost[f'mlp{L}'] = d0
    elif L == 1: expl_cost[f'mlp{L}'] = d1
    elif L == 16: expl_cost[f'mlp{L}'] = 0.024
    else: expl_cost[f'mlp{L}'] = mf
floors = {f'attn{L}': AF[str(L)] for L in range(18)} | {f'mlp{L}': MF[str(L)] for L in range(18)}
tot_floor = sum(floors.values()); tot_expl = sum(max(0.0, floors[k] - expl_cost[k]) for k in floors)
share = tot_expl / tot_floor
rows = sorted(floors, key=lambda k: -(floors[k] - expl_cost[k]))
print(f"\nLEDGER v2: total floor {tot_floor:.3f} nats | explained {tot_expl:.3f} | UNDERSTOOD {share:.1%}", flush=True)
unex = sorted(floors, key=lambda k: -min(expl_cost[k], floors[k]))[:6]
print("largest remaining unexplained:", [(k, round(min(expl_cost[k], floors[k]), 3)) for k in unex], flush=True)
json.dump({'mlp0_program_dCE': round(d0, 5), 'mlp1_program_dCE': round(d1, 5), 'both_dCE': round(dboth, 5),
           'total_floor': round(tot_floor, 4), 'explained': round(tot_expl, 4), 'understood_share': round(share, 4),
           'per_interface_cost': {k: round(v, 5) for k, v in expl_cost.items()}},
          open(f'{QK}/qk_ledger_v2.json', 'w'), indent=2)
print("QK LEDGER RERUN DONE", flush=True)
