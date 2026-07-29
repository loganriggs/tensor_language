"""Where does MLP1's remaining ~6% live? Decompose the program tail by SOURCE-STREAM PAIR.
MLP1's input is a sum of named streams: E (embedding), A0 (attn-0 out), M0 (MLP-0 out), A1
(attn-1 out). The R512 program already captures current-residual quadratics; the tail is fit
against restricted CROSS-STREAM quadratic families (a_r.s)(b_r.t) for stream pairs (s,t), R=64
each. If A1-involved families dominate -> the tail is attention-carried and FOLDABLE into
attention-head weights / RoPE-bearing pattern terms (Logan's hypothesis). Families: ExE (control),
ExA1, A1xA1, M0xA1, M0xM0, ExM0. Held-out FVU-of-tail per family + best pair combined.
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
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
A512, U512 = torch.load(f'{QK}/qk_mlp1_r512.pt', map_location=DEV)['table_prev_R512']


@torch.no_grad()
def streams(idx):
    """per-position: E, A0, M0, A1, hin1, y1(=mlp1 out), tok, prev."""
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    outs = {}
    for li in range(2):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh4.reshape(B, T, -1))
        outs[f'A{li}'] = aout.reshape(-1, D)
        x = x + aout; hin = F.rms_norm(x, (D,))
        mo = blk.mlp(hin)
        if li == 0: outs['M0'] = mo.reshape(-1, D)
        if li == 1:
            outs['hin1'] = hin.reshape(-1, D); outs['y1'] = mo.reshape(-1, D)
        x = x + mo
    outs['E'] = x0.reshape(-1, D)
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
    return outs, idx.reshape(-1), prv.reshape(-1)

# tables (train recipe) + tail computation, streamed
def cond_table(keys, target, prior=3.0):
    ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
    ts.index_add_(0, keys, target); tc.index_add_(0, keys, torch.ones_like(keys, dtype=torch.float32))
    lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+prior)
    return lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*target.mean(0)

buf = {k: [] for k in ['E', 'A0', 'M0', 'A1', 'hin1', 'y1']}; toks, prvs = [], []
for i in range(0, 400, 8):
    o, t, p = streams(COOC[i:i+8].to(DEV)[:, :128])
    for k in buf: buf[k].append(o[k])
    toks.append(t); prvs.append(p)
S = {k: torch.cat(v) for k, v in buf.items()}; TOK = torch.cat(toks); PRV = torch.cat(prvs)
# per-stream global normalization (streams have wildly different norms; unnormalized fits diverge)
SN = {k: (S[k] / S[k].std()).contiguous() for k in ['E', 'A0', 'M0', 'A1', 'hin1']}
TT1 = cond_table(TOK, S['y1'])
R1 = S['y1'] - TT1[TOK]
PT1 = cond_table(PRV, R1)
R2 = R1 - PT1[PRV]
TAIL = R2 - ((S['hin1'] @ A512.T)**2) @ U512
tail_share = float(TAIL.pow(2).sum() / (S['y1'] - S['y1'].mean(0)).pow(2).sum())
print(f"tail = {tail_share:.3f} of MLP1 output variance", flush=True)
n = TAIL.shape[0]; tr = torch.arange(n, device=DEV) < int(n*0.85); te = ~tr

def fit_cross(s, t, R=64, steps=3200):
    A = torch.nn.Parameter(torch.randn(R, D, device=DEV)*0.02)
    Bp = torch.nn.Parameter(torch.randn(R, D, device=DEV)*0.02)
    U = torch.nn.Parameter(torch.randn(R, D, device=DEV)*0.02)
    opt = torch.optim.Adam([A, Bp, U], lr=1e-3)
    idx_tr = tr.nonzero().squeeze(1)
    for _ in range(steps):
        ii = idx_tr[torch.randint(0, len(idx_tr), (8192,), device=DEV)]
        f = (SN[s][ii] @ A.T) * (SN[t][ii] @ Bp.T)
        loss = (f @ U - TAIL[ii]).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([A, Bp, U], 1.0)
        opt.step()
    with torch.no_grad():
        f = (SN[s][te] @ A.T) * (SN[t][te] @ Bp.T)
        fvu = float((f @ U - TAIL[te]).pow(2).sum() / TAIL[te].pow(2).sum())
    return 1 - fvu   # fraction of tail explained (held-out)

FAM = [('hin1', 'hin1'), ('E', 'E'), ('E', 'A1'), ('A1', 'A1'), ('M0', 'A1'), ('M0', 'M0'), ('E', 'M0'), ('A0', 'A1')]
res = {'tail_share_of_output_var': round(tail_share, 4)}
for s, t in FAM:
    ex = fit_cross(s, t)
    res[f'{s}x{t}'] = round(ex, 4)
    print(f"{s}x{t}: explains {ex:.1%} of tail (held-out)", flush=True)
json.dump(res, open(f'{QK}/qk_mlp1_tail_source.json', 'w'), indent=2)
print("QK MLP1 TAIL SOURCE DONE", flush=True)
