"""Step 3: DAS-lite at layer 13 (layer of top patching component, head (13,8)).
Learn an r-dim orthonormal subspace Q of the residual stream entering layer 13
such that interchange-patching ONLY that subspace (clean -> corrupted) at the
opener position and/or final position recovers the closer logit.
  x_patched[pos] = x_corr[pos] + Q Q^T (x_clean_src - x_corr[pos])
Positions: 'opener' (corr pos op <- clean pos op, the actual opener residual),
'final' (corr final <- clean final, deletion-mapped), 'both'.
r in {1,4,16}; Q via QR of trainable M (differentiable interchange); trained on
30+30 analysis pairs jointly (parens + quotes), Adam; eval mean recovery and
flip rate (recovery > 0.5) on the 10+10 held-out pairs. Controls: random
orthonormal subspace (5 seeds) at same r; ceiling = full-vector replacement.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/bracket'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = len(m.transformer.h)
L = 13

S = json.load(open(f'{OUT}/stimuli.json'))
CLOSER = {t: S['summary'][t]['closer_id'] for t in ['paren', 'quote']}


def pad_batch(ids_list, T):
    idx = torch.full((len(ids_list), T), 50256, dtype=torch.long)
    for j, c in enumerate(ids_list):
        idx[j, :len(c)] = torch.tensor(c)
    return idx.to(DEV)


@torch.no_grad()
def prefix_state(idx):
    """run layers 0..L-1, return (x, x0, v1) at entry of layer L (pre-lambdas)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(L):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D,))
        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return x, x0, v1


def suffix_forward(x, x0, v1):
    """layers L..NL-1 (differentiable; params frozen)."""
    B, T = x.shape[:2]
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(L, NL):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D,))
        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


# ---- precompute per-batch states (fixed batches of 5, per task, train+held) ----
BATCH = 5
def make_batches(pairs, task):
    out = []
    for i0 in range(0, len(pairs), BATCH):
        chunk = pairs[i0:i0 + BATCH]
        Tc = max(len(p['clean_ids']) for p in chunk)
        idx_c = pad_batch([p['clean_ids'] for p in chunk], Tc)
        idx_x = pad_batch([p['corr_ids'] for p in chunk], Tc)
        fin_c = torch.tensor([len(p['clean_ids']) - 1 for p in chunk], device=DEV)
        fin_x = torch.tensor([len(p['corr_ids']) - 1 for p in chunk], device=DEV)
        op = torch.tensor([p['opener_pos'] for p in chunk], device=DEV)
        xc, x0c, v1c = prefix_state(idx_c)
        xx, x0x, v1x = prefix_state(idx_x)
        ar = torch.arange(len(chunk), device=DEV)
        with torch.no_grad():
            lg_c = suffix_forward(xc, x0c, v1c).float()
            lg_x = suffix_forward(xx, x0x, v1x).float()
        cid = CLOSER[task]
        out.append({
            'task': task, 'B': len(chunk), 'cid': cid,
            'x_corr': xx, 'x0': x0x, 'v1': v1x, 'fin_x': fin_x, 'op': op,
            'src_opener': xc[ar, op].clone(),        # clean residual at "(" pos
            'src_final': xc[ar, fin_c].clone(),      # clean residual at final pos
            'lc': lg_c[ar, fin_c, cid].clone(), 'lx': lg_x[ar, fin_x, cid].clone()})
    return out

train_b, held_b = [], []
for task in ['paren', 'quote']:
    train_b += make_batches(S['pairs'][task][:30], task)
    held_b += make_batches(S['pairs'][task][30:], task)
print(f'batches: {len(train_b)} train, {len(held_b)} held', flush=True)


def recovery(bt, Q, mode):
    """Q: (D, r) orthonormal (or None => full replacement ceiling)."""
    x = bt['x_corr'].clone()
    ar = torch.arange(bt['B'], device=DEV)
    def patch(pos, src):
        cur = x[ar, pos]
        delta = src - cur
        if Q is None:
            x[ar, pos] = src
        else:
            x[ar, pos] = cur + (delta @ Q) @ Q.T
    if mode in ('opener', 'both'):
        patch(bt['op'], bt['src_opener'])
    if mode in ('final', 'both'):
        patch(bt['fin_x'], bt['src_final'])
    lg = suffix_forward(x, bt['x0'], bt['v1']).float()
    lp = lg[ar, bt['fin_x'], bt['cid']]
    return (lp - bt['lx']) / (bt['lc'] - bt['lx'])


def evaluate(batches, Q, mode):
    with torch.no_grad():
        rs = torch.cat([recovery(bt, Q, mode) for bt in batches])
    return float(rs.mean()), float((rs > 0.5).float().mean()), rs.cpu().tolist()


results = {}
EPOCHS = 25
for mode in ['opener', 'final', 'both']:
    # ceiling
    ceil_mean, ceil_flip, _ = evaluate(held_b, None, mode)
    results[f'ceiling_{mode}'] = {'held_recovery': round(ceil_mean, 4),
                                  'held_flip': round(ceil_flip, 4)}
    print(f'\nceiling (full-vector) {mode}: held recovery {ceil_mean:.3f}', flush=True)
    for r in [1, 4, 16]:
        torch.manual_seed(100 + r)
        M = torch.randn(D, r, device=DEV) * 0.02
        M.requires_grad_(True)
        opt = torch.optim.Adam([M], lr=5e-3)
        for ep in range(EPOCHS):
            perm = np.random.RandomState(ep).permutation(len(train_b))
            for bi in perm:
                bt = train_b[bi]
                Q, _ = torch.linalg.qr(M)
                rec = recovery(bt, Q, mode)
                loss = ((1 - rec) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Q, _ = torch.linalg.qr(M)
        tr_mean, _, _ = evaluate(train_b, Q, mode)
        he_mean, he_flip, he_all = evaluate(held_b, Q, mode)
        hp = float(np.mean(he_all[:2 * 1]))  # first batches are paren
        # split held per task: first 2 batches (10 pairs) paren, last 2 quote
        he_paren = float(np.mean(he_all[:10])); he_quote = float(np.mean(he_all[10:]))
        # random control
        rnd = []
        for sd in range(5):
            g = torch.Generator(device='cpu').manual_seed(sd)
            Qr, _ = torch.linalg.qr(torch.randn(D, r, generator=g).to(DEV))
            rm, _, _ = evaluate(held_b, Qr, mode)
            rnd.append(rm)
        results[f'{mode}_r{r}'] = {
            'train_recovery': round(tr_mean, 4),
            'held_recovery': round(he_mean, 4), 'held_flip': round(he_flip, 4),
            'held_paren': round(he_paren, 4), 'held_quote': round(he_quote, 4),
            'random_control_held': round(float(np.mean(rnd)), 4),
            'random_control_std': round(float(np.std(rnd)), 4)}
        print(f'{mode} r={r}: train {tr_mean:.3f} held {he_mean:.3f} '
              f'(paren {he_paren:.3f} quote {he_quote:.3f}) flip {he_flip:.2f} '
              f'rand {np.mean(rnd):.3f}+-{np.std(rnd):.3f}', flush=True)
        if mode == 'both' and r == 16:
            torch.save(M.detach().cpu(), f'{OUT}/das_Q_both_r16.pt')

json.dump(results, open(f'{OUT}/das.json', 'w'), indent=1)
print('S3 DONE', flush=True)
