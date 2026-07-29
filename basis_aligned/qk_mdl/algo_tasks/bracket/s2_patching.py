"""Step 2: single-component activation PATCHING importance.
Run the corrupted prompt; patch in the clean-run activation of ONE component
(162 head outputs yh4[:,:,h,:] + 18 MLP outputs) at ALL positions via the
deletion position map (corr t -> clean t for t<opener_pos, else clean t+1).
Metric: recovered fraction of (clean - corr) closer-token LOGIT difference at
the final position, averaged over the 30 analysis pairs per task.
Then compare the 180-vector against qk_circuit_atlas.json punct knockout
importances (Spearman + top-10 overlap).
Forward loop adapted from qk_circuit_atlas.py run().
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
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]

S = json.load(open(f'{OUT}/stimuli.json'))
PAIRS = {t: S['pairs'][t][:30] for t in ['paren', 'quote']}
CLOSER = {t: S['summary'][t]['closer_id'] for t in ['paren', 'quote']}


def pad_batch(ids_list, T):
    idx = torch.full((len(ids_list), T), 50256, dtype=torch.long)
    for j, c in enumerate(ids_list):
        idx[j, :len(c)] = torch.tensor(c)
    return idx.to(DEV)


@torch.no_grad()
def run(idx, patch=None, collect=False):
    """patch = (component, cache, pos_map) where pos_map is (B, T) long into
    the cached clean T-axis. collect=True returns per-component activations."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    acts = {}
    for li in range(NL):
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
        if collect:
            acts[('h', li)] = yh4.clone()
        if patch is not None and patch[0][0] == 'h' and patch[0][1] == li:
            hh = patch[0][2]
            src = patch[1][('h', li)][:, :, hh, :]           # (B, Tc, HD)
            pm = patch[2].unsqueeze(-1).expand(-1, -1, HD)   # (B, T, HD)
            yh4[:, :, hh, :] = torch.gather(src, 1, pm)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            acts[('m', li)] = mo.clone()
        if patch is not None and patch[0] == ('m', li):
            src = patch[1][('m', li)]
            pm = patch[2].unsqueeze(-1).expand(-1, -1, D)
            mo = torch.gather(src, 1, pm)
        x = x + mo
    lg = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30).float()
    return lg, acts


recov = {t: {c: [] for c in ALL} for t in PAIRS}
BATCH = 6
for task, pairs in PAIRS.items():
    cid = CLOSER[task]
    for i0 in range(0, len(pairs), BATCH):
        chunk = pairs[i0:i0 + BATCH]
        Tc = max(len(p['clean_ids']) for p in chunk)
        idx_clean = pad_batch([p['clean_ids'] for p in chunk], Tc)
        idx_corr = pad_batch([p['corr_ids'] for p in chunk], Tc)  # padded to same T
        fin_clean = torch.tensor([len(p['clean_ids']) - 1 for p in chunk], device=DEV)
        fin_corr = torch.tensor([len(p['corr_ids']) - 1 for p in chunk], device=DEV)
        B = len(chunk)
        # position map corr->clean
        pm = torch.zeros(B, Tc, dtype=torch.long, device=DEV)
        for j, p in enumerate(chunk):
            op = p['opener_pos']
            t = torch.arange(Tc)
            pm[j] = torch.where(t < op, t, (t + 1).clamp(max=Tc - 1))
        lg_clean, cache = run(idx_clean, collect=True)
        lg_corr, _ = run(idx_corr)
        ar = torch.arange(B, device=DEV)
        lc = lg_clean[ar, fin_clean, cid]
        lx = lg_corr[ar, fin_corr, cid]
        denom = (lc - lx)
        for c in ALL:
            lg_p, _ = run(idx_corr, patch=(c, cache, pm))
            lp = lg_p[ar, fin_corr, cid]
            r = ((lp - lx) / denom).cpu().tolist()
            recov[task][c].extend(r)
        del cache
        torch.cuda.empty_cache()
        print(f'{task} batch {i0 // BATCH} done', flush=True)

imp = {}
for c in ALL:
    pv = float(np.mean(recov['paren'][c]))
    qv = float(np.mean(recov['quote'][c]))
    imp[c] = {'paren': round(pv, 4), 'quote': round(qv, 4),
              'mean': round((pv + qv) / 2, 4)}

order = sorted(ALL, key=lambda c: -imp[c]['mean'])
print('\nTOP-10 by mean recovery:', flush=True)
for c in order[:10]:
    print(f"  {c}: mean {imp[c]['mean']:.3f}  paren {imp[c]['paren']:.3f}  quote {imp[c]['quote']:.3f}", flush=True)

# cumulative recovery of top-k: patch top-k simultaneously (per task, one batch loop)
@torch.no_grad()
def run_multi(idx, comps, cache, pm):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    comps = set(comps)
    for li in range(NL):
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
        for hh in range(NH):
            if ('h', li, hh) in comps:
                src = cache[('h', li)][:, :, hh, :]
                pmx = pm.unsqueeze(-1).expand(-1, -1, HD)
                yh4[:, :, hh, :] = torch.gather(src, 1, pmx)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if ('m', li) in comps:
            mo = torch.gather(cache[('m', li)], 1, pm.unsqueeze(-1).expand(-1, -1, D))
        x = x + mo
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30).float()


cum = {t: {} for t in PAIRS}
for task, pairs in PAIRS.items():
    cid = CLOSER[task]
    for K in [1, 2, 3, 5, 10, 20]:
        vals = []
        for i0 in range(0, len(pairs), BATCH):
            chunk = pairs[i0:i0 + BATCH]
            Tc = max(len(p['clean_ids']) for p in chunk)
            idx_clean = pad_batch([p['clean_ids'] for p in chunk], Tc)
            idx_corr = pad_batch([p['corr_ids'] for p in chunk], Tc)
            fin_clean = torch.tensor([len(p['clean_ids']) - 1 for p in chunk], device=DEV)
            fin_corr = torch.tensor([len(p['corr_ids']) - 1 for p in chunk], device=DEV)
            B = len(chunk)
            pm = torch.zeros(B, Tc, dtype=torch.long, device=DEV)
            for j, p in enumerate(chunk):
                op = p['opener_pos']
                t = torch.arange(Tc)
                pm[j] = torch.where(t < op, t, (t + 1).clamp(max=Tc - 1))
            lg_clean, cache = run(idx_clean, collect=True)
            lg_corr, _ = run(idx_corr)
            ar = torch.arange(B, device=DEV)
            lc = lg_clean[ar, fin_clean, cid]
            lx = lg_corr[ar, fin_corr, cid]
            lg_p = run_multi(idx_corr, order[:K], cache, pm)
            lp = lg_p[ar, fin_corr, cid]
            vals.extend(((lp - lx) / (lc - lx)).cpu().tolist())
            del cache
            torch.cuda.empty_cache()
        cum[task][K] = round(float(np.mean(vals)), 4)
    print(f'cumulative {task}: {cum[task]}', flush=True)

# ---- atlas comparison ----
atlas = json.load(open('/workspace/tensor_language/basis_aligned/qk_mdl/qk_circuit_atlas.json'))
punct = atlas['importance_matrix']['punct']
mine = np.array([imp[c]['mean'] for c in ALL])
theirs = np.array([punct[str(c)] for c in ALL])
from scipy.stats import spearmanr, pearsonr
sp = spearmanr(mine, theirs)
pe = pearsonr(mine, theirs)
my_top10 = [str(c) for c in order[:10]]
their_top10 = [k for k, _ in sorted(punct.items(), key=lambda kv: -kv[1])[:10]]
overlap = [c for c in my_top10 if c in their_top10]
print(f'\nSpearman vs atlas punct: {sp.statistic:.3f} (p={sp.pvalue:.1e}); Pearson {pe.statistic:.3f}', flush=True)
print('my top10   :', my_top10, flush=True)
print('atlas top10:', their_top10, flush=True)
print('overlap:', overlap, flush=True)

json.dump({
    'importance': {str(c): imp[c] for c in ALL},
    'top10': [{'comp': str(c), **imp[c]} for c in order[:10]],
    'cumulative_topk_recovery': cum,
    'atlas_comparison': {
        'spearman': round(float(sp.statistic), 4), 'spearman_p': float(sp.pvalue),
        'pearson': round(float(pe.statistic), 4),
        'my_top10': my_top10, 'atlas_punct_top10': their_top10,
        'top10_overlap': overlap},
}, open(f'{OUT}/patching.json', 'w'), indent=1)
print('S2 DONE', flush=True)
