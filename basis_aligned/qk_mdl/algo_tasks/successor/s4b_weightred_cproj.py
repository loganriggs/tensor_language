"""Step 4 (corrected target): data-conditioned weight reduction on the
load-bearing head-specific W for L8.H3.

Diagnostic (s4 + zero-ablation) showed lamb(L8)=4.0 so the head's value is
-3*c_v_L8(h) + 4*v1_L0; the L8 c_v head-3 slice is nearly null (zeroing it
keeps 98.7% of task score). Head-specific load-bearing W = c_proj columns for
head 3 at layer 8 (1152 x 128; zeroing -> 39.7% of task score).

X = task inputs to W = head-3 pattern-weighted value output yh4[:,:,3,:] over
all positions of the extended battery (n=3920). Y = W X^T; SVD; W'_r =
Y_r @ pinv(X^T, rcond=1e-4). Control = data-free SVD of W at same r."""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/successor')
from successor_lib import (HERE, DEV, FAMILIES, CYCLIC, PRED_POS, load_model,
                           load_stimuli, run, pairs_tensors)

torch.manual_seed(0)
m, cfg = load_model()
NH, D = cfg['n_head'], cfg['n_embd']
HD = D // NH
LI, HEAD = 8, 3
COLS = slice(HEAD * HD, (HEAD + 1) * HD)
W_ORIG = m.transformer.h[LI].attn.c_proj.weight[:, COLS].clone()   # (1152,128)

tok = AutoTokenizer.from_pretrained('gpt2')
COMMA = tok(',')['input_ids'][0]


def tid(word, space):
    ids = tok((' ' if space else '') + word)['input_ids']
    assert len(ids) == 1
    return ids[0]


def extended_prompts(families):
    prompts = []
    for fam_name in families:
        fam = FAMILIES[fam_name]
        n = len(fam)
        for L in range(3, 11):
            for s in range(n):
                if not CYCLIC[fam_name] and s + L >= n:
                    continue
                e = [(s + k) % n for k in range(L)]
                t = [tid(fam[e[0]], False), COMMA]
                for x in e[1:]:
                    t += [tid(fam[x], True), COMMA]
                prompts.append(t)
    return prompts


@torch.no_grad()
def collect_X(prompts, bs=8):
    """Head-3 pre-c_proj outputs at layer LI, all positions."""
    from collections import defaultdict
    groups = defaultdict(list)
    for p in prompts:
        groups[len(p)].append(p)
    out = []
    for T, ps in groups.items():
        for i in range(0, len(ps), bs):
            idx = torch.tensor(ps[i:i + bs], device=DEV)
            _, cache = run(m, cfg, idx, collect=True)
            yh = cache[('h', LI)][:, :, HEAD, :]        # (B,T,HD)
            out.append(yh.reshape(-1, HD))
    return torch.cat(out).float()


def build_Wr(X, r):
    Y = W_ORIG @ X.T                                    # (D, n)
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    Yr = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
    return Yr @ torch.linalg.pinv(X.T, rcond=1e-4)


def build_Wr_free(r):
    U, S, Vh = torch.linalg.svd(W_ORIG, full_matrices=False)
    return U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]


def set_W(Wnew):
    m.transformer.h[LI].attn.c_proj.weight[:, COLS] = Wnew


stim = load_stimuli()
ci, xi, ca, xa, rows = pairs_tensors(stim, split='heldout')
fam_of = [r['family'] for r in rows]


@torch.no_grad()
def task_score():
    lg_c = torch.cat([run(m, cfg, ci[i:i + 8])[0] for i in range(0, len(ci), 8)])
    lg_x = torch.cat([run(m, cfg, xi[i:i + 8])[0] for i in range(0, len(xi), 8)])
    n = len(ci)
    mc = lg_c[range(n), PRED_POS, ca] - lg_c[range(n), PRED_POS, xa]
    mx = lg_x[range(n), PRED_POS, ca] - lg_x[range(n), PRED_POS, xa]
    d = (mc - mx)
    acc = (lg_c[:, PRED_POS].argmax(-1) == ca).float()
    out = {'score': d.mean().item(), 'clean_acc': acc.mean().item()}
    for f in FAMILIES:
        msk = torch.tensor([x == f for x in fam_of], device=DEV)
        out[f'score_{f}'] = d[msk].mean().item()
    return out


FW = torch.from_numpy(
    np.load('/workspace/tensor_language/data_fineweb_tokens.npy')
    .astype(np.int64))[500:520, :128].to(DEV)


@torch.no_grad()
def fineweb_ce():
    tot, n = 0.0, 0
    for i in range(0, len(FW), 8):
        b = FW[i:i + 8]
        lg, _ = run(m, cfg, b[:, :-1])
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


X_all = collect_X(extended_prompts(['weekday', 'month', 'alphabet']))
X_wk = collect_X(extended_prompts(['weekday']))
print(f'X_all {tuple(X_all.shape)}  X_weekday {tuple(X_wk.shape)}', flush=True)

base = task_score()
base_ce = fineweb_ce()
print('baseline:', base, 'fineweb CE', round(base_ce, 4), flush=True)
set_W(torch.zeros_like(W_ORIG))
z = task_score()
zc = fineweb_ce()
set_W(W_ORIG)
print(f'W zeroed: retention {z["score"]/base["score"]:.3f}, CE +{zc-base_ce:.4f}',
      flush=True)

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
res = {'W': f'layer {LI} c_proj columns head {HEAD} (1152x128)',
       'n_X_all': int(X_all.shape[0]), 'n_X_weekday': int(X_wk.shape[0]),
       'baseline': base, 'baseline_fineweb_ce': base_ce,
       'zeroed': {'retention': z['score'] / base['score'],
                  'fineweb_ce_delta': zc - base_ce},
       'ranks': {}}
for r in RANKS:
    row = {}
    for name, Wr in [('data_cond', build_Wr(X_all, r)),
                     ('svd_free', build_Wr_free(r))]:
        set_W(Wr)
        ts = task_score()
        row[name] = {'retention': ts['score'] / base['score'],
                     'clean_acc': ts['clean_acc'],
                     **{f'retention_{f}': ts[f'score_{f}'] / base[f'score_{f}']
                        for f in FAMILIES},
                     'fineweb_ce_delta': fineweb_ce() - base_ce}
        set_W(W_ORIG)
    res['ranks'][r] = row
    print(f"r={r}: data {row['data_cond']['retention']:.3f} "
          f"(CE +{row['data_cond']['fineweb_ce_delta']:.4f}) | "
          f"svd {row['svd_free']['retention']:.3f} "
          f"(CE +{row['svd_free']['fineweb_ce_delta']:.4f})", flush=True)

min_r = next((r for r in RANKS if res['ranks'][r]['data_cond']['retention'] >= 0.9),
             None)
min_r_free = next((r for r in RANKS
                   if res['ranks'][r]['svd_free']['retention'] >= 0.9), None)
res['min_rank_data_cond'] = min_r
res['min_rank_svd_free'] = min_r_free
print('minimal rank >=90%: data-conditioned', min_r, '| data-free SVD',
      min_r_free, flush=True)

if min_r is not None:
    share = {}
    for r in sorted({min_r, min(2 * min_r, 128)}):
        set_W(build_Wr(X_wk, r))
        ts = task_score()
        set_W(W_ORIG)
        share[r] = {'retention_overall': ts['score'] / base['score'],
                    **{f'retention_{f}': ts[f'score_{f}'] / base[f'score_{f}']
                       for f in FAMILIES}}
        print(f'weekday-fit r={r}: ' + json.dumps(share[r]), flush=True)
    res['weekday_fit_sharing'] = share

json.dump(res, open(f'{HERE}/weightred_cproj.json', 'w'), indent=1)
print('saved weightred_cproj.json', flush=True)
