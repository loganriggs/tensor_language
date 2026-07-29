"""Step 4: Ethan's data-conditioned weight-rank reduction on the top component,
head (13,8).
1) Probe which of the head's matrices (c_q/c_k/c_q2/c_k2/c_v slices 64x1152,
   c_proj column-slice 1152x64) is most task-critical: data-free rank-4 SVD
   truncation of each, measure held-out closer boost drop. Pick the worst.
2) Ethan's method on the picked W: X = actual inputs W receives on task stimuli
   (~6000 positions from 300 generated clean prompts), Y = W X^T, SVD-truncate
   Y to rank r, W'_r = Y_r @ pinv(X^T, rcond=1e-4). Substitute; measure
   (a) held-out closer-logprob boost vs r (minimal r keeping >=90%),
   (b) CE on FineWeb rows 500-519 len 128 vs unmodified.
   Control: plain data-free SVD of W truncated to same r.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/bracket'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
L, H = 13, 8
attn = m.transformer.h[L].attn
tok = AutoTokenizer.from_pretrained('gpt2')

S = json.load(open(f'{OUT}/stimuli.json'))
CLOSER = {t: S['summary'][t]['closer_id'] for t in ['paren', 'quote']}
HELD = {t: S['pairs'][t][30:] for t in ['paren', 'quote']}

MATS = {  # name -> (module, row_slice?, orientation)
    'c_q': attn.c_q, 'c_k': attn.c_k, 'c_q2': attn.c_q2, 'c_k2': attn.c_k2,
    'c_v': attn.c_v, 'c_proj': attn.c_proj}
ORIG = {n: MATS[n].weight.data.clone() for n in MATS}
rs, re = H * HD, (H + 1) * HD


def get_slice(name):
    if name == 'c_proj':
        return ORIG[name][:, rs:re].clone()   # 1152 x 64, input = head output
    return ORIG[name][rs:re, :].clone()       # 64 x 1152, input = hcur


def set_slice(name, Wnew):
    w = MATS[name].weight.data
    if name == 'c_proj':
        w[:, rs:re] = Wnew
    else:
        w[rs:re, :] = Wnew


def restore(name):
    MATS[name].weight.data.copy_(ORIG[name])


def pad_batch(ids_list, T):
    idx = torch.full((len(ids_list), T), 50256, dtype=torch.long)
    for j, c in enumerate(ids_list):
        idx[j, :len(c)] = torch.tensor(c)
    return idx.to(DEV)


@torch.no_grad()
def heldout_boost():
    """mean closer-logprob boost (clean-corr) over the 20 held-out pairs."""
    boosts = []
    for task in ['paren', 'quote']:
        cid = CLOSER[task]
        pairs = HELD[task]
        for i0 in range(0, len(pairs), 5):
            chunk = pairs[i0:i0 + 5]
            for ids_key, store in [('clean_ids', 'c'), ('corr_ids', 'x')]:
                T = max(len(p[ids_key]) for p in chunk)
                idx = pad_batch([p[ids_key] for p in chunk], T)
                lg = reference_forward(m, idx).float()
                for j, p in enumerate(chunk):
                    lp = F.log_softmax(lg[j, len(p[ids_key]) - 1], -1)[cid].item()
                    p['_' + store] = lp
            boosts += [p['_c'] - p['_x'] for p in chunk]
    return float(np.mean(boosts))


base_boost = heldout_boost()
print(f'held-out boost, original model: {base_boost:.3f}', flush=True)

# ---- 1) pick the most task-critical matrix (data-free rank-4 truncation) ----
probe = {}
for name in MATS:
    W = get_slice(name)
    U, Sv, Vh = torch.linalg.svd(W, full_matrices=False)
    W4 = U[:, :4] @ torch.diag(Sv[:4]) @ Vh[:4]
    set_slice(name, W4)
    b = heldout_boost()
    restore(name)
    probe[name] = round(b, 3)
    print(f'  rank-4 data-free {name}: boost {b:.3f} (orig {base_boost:.3f})', flush=True)
picked = min(probe, key=probe.get)
print(f'picked W = {picked} of head ({L},{H})', flush=True)

# ---- 2) collect X on task stimuli ----
rng = np.random.RandomState(99)
NOUNS = ['dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars', 'rivers',
         'houses', 'chairs', 'apples', 'stones', 'clouds', 'roads', 'lamps',
         'boats', 'fields', 'doors', 'horses', 'flowers', 'windows', 'tables',
         'papers', 'shoes', 'clocks', 'plants', 'walls', 'coats', 'spoons']
P_PRE = ['The', 'Yesterday the', 'In the garden the', 'My friend saw the',
         'We think that the', 'Near the river the', 'After lunch the', 'Slowly the']
P_CL = ['which was near the {a} and the {b}', 'which sat beside the {a}',
        'which the {a} chased across the {b} and past the {c}', 'not the {a}',
        'the one near the {a}, the {b} and the {c}',
        'formerly known as the {a} of the {b}']
Q_PRE = ['She said', 'He whispered', 'The teacher announced', 'Then John replied',
         'My mother shouted', 'Someone wrote', 'The captain repeated', 'A voice cried']
Q_CT = ['the {a} is near the {b}', 'the {a} will follow the {b} and the {c}',
        'bring me the {a} and the {b}', 'the {a} belongs to the {b}, not the {c}',
        'watch the {a} by the {b}', 'the {a} moved past the {b} toward the {c}']
prompts = []
for i in range(150):
    w = rng.choice(NOUNS, 4, replace=False)
    body = P_CL[i % len(P_CL)].format(a=w[1], b=w[2], c=w[3])
    prompts.append(f'{P_PRE[i % len(P_PRE)]} {w[0]} ( {body}')
    w = rng.choice(NOUNS, 4, replace=False)
    body = Q_CT[i % len(Q_CT)].format(a=w[0], b=w[1], c=w[2])
    prompts.append(f'{Q_PRE[i % len(Q_PRE)]} " {body}')
prompt_ids = [tok(p)['input_ids'] for p in prompts]


@torch.no_grad()
def collect_inputs(ids_list, batch=8):
    """hcur at layer L (input to c_q/c_k/c_q2/c_k2/c_v slices) and head-8
    attention output yh4[:,:,H,:] (input to the c_proj column slice)."""
    Xh, Xp = [], []
    for i0 in range(0, len(ids_list), batch):
        chunk = ids_list[i0:i0 + batch]
        T = max(len(c) for c in chunk)
        idx = pad_batch(chunk, T)
        B = idx.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        for li in range(L + 1):
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
            if li == L:
                for j, c in enumerate(chunk):
                    Xh.append(hcur[j, :len(c)].cpu())
                    Xp.append(yh4[j, :len(c), H, :].cpu())
                break
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    return torch.cat(Xh), torch.cat(Xp)


Xh, Xp = collect_inputs(prompt_ids)
X = (Xp if picked == 'c_proj' else Xh).to(DEV).double()
print(f'X: {tuple(X.shape)} positions x d_in', flush=True)

W = get_slice(picked).double().to(DEV)
Y = W @ X.T                                     # d_out x n
pinvXT = torch.linalg.pinv(X.T, rcond=1e-4)     # n x d_in
Uy, Sy, Vhy = torch.linalg.svd(Y, full_matrices=False)
Uw, Sw, Vhw = torch.linalg.svd(W, full_matrices=False)

FW = torch.from_numpy(
    np.load('/workspace/tensor_language/data_fineweb_tokens.npy')[500:520, :128].astype(np.int64))


@torch.no_grad()
def fineweb_ce(batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FW), batch):
        b = FW[i:i + batch].to(DEV)
        lg = reference_forward(m, b[:, :-1]).float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


ce_base = fineweb_ce()
print(f'FineWeb CE, original model: {ce_base:.4f}', flush=True)

# floor: W = 0 (kills the head's pattern entirely) — defines the dynamic range
set_slice(picked, torch.zeros_like(get_slice(picked)))
floor_boost = heldout_boost(); floor_ce = fineweb_ce()
restore(picked)
print(f'floor (W=0): boost {floor_boost:.3f}, CE {floor_ce:.4f} '
      f'(dynamic range {base_boost - floor_boost:.3f} nats)', flush=True)

sweep = {'picked_matrix': picked, 'probe_rank4_datafree': probe,
         'base_boost_heldout': round(base_boost, 4),
         'floor_boost_W0': round(floor_boost, 4),
         'floor_fineweb_ce_W0': round(floor_ce, 4),
         'base_fineweb_ce': round(ce_base, 4), 'ranks': {}}
for r in [1, 2, 4, 8, 16, 32, 64]:
    Yr = Uy[:, :r] @ torch.diag(Sy[:r]) @ Vhy[:r]
    Wdata = (Yr @ pinvXT).float()
    Wfree = (Uw[:, :r] @ torch.diag(Sw[:r]) @ Vhw[:r]).float()
    row = {}
    for tag, Wnew in [('data_conditioned', Wdata), ('data_free', Wfree)]:
        set_slice(picked, Wnew)
        b = heldout_boost(); ce = fineweb_ce()
        restore(picked)
        row[tag] = {'boost': round(b, 4), 'boost_frac': round(b / base_boost, 4),
                    'boost_frac_floor': round((b - floor_boost) / (base_boost - floor_boost), 4),
                    'fineweb_ce': round(ce, 4),
                    'ce_damage': round(ce - ce_base, 4)}
    sweep['ranks'][r] = row
    print(f"r={r:3d}: data-cond boost {row['data_conditioned']['boost']:.3f} "
          f"(floor-frac {row['data_conditioned']['boost_frac_floor']:.2f}) CE +{row['data_conditioned']['ce_damage']:.4f} | "
          f"data-free boost {row['data_free']['boost']:.3f} "
          f"(floor-frac {row['data_free']['boost_frac_floor']:.2f}) CE +{row['data_free']['ce_damage']:.4f}", flush=True)

for tag in ['data_conditioned', 'data_free']:
    for crit in ['boost_frac', 'boost_frac_floor']:
        mins = [r for r in sweep['ranks'] if sweep['ranks'][r][tag][crit] >= 0.9]
        sweep[f'min_rank_90pct_{tag}_{crit}'] = min(mins) if mins else None
sweep['min_rank_90pct_data_conditioned'] = sweep['min_rank_90pct_data_conditioned_boost_frac_floor']
sweep['min_rank_90pct_data_free'] = sweep['min_rank_90pct_data_free_boost_frac_floor']
print('minimal r >=90%:', sweep['min_rank_90pct_data_conditioned'],
      '(data-cond) vs', sweep['min_rank_90pct_data_free'], '(data-free)', flush=True)

json.dump(sweep, open(f'{OUT}/weight_reduction.json', 'w'), indent=1)
print('S4 DONE', flush=True)
