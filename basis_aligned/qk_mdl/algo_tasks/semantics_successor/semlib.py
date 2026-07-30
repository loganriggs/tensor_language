"""Shared lib: SEMANTICS VERIFICATION of the successor payload channel in bilin18.

Hypothesis H: the layer-8 broadcast channel (heads L8H3 + L8H7, riding the v1
value cache with lamb=4) carries the IDENTITY of the last sequence element (a
token pointer) at the prediction position, and MLPs 8-14 implement per-family
successor TABLES over it.

CODE: payload(e) ~ W * emb(e) + b, a single linear map from the token embedding
of the pointed-at element to the concatenated head-space outputs of L8H3 and
L8H7 at the prediction position (2 x 128 dims), calibrated across all families
(weekday / month / alphabet / digit-comma / numbered-list).

Forward replicates tier2_model / qk_circuit_atlas semantics (bf16 rotary,
unnormalized bilinear pattern (q1k1)(q2k2)/d^2, v1 block-0 value lerp,
30*tanh soft-cap), extended with head-output substitution / scaling, v1-slice
substitution (site B), and pattern caching.
"""
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot  # noqa: E402

HERE = '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor'
DEV = 'cuda'
L_PAY = 8          # payload layer
HEADS = (3, 7)     # payload heads (successor: H3 dominant; increment: H7 dominant)

_model, _cfg, _tok = None, None, None


def get_model():
    global _model, _cfg
    if _model is None:
        _model, _cfg = load_elriggs('bilin18')
    return _model, _cfg


def get_tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained('gpt2')
    return _tok


def free_gb():
    f, t = torch.cuda.mem_get_info()
    return f / 1e9


def retry_oom(fn, *args, **kw):
    for attempt in range(5):
        try:
            return fn(*args, **kw)
        except torch.cuda.OutOfMemoryError:
            print(f'OOM, retrying in 60s (attempt {attempt + 1})', flush=True)
            torch.cuda.empty_cache()
            time.sleep(60)
    raise RuntimeError('OOM after 5 retries')


# ---------------------------------------------------------------- forward ----
@torch.no_grad()
def run(m, cfg, idx, head_sub=None, head_scale=None, v1_sub=None,
        want_pat=(), want_head=()):
    """idx (B,T) -> logits (B,T,V), cache.

    head_sub:   {(li,h): (val, pos)}  substitute head h's pre-c_proj output
                (yh4[:,pos,h,:]) at layer li.  val (B,HD) with pos = int or
                LongTensor(B); or val (B,T,HD) with pos None (all positions).
    head_scale: {(li,h): s}  multiply head h's yh4 by scalar s at all positions.
    v1_sub:     {(li,h): (val, pos)}  at layer li, pretend the v1 cache slice
                for head h at key position pos was `val` (B,HD): the lerped
                value becomes (1-lamb)*v_own + lamb*val there. (Site B.)
    want_pat:   layers li for which cache[('pat',li)] = pattern (B,NH,T,T).
    want_head:  layers li for which cache[('h',li)] = yh4 (B,T,NH,HD).
    """
    NH, D = cfg['n_head'], cfg['n_embd']
    HD, NL = D // NH, cfg['n_layer']
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0, v1 = x, None
    cos, sin = rope_tables(T, HD, idx.device, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    cache = {}
    ar = torch.arange(B, device=idx.device)
    for li in range(NL):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (D,))

        def qk(lin):
            z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if v1_sub:
            for (pl, ph), (val, pos) in v1_sub.items():
                if pl == li:
                    v = v.clone()
                    lam = a.lamb
                    if pos is None:     # val (B,T,HD), all positions
                        v[:, :, ph, :] += lam * (val - v1[:, :, ph, :])
                    else:
                        v[ar, pos, ph, :] += lam * (val - v1[ar, pos, ph, :])
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)          # UNNORMALIZED
        if li in want_pat:
            cache[('pat', li)] = pat.detach().clone()
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if head_scale:
            for (pl, ph), s in head_scale.items():
                if pl == li:
                    yh4 = yh4.clone()
                    yh4[:, :, ph, :] *= s
        if head_sub:
            for (pl, ph), (val, pos) in head_sub.items():
                if pl == li:
                    yh4 = yh4.clone()
                    if pos is None:
                        yh4[:, :, ph, :] = val
                    else:
                        yh4[ar, pos, ph, :] = val
        if li in want_head:
            cache[('h', li)] = yh4.detach().clone()
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    lg = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30).float()
    return lg, cache


def batched_run(m, cfg, idx, bs=6, **kw):
    """Run in batches <=bs; returns concat logits (cpu float32) and caches list."""
    outs, caches = [], []
    for i in range(0, len(idx), bs):
        sub = dict(kw)
        # slice per-batch substitution values
        for key in ('head_sub', 'v1_sub'):
            if kw.get(key):
                sub[key] = {k2: (v[0][i:i + bs], v[1] if not torch.is_tensor(v[1])
                                 else v[1][i:i + bs]) for k2, v in kw[key].items()}
        lg, c = retry_oom(run, m, cfg, idx[i:i + bs], **sub)
        outs.append(lg.cpu())
        caches.append(c)
    return torch.cat(outs, 0), caches


# ---------------------------------------------------------------- families ---
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
LETTERS = [chr(ord('a') + i) for i in range(26)]
DIGITS = [str(i) for i in range(10)]
WORDNUM = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
NOUNS = ['dogs', 'cats', 'birds', 'fish', 'cars', 'books', 'trees', 'houses',
         'chairs', 'tables', 'apples', 'stones']

# comma-list families: elements cycle (weekday/month) or truncate (alphabet/digit)
FAMILIES = {'weekday': DAYS, 'month': MONTHS, 'alphabet': LETTERS, 'digit': DIGITS}
CYCLIC = {'weekday': True, 'month': True, 'alphabet': False, 'digit': False}


def succ(family, e):
    """Ground-truth successor ELEMENT string, or None at a non-cyclic end.
    For cyclic families returns the wrap successor (model known not to wrap)."""
    lst = FAMILIES[family]
    i = lst.index(e)
    if i + 1 < len(lst):
        return lst[i + 1]
    return lst[0] if CYCLIC[family] else None


def tok1(s):
    ids = get_tok()(s)['input_ids']
    assert len(ids) == 1, f'{s!r} -> {ids}'
    return ids[0]


def comma_prompt(family, start, length):
    """Elements lst[start], lst[start+1], ... (mod for cyclic), as
    'E0, E1, ..., Ek,'  -> tokens [E0][,][ E1][,]...[ Ek][,], 2*length tokens.
    Returns dict(tokens, last_tok, last_elem, ans_tok(or None), family, ...)."""
    lst = FAMILIES[family]
    n = len(lst)
    ii = [(start + j) % n if CYCLIC[family] else start + j for j in range(length)]
    assert all(i < n for i in ii)
    elems = [lst[i] for i in ii]
    toks = [tok1(elems[0])]
    for e in elems[1:]:
        toks += [tok1(','), tok1(' ' + e)]
    toks += [tok1(',')]
    last = elems[-1]
    sc = succ(family, last)
    return {'family': family, 'tokens': toks, 'pred_pos': len(toks) - 1,
            'last_pos': len(toks) - 2, 'last_tok': tok1(' ' + last),
            'last_elem': last, 'succ_elem': sc,
            'ans_tok': tok1(' ' + sc) if sc is not None else None,
            'text': elems[0] + ''.join(', ' + e for e in elems[1:]) + ','}


def numlist_prompt(k, w1, w2):
    """'{k}. {w1}\n{k+1}. {w2}\n' -> 8 tokens; element = str(k+1) at pos 4,
    pred pos 7, answer str(k+2)."""
    s = f'{k}. {w1}\n{k + 1}. {w2}\n'
    toks = get_tok()(s)['input_ids']
    assert len(toks) == 8, (s, toks)
    return {'family': 'numlist', 'tokens': toks, 'pred_pos': 7, 'last_pos': 4,
            'last_tok': tok1(str(k + 1)), 'last_elem': str(k + 1),
            'succ_elem': str(k + 2), 'ans_tok': tok1(str(k + 2)), 'text': s}


def follow_ans_tok(family, elem):
    """Token the model should output if it treats `elem` as the pointed element
    in a `family`-format context. None if no defined successor."""
    if family == 'numlist':
        if int(elem) + 1 > 9:
            return None
        return tok1(str(int(elem) + 1))
    fam = 'digit' if family == 'digit' else family
    s = succ(fam, elem)
    if s is None:
        return None
    if CYCLIC.get(fam, False) and FAMILIES[fam].index(elem) == len(FAMILIES[fam]) - 1:
        return None   # model does not wrap; treat wrap as undefined for follow
    return tok1(' ' + s)


# ------------------------------------------------------------- code (W) ------
def fit_ridge(Phi, Y, lam_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0), seed=0):
    """Phi (N,Din) float64 cpu, Y (N,Dout). Appends bias column. Selects lam by
    random 80/20 split R^2, refits on all. Returns W (Din+1, Dout), best lam,
    holdout R^2 at best lam."""
    N = Phi.shape[0]
    X = torch.cat([Phi, torch.ones(N, 1, dtype=Phi.dtype)], 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    ntr = int(0.8 * N)
    tr, te = perm[:ntr], perm[ntr:]
    scale = (X[tr].T @ X[tr]).diagonal().mean()
    best = None
    for lam in lam_grid:
        A = X[tr].T @ X[tr] + lam * scale / X.shape[1] * torch.eye(X.shape[1], dtype=X.dtype)
        W = torch.linalg.solve(A, X[tr].T @ Y[tr])
        pred = X[te] @ W
        r2 = 1 - ((pred - Y[te]) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum()
        if best is None or r2 > best[1]:
            best = (lam, r2.item())
    lam = best[0]
    A = X.T @ X + lam * scale / X.shape[1] * torch.eye(X.shape[1], dtype=X.dtype)
    W = torch.linalg.solve(A, X.T @ Y)
    return W, lam, best[1]


def coded_payload(W, m, toks):
    """W (1153, 2*HD) float64 cpu; toks LongTensor(B) on DEV.
    Returns dict {(L_PAY,h): val (B,HD) float32 DEV} (no position attached)."""
    emb = m.transformer.wte.weight[toks].detach().double().cpu()      # (B,1152)
    X = torch.cat([emb, torch.ones(len(toks), 1, dtype=torch.float64)], 1)
    out = (X @ W).float().to(DEV)                                     # (B,256)
    HD = out.shape[1] // len(HEADS)
    return {h: out[:, i * HD:(i + 1) * HD] for i, h in enumerate(HEADS)}


def build_pairs(seed=1, length=4):
    """Clean/donor pairs: same context, last element replaced (donor's follow
    token defined). Comma families length `length` + numlist. All 2*length==8
    tokens when length==4. Returns list of (clean_dict, donor_dict)."""
    rng = random.Random(seed)
    tok = get_tok()
    pairs = []
    for fam, lst in FAMILIES.items():
        n = len(lst)
        starts = range(n) if CYCLIC[fam] else range(n - length + 1)
        for s in starts:
            p = comma_prompt(fam, s, length)
            if p['ans_tok'] is None or (CYCLIC[fam] and (s + length - 1) % n == n - 1):
                continue   # skip prompts whose own answer is a wrap
            cands = [e for e in lst if e != p['last_elem']
                     and follow_ans_tok(fam, e) is not None]
            e2 = rng.choice(cands)
            q = dict(p)
            q['tokens'] = list(p['tokens'])
            q['tokens'][p['last_pos']] = tok1(' ' + e2)
            q['last_elem'], q['last_tok'] = e2, tok1(' ' + e2)
            q['follow_tok'] = follow_ans_tok(fam, e2)
            pairs.append((p, q))
    for k in range(1, 7):
        p = numlist_prompt(k, *rng.sample(NOUNS, 2))
        k2 = rng.choice([j for j in range(1, 8) if j != k])
        w1 = tok.decode([p['tokens'][2]]).strip()
        w2 = tok.decode([p['tokens'][6]]).strip()
        q = numlist_prompt(k2, w1, w2)
        q['follow_tok'] = q['ans_tok']
        pairs.append((p, q))
    return pairs


def v1_of_tokens(m, cfg, toks):
    """v1 cache slice (token-determined layer-0 c_v) for token ids (B,).
    Returns (B, NH, HD)."""
    D, NH = cfg['n_embd'], cfg['n_head']
    with torch.no_grad():
        e0 = F.rms_norm(m.transformer.wte(toks), (D,))
        blk0 = m.transformer.h[0]
        x_in = (blk0.lambdas[0] + blk0.lambdas[1]) * e0
        return blk0.attn.c_v(F.rms_norm(x_in, (D,))).view(len(toks), NH, -1)


# ------------------------------------------------------------ natural CE -----
def load_rows(which, sl):
    p = {'fineweb': '/workspace/tensor_language/data_fineweb_tokens.npy',
         'cooc': '/workspace/tensor_language/data_fineweb_cooc_tokens.npy'}[which]
    a = np.load(p, mmap_mode='r')[sl]
    return torch.tensor(np.asarray(a, dtype=np.int64))


@torch.no_grad()
def ce_per_token(m, cfg, rows, bs=4, hook=None):
    """rows (N,513) cpu. Returns per-token loss (N,512) cpu float32.
    hook(idx_batch) -> kwargs dict for run() (e.g. head_scale / head_sub)."""
    outs = []
    for i in range(0, len(rows), bs):
        idx = rows[i:i + bs].to(DEV)
        kw = hook(idx) if hook is not None else {}
        lg, _ = retry_oom(run, m, cfg, idx[:, :-1], **kw)
        ls = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), idx[:, 1:].reshape(-1),
                             reduction='none').view(idx.shape[0], -1)
        outs.append(ls.float().cpu())
        del lg
    return torch.cat(outs, 0)


def dce_stats(base, mod):
    """Paired per-token dCE. Returns mean, row-clustered SE, per-token SE."""
    d = (mod - base)
    rows = d.mean(1)
    return {'dce': d.mean().item(),
            'se_row': (rows.std(unbiased=True) / len(rows) ** 0.5).item(),
            'se_tok': (d.std(unbiased=True) / d.numel() ** 0.5).item()}


def save_json(name, obj):
    with open(f'{HERE}/{name}', 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f'wrote {name}', flush=True)
