"""Shared infrastructure for the numbered-list INCREMENT task decomposition (bilin18).

Task: "{k}. {w1}\n{k+1}. {w2}\n" -> next token "{k+2}".
All prompts tokenize to exactly 8 GPT-2 tokens:
  pos 0: "k"   pos 1: "."   pos 2: " w1"  pos 3: "\n"
  pos 4: "k+1" pos 5: "."   pos 6: " w2"  pos 7: "\n"   (final; predicts k+2)

Forward replicates tier2_model.reference_forward / qk_circuit_atlas.run semantics
(bf16 rotary tables, unnormalized bilinear pattern, v1 block-0 mixing, soft-cap),
extended with:
  - component activation caching   (heads: per-head pre-c_proj output; MLPs: mlp output)
  - component activation patching  (optionally restricted to a set of positions)
  - residual-stream hooks at pre-attn / pre-mlp read points (for DAS)
  - optional gradient flow (for DAS training; model params are frozen by loader)
"""
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/algo_tasks/increment'
DEV = 'cuda'

_model = None
_cfg = None

def get_model():
    global _model, _cfg
    if _model is None:
        _model, _cfg = load_elriggs('bilin18')
    return _model, _cfg


def forward(m, idx, cache=None, patch=None, resid_hook=None):
    """idx [B,T] -> logits [B,T,V].

    cache: dict to be filled with ('h',li)-> [B,T,NH,HD] (per-head mixed-v pattern
           output, BEFORE c_proj) and ('m',li)-> [B,T,D] (mlp output incl. bias);
           also ('resid_a',li) / ('resid_m',li): residual stream x as read by the
           attn / mlp of layer li (pre-rms_norm).
    patch: dict {('h',li,h): (act[B,T,HD], pos_or_None), ('m',li): (act[B,T,D], pos_or_None)}
           substitutes the component's output with `act` at positions `pos`
           (list/tensor of positions; None = all positions).
    resid_hook: fn(site, li, x) -> x, site in {'pre_attn','pre_mlp'}; applied to the
           residual stream (modifies everything downstream).
    """
    cfg = m.config
    NH, D = cfg.n_head, cfg.n_embd
    HD = D // NH
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x
    v1 = None
    cos, sin = rope_tables(T, HD, idx.device, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li, blk in enumerate(m.transformer.h):
        a = blk.attn
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if resid_hook is not None:
            x = resid_hook('pre_attn', li, x)
        if cache is not None:
            cache[('resid_a', li)] = x.detach().clone()
        h = F.rms_norm(x, (D,))

        def qk(lin):
            z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)          # UNNORMALIZED
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if cache is not None:
            cache[('h', li)] = yh4.detach().clone()
        if patch is not None:
            for hh in range(NH):
                key = ('h', li, hh)
                if key in patch:
                    act, pos = patch[key]
                    if pos is None:
                        yh4 = yh4.clone(); yh4[:, :, hh, :] = act
                    else:
                        yh4 = yh4.clone(); yh4[:, pos, hh, :] = act[:, pos]
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        if resid_hook is not None:
            x = resid_hook('pre_mlp', li, x)
        if cache is not None:
            cache[('resid_m', li)] = x.detach().clone()
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if cache is not None:
            cache[('m', li)] = mo.detach().clone()
        if patch is not None and ('m', li) in patch:
            act, pos = patch[('m', li)]
            if pos is None:
                mo = act
            else:
                mo = mo.clone(); mo[:, pos] = act[:, pos]
        x = x + mo
    x = F.rms_norm(x, (D,))
    return 30 * torch.tanh(m.lm_head(x) / 30)


def batched(fn, idx, bs=8):
    """Apply fn to batches of <=8 rows (GPU shared with other agents), concat results."""
    outs = []
    for i in range(0, len(idx), bs):
        outs.append(fn(idx[i:i + bs]))
    return torch.cat(outs, 0)


def retry_oom(fn, *args, **kw):
    for attempt in range(5):
        try:
            return fn(*args, **kw)
        except torch.cuda.OutOfMemoryError:
            print(f'OOM, retrying in 60s (attempt {attempt+1})', flush=True)
            torch.cuda.empty_cache()
            time.sleep(60)
    raise RuntimeError('OOM after 5 retries')


# ---- stimuli --------------------------------------------------------------
NOUN_CANDIDATES = [
    'dogs', 'cats', 'birds', 'fish', 'cars', 'books', 'trees', 'houses',
    'chairs', 'tables', 'apples', 'stones', 'rivers', 'clouds', 'horses',
    'ships', 'roads', 'songs', 'games', 'doors', 'walls', 'boxes', 'coins',
    'stars', 'plants', 'shoes', 'hats', 'cups', 'keys', 'maps',
]


def build_stimuli(n_pairs=40, seed=0):
    """Returns dict with clean/corr token id tensors [n,8], answer ids, metadata.

    Corruption: CONSTANT SHIFT — both list numbers k, k+1 are replaced by
    k', k'+1 with k' != k (same words). The corrupted prompt is a well-formed
    list whose correct next token is k'+2 != k+2, so the corruption cleanly
    MOVES the answer rather than destroying the task (keeps the logit-margin
    metric well-defined at both endpoints).
    """
    import random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    # filter nouns to single-token-with-leading-space
    nouns = [w for w in NOUN_CANDIDATES if len(tok(' ' + w)['input_ids']) == 1]
    rng = random.Random(seed)
    clean, corr, meta = [], [], []
    clean_ans, corr_ans = [], []
    seen = set()
    while len(clean) < n_pairs:
        k = rng.randint(1, 7)
        kp = rng.choice([j for j in range(1, 8) if j != k])
        w1, w2 = rng.sample(nouns, 2)
        sig = (k, kp, w1, w2)
        if sig in seen:
            continue
        seen.add(sig)
        s_clean = f"{k}. {w1}\n{k+1}. {w2}\n"
        s_corr = f"{kp}. {w1}\n{kp+1}. {w2}\n"
        ic = tok(s_clean)['input_ids']
        ix = tok(s_corr)['input_ids']
        if len(ic) != 8 or len(ix) != 8:
            continue
        # structural check: digits at 0 and 4
        a_c = tok(str(k + 2))['input_ids']
        a_x = tok(str(kp + 2))['input_ids']
        assert len(a_c) == 1 and len(a_x) == 1
        clean.append(ic); corr.append(ix)
        clean_ans.append(a_c[0]); corr_ans.append(a_x[0])
        meta.append({'k': k, 'k_corr': kp, 'w1': w1, 'w2': w2,
                     'clean': s_clean, 'corr': s_corr})
    return {'clean': torch.tensor(clean), 'corr': torch.tensor(corr),
            'clean_ans': torch.tensor(clean_ans), 'corr_ans': torch.tensor(corr_ans),
            'meta': meta}


POSITIONS = {'digit1': [0], 'digit2': [4], 'digits': [0, 4], 'words': [2, 6],
             'newlines': [3, 7], 'final': [7], 'all': None}
