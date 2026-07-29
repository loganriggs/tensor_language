"""Shared lib for the memorized-successor-sequence decomposition of bilin18.

Prompt format (6 tokens, all single-token elements):
  [e0][,][ e1][,][ e2][,]  -> predict [ succ(e2)] at final position 5.
Last-element token position = 4 (LAST_POS), prediction position = 5 (PRED_POS).
Corrupted pair: same prompt with e2 replaced by a different family member c;
correct answer becomes succ(c). Position-counting answer stays succ(e2)."""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot  # noqa: E402

HERE = '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/successor'
DEV = 'cuda'
LAST_POS, PRED_POS = 4, 5

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
LETTERS = [chr(ord('a') + i) for i in range(26)]
FAMILIES = {'weekday': DAYS, 'month': MONTHS, 'alphabet': LETTERS}
CYCLIC = {'weekday': True, 'month': True, 'alphabet': False}


def load_model():
    m, cfg = load_elriggs('bilin18')
    return m, cfg


def run(m, cfg, idx, patch_head=None, patch_mlp=None, resid_fn=None,
        collect=False, grad=False):
    """Forward replicating atlas semantics. idx: (B,T) LongTensor.
    patch_head: {(li,h): (B,T,HD)} replaces that head's pre-c_proj output.
    patch_mlp: {li: (B,T,D)} replaces that MLP's output.
    resid_fn(li, x)->x applied to the residual after block li.
    collect=True returns cache of ('h',li) (B,T,NH,HD), ('m',li), ('r',li)."""
    NH, D = cfg['n_head'], cfg['n_embd']
    HD, NL = D // NH, cfg['n_layer']
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        B, T = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0, v1 = x, None
        cos, sin = rope_tables(T, HD, idx.device, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        cache = {}
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
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            if patch_head:
                for (pl, ph), val in patch_head.items():
                    if pl == li:
                        yh4 = yh4.clone()
                        yh4[:, :, ph, :] = val
            if collect:
                cache[('h', li)] = yh4.detach().clone()
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            mo = blk.mlp(F.rms_norm(x, (D,)))
            if patch_mlp and li in patch_mlp:
                mo = patch_mlp[li]
            if collect:
                cache[('m', li)] = mo.detach().clone()
            x = x + mo
            if resid_fn is not None:
                x = resid_fn(li, x)
            if collect:
                cache[('r', li)] = x.detach().clone()
        lg = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30).float()
    return lg, cache


def load_stimuli():
    return json.load(open(f'{HERE}/stimuli.json'))


def pairs_tensors(stim, split=None, family=None):
    """Returns clean_idx, corr_idx (N,6), clean_ans, corr_ans (N,) tensors."""
    rows = [r for r in stim['pairs']
            if (split is None or r['split'] == split)
            and (family is None or r['family'] == family)]
    ci = torch.tensor([r['clean_tokens'] for r in rows], device=DEV)
    xi = torch.tensor([r['corr_tokens'] for r in rows], device=DEV)
    ca = torch.tensor([r['clean_ans'] for r in rows], device=DEV)
    xa = torch.tensor([r['corr_ans'] for r in rows], device=DEV)
    return ci, xi, ca, xa, rows
