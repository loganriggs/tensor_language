"""Shared utilities for semantics_category verification of the bilin18 category-code directions.

Model: bilin18 (18L, 9H, 1152d, two-branch bilinear attention, UNNORMALIZED pattern).
Forward copied from qk_circuit_atlas.py, extended with:
  - edit_fn applied to the residual right AFTER block 3 (post-MLP3), i.e. the 'blk3' depth
    of qk_category_engine.py;
  - optional residual collection at named depths.

Category code (identical to qk_category_engine.py):
  0 subword, 1 punct, 2 capital, 3 digit, 4 funcword, 5 other — label of the NEXT token.

Data:
  exploration = cooc rows 0-2400 of data_fineweb_cooc_tokens.npy (6000,513)
  held-back audit = data_fineweb_tokens.npy rows 448:600 (600,513)
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/algo_tasks/semantics_category'
T_CTX = 128          # tokens per row used everywhere (127 predictions)
BATCH = 6            # GPU shared with 2 other agents

# ---------------- GPU guards ----------------

def wait_free(min_mb=3000, tries=30):
    for _ in range(tries):
        free, _tot = torch.cuda.mem_get_info()
        if free / 2**20 >= min_mb:
            return
        print(f"  [gpu] only {free/2**20:.0f}MB free; waiting 60s", flush=True)
        time.sleep(60)

def oom_retry(fn, *a, **kw):
    for attempt in range(6):
        try:
            return fn(*a, **kw)
        except torch.cuda.OutOfMemoryError:
            print(f"  [gpu] OOM (attempt {attempt}); sleeping 60s", flush=True)
            torch.cuda.empty_cache(); time.sleep(60)
    raise RuntimeError('OOM after 6 retries')

# ---------------- model + data ----------------

wait_free()
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
HELD_ROWS = range(448, 600)          # audit
EXPLORE_ROWS = range(0, 2400)        # exploration (cooc)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
CAT = torch.full((V,), 5, dtype=torch.long)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
CAT = CAT.to(DEV)
CATNAMES = ['subword', 'punct', 'capital', 'digit', 'funcword', 'other']
CATM = torch.stack([CAT == c for c in range(6)])  # (6,V) bool

EDIT_LAYER = 3   # edit applied to residual after block 3 (post-MLP3)

@torch.no_grad()
def forward(idx, edit_fn=None, collect=(), edit_layer=EDIT_LAYER):
    """idx (B,T). edit_fn: x(B,T,D)->x applied right after block `edit_layer`.
    collect: iterable of 'blk{li}' names -> returned dict of (B*T,D) residuals.
    Returns (logits(B,T,V) float, collected)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    got = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
        if li == edit_layer and edit_fn is not None:
            x = edit_fn(x)
        if f'blk{li}' in collect:
            got[f'blk{li}'] = x.reshape(-1, D).clone()
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
    return lg, got

def batches(data, rows, bs=BATCH):
    rows = list(rows)
    for i in range(0, len(rows), bs):
        yield data[rows[i:i+bs], :T_CTX].to(DEV)

# ---------------- metrics ----------------

def cat_mass(lg):
    """(B,T,V) logits -> (B,T,6) probability mass per category."""
    lp = F.log_softmax(lg, -1)
    return torch.stack([torch.logsumexp(lp[:, :, CATM[c]], -1) for c in range(6)], -1).exp()

def per_token_ce(lg, idx):
    """lg computed on idx[:, :-1] -> aligns 1:1 with targets idx[:, 1:]."""
    tgt = idx[:, 1:]
    return F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(tgt.shape)

def cat_within_ce(lg, idx):
    """exact split CE = categoryCE + withinCE (per token). lg computed on idx[:, :-1].
    Returns (cat_ce, within_ce) (B,T-1)."""
    tgt = idx[:, 1:]
    lp = F.log_softmax(lg, -1)
    ls_cat = torch.stack([torch.logsumexp(lp[:, :, CATM[c]], -1) for c in range(6)], -1)
    tc = CAT[tgt]
    cat_ce = -ls_cat.gather(-1, tc.unsqueeze(-1)).squeeze(-1)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    within_ce = -(tok_lp + cat_ce)
    return cat_ce, within_ce

def paired_stats(delta_flat, n_rows):
    """delta_flat: 1D per-token deltas ordered row-major. Returns mean, per-token SE, row-clustered SE."""
    d = delta_flat.double()
    mean = d.mean().item()
    se_tok = (d.std(unbiased=True) / d.numel()**0.5).item()
    rm = d.view(n_rows, -1).mean(1)
    se_row = (rm.std(unbiased=True) / n_rows**0.5).item()
    return mean, se_tok, se_row

def load_probe():
    p = torch.load(f'{OUT}/probe_blk3.pt', map_location=DEV)
    return p
