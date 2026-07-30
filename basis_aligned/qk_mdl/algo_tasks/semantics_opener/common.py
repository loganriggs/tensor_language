"""Shared utilities for the pending-opener semantics verification.

Model: bilin18 (18L/9H/D=1152, NO softmax, pattern=(q1.k1)(q2.k2)/128^2 causal
unnormalized, bilinear MLP). Forward copied from the verified hand-written
prefix/suffix in algo_tasks/bracket/s3_das.py (which matches
tier2_model.reference_forward).

Channel: r-dim orthonormal subspace Q of the residual stream ENTERING layer 13
(after layer 12's MLP add, before block 13's lambda mixing) — same space as the
bracket DAS.

Coded opener-state: byte-level tracker over the raw token stream (independent
knowledge, no model involved): depth of ( ) [ ] { }, ASCII double-quote parity,
and curly-quote depth. State at position t = after consuming token t (the state
relevant for predicting token t+1).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot  # noqa: E402

DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_opener'
BRACKET = '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/bracket'
L = 13          # intervention layer (residual entering layer 13)
BATCH = 6       # GPU shared with other agents

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_elriggs('bilin18')
    return _model


def gpu_free_mb():
    free, total = torch.cuda.mem_get_info()
    return free / 2**20


def safe(fn, *a, **k):
    """retry OOM after 60 s (GPU shared with 2 other agents)."""
    for attempt in range(6):
        try:
            return fn(*a, **k)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'OOM (free {gpu_free_mb():.0f} MB), retry in 60 s', flush=True)
            time.sleep(60)
    return fn(*a, **k)


# ---------------------------------------------------------------- forward ----
def _layer(m, cfg, x, x0, v1, li, cosb, sinb, mask):
    NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
    B, T = x.shape[:2]
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
    return x, v1


@torch.no_grad()
def forward_hooked(m, cfg, idx, hook=None, stop_at_L=False):
    """Full forward with an optional intervention on the residual entering
    layer L. hook(x) -> x_new, applied to the (B,T,D) residual after layer
    L-1's MLP, before block L's lambdas. stop_at_L: return that residual
    instead of logits (activation collection)."""
    NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
    NL = len(m.transformer.h)
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x
    v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        if li == L:
            if stop_at_L:
                return x
            if hook is not None:
                x = hook(x)
        x, v1 = _layer(m, cfg, x, x0, v1, li, cosb, sinb, mask)
    return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)


def sub_hook(Q, coded):
    """Replace channel activation with coded value:
    x <- x - Q Q^T x + Q c   where coded is (B,T,r) (or None => zero)."""
    def h(x):
        a = torch.einsum('btd,dr->btr', x, Q)
        repl = torch.zeros_like(a) if coded is None else coded.to(a.dtype)
        return x + torch.einsum('btr,dr->btd', repl - a, Q)
    return h


def scale_hook(Q, s):
    def h(x):
        a = torch.einsum('btd,dr->btr', x, Q)
        return x + (s - 1.0) * torch.einsum('btr,dr->btd', a, Q)
    return h


# ------------------------------------------------------------- CE metrics ----
@torch.no_grad()
def ce_per_token(m, cfg, rows, hook=None, batch=BATCH):
    """rows: (N,T) int array. Returns (N, T-1) per-token CE (float32 cpu)."""
    outs = []
    for i in range(0, len(rows), batch):
        idx = torch.from_numpy(np.ascontiguousarray(rows[i:i + batch]).astype(np.int64)).to(DEV)
        lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
        ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                             idx[:, 1:].reshape(-1), reduction='none')
        outs.append(ce.view(idx.shape[0], -1).cpu())
        del lg
    return torch.cat(outs).numpy()


@torch.no_grad()
def closer_logprobs(m, cfg, rows, closer_ids, hook=None, batch=BATCH):
    """log p(closer) at every position. Returns (N, T, n_closers)."""
    outs = []
    for i in range(0, len(rows), batch):
        idx = torch.from_numpy(np.ascontiguousarray(rows[i:i + batch]).astype(np.int64)).to(DEV)
        lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
        lp = F.log_softmax(lg, -1)[:, :, closer_ids]
        outs.append(lp.cpu())
        del lg
    return torch.cat(outs).numpy()


@torch.no_grad()
def collect_activations(m, cfg, rows, Q, batch=BATCH):
    """channel activation Q^T x at layer-L entry, all positions. (N,T,r)."""
    outs = []
    for i in range(0, len(rows), batch):
        idx = torch.from_numpy(np.ascontiguousarray(rows[i:i + batch]).astype(np.int64)).to(DEV)
        x = safe(forward_hooked, m, cfg, idx, stop_at_L=True)
        outs.append(torch.einsum('btd,dr->btr', x, Q).float().cpu())
        del x
    return torch.cat(outs).numpy()


# ---------------------------------------------------- coded opener state -----
_tok = None
_vocab_bytes = None


def _bytes_to_unicode():
    """standard GPT-2 byte<->unicode mapping (transformers 5.x dropped
    byte_decoder, so reproduce it)."""
    bs = (list(range(ord('!'), ord('~') + 1)) + list(range(0xa1, 0xad))
          + list(range(0xae, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def vocab_bytes():
    """id -> raw bytes for every gpt2 token (byte-level BPE decoded exactly)."""
    global _tok, _vocab_bytes
    if _vocab_bytes is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
        bd = {v: k for k, v in _bytes_to_unicode().items()}
        _vocab_bytes = {}
        for tstr, tid in _tok.get_vocab().items():
            _vocab_bytes[tid] = bytes(bd[c] for c in tstr)
    return _vocab_bytes


def get_tok():
    vocab_bytes()
    return _tok

FEATS = ['p_depth', 'b_depth', 'c_depth', 'q_par', 'q_curly']
CURLY_OPEN = '“'.encode('utf-8')   # b'\xe2\x80\x9c'
CURLY_CLOSE = '”'.encode('utf-8')  # b'\xe2\x80\x9d'


def coded_state(ids_row, clip=8):
    """Track opener state over the byte stream of a token row.
    Returns (T, 5) int array: state AFTER each token.
    Features: paren depth, square-bracket depth, curly-brace depth,
    ASCII double-quote parity, curly-quote depth."""
    vb = vocab_bytes()
    p = b = c = qc = 0
    qp = 0
    out = np.zeros((len(ids_row), 5), dtype=np.int64)
    carry = b''
    for t, tid in enumerate(ids_row):
        stream = carry + vb[int(tid)]
        i = 0
        consumed_end = len(stream)
        events = []
        while i < len(stream):
            ch = stream[i:i + 1]
            if stream[i:i + 3] == CURLY_OPEN:
                events.append('qc+'); i += 3; continue
            if stream[i:i + 3] == CURLY_CLOSE:
                events.append('qc-'); i += 3; continue
            if len(stream) - i < 3 and stream[i] == 0xe2:
                # possible split curly quote -> leave for carry
                consumed_end = i
                break
            if ch == b'(':
                events.append('p+')
            elif ch == b')':
                events.append('p-')
            elif ch == b'[':
                events.append('b+')
            elif ch == b']':
                events.append('b-')
            elif ch == b'{':
                events.append('c+')
            elif ch == b'}':
                events.append('c-')
            elif ch == b'"':
                events.append('q~')
            i += 1
        carry = stream[consumed_end:] if consumed_end < len(stream) else b''
        for e in events:
            if e == 'p+':
                p = min(p + 1, clip)
            elif e == 'p-':
                p = max(p - 1, 0)
            elif e == 'b+':
                b = min(b + 1, clip)
            elif e == 'b-':
                b = max(b - 1, 0)
            elif e == 'c+':
                c = min(c + 1, clip)
            elif e == 'c-':
                c = max(c - 1, 0)
            elif e == 'q~':
                qp ^= 1
            elif e == 'qc+':
                qc = min(qc + 1, clip)
            elif e == 'qc-':
                qc = max(qc - 1, 0)
        out[t] = (p, b, c, qp, qc)
    return out


def coded_states(rows):
    """(N,T) ids -> (N,T,5)."""
    return np.stack([coded_state(r) for r in rows])


def derived(states):
    """add derived features: total bracket depth, any-open binary, quote-any."""
    p, b, c, qp, qc = [states[..., i] for i in range(5)]
    tot = p + b + c
    qany = ((qp + qc) > 0).astype(np.int64)
    anyo = ((tot > 0) | (qany > 0)).astype(np.int64)
    return {'p_depth': p, 'b_depth': b, 'c_depth': c, 'q_par': qp,
            'q_curly': qc, 'tot_depth': tot, 'q_any': qany, 'any_open': anyo}


# ------------------------------------------------------------------ data -----
def fineweb_audit():
    a = np.load('/workspace/tensor_language/data_fineweb_tokens.npy', mmap_mode='r')
    return np.asarray(a[448:600]).astype(np.int64)   # HELD-BACK final numbers


def cooc(rows):
    a = np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy', mmap_mode='r')
    return np.asarray(a[rows[0]:rows[1]]).astype(np.int64)


def paired_dce(ce_cond, ce_base):
    """paired per-token dCE with token-level SE and sequence-clustered SE."""
    d = (ce_cond - ce_base).astype(np.float64)
    flat = d.ravel()
    seq = d.mean(1)
    return {'dce': float(flat.mean()),
            'se_token': float(flat.std(ddof=1) / np.sqrt(flat.size)),
            'se_seq': float(seq.std(ddof=1) / np.sqrt(seq.size)),
            'n_tokens': int(flat.size)}
