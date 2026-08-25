"""swarm_lib — verified GPU primitives for the datapoint-first circuit swarm (2026-08-25).

THE ALGORITHM THE WORKERS RUN (user-specified):
  1. well_predicted(): sample positions the model predicts extremely well.
  2. head_dce()/mlp_dce(): find the causal components AT THAT ONE POSITION —
     at OV grain (the head's payload), QK grain (the head's selection pattern),
     and MLP-layer grain. "You might not want to remove the full head."
  3. specificity_and_scope(): remove the ONE named part and run on MORE data —
     does it affect only this datapoint's kind? What is the CATEGORY of positions
     it actually serves (the circuit's true scope)?
Workers interpret; this library computes. Behavior frozen — extend by adding
functions (census_lib convention). All CE in nats, fp32, batch 8, T=256.

Grain definitions (bilin18: 18 layers x 9 heads x 128 dims):
  ('head_ov', L, h): the head's output slice y_h -> its batch-mean (payload removed,
      selection irrelevant). The standard head ablation of §1284-1360.
  ('head_qk', L, h): the head's PATTERN -> its mean pattern over reference rows,
      values kept live (selection removed, payload machinery intact).
  ('mlp', L): the MLP's output -> batch mean.
"""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; NH = 9; NL = 18
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
_C = {}


@torch.no_grad()
def _forward(idx, spec=None):
    """Manual bilin18 forward with optional single-part ablation `spec`."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, NH, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, NH, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, NH, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, NH, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, NH, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        if spec is not None and spec[0] == 'head_qk' and spec[1] == L:
            pat = pat.clone()
            pat[:, spec[2]] = _C['meanpat'][L][spec[2]].unsqueeze(0)
        v = at.c_v(xin).view(B, T, NH, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        if spec is not None and spec[0] == 'head_ov' and spec[1] == L:
            y = y.clone()
            y[:, :, spec[2]] = _C['ymean'][L][spec[2]].to(y.dtype)
        x = xm + at.c_proj(y.reshape(B, T, D))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if spec is not None and spec[0] == 'mlp' and spec[1] == L:
            mo = _C['mlpmean'][L].to(mo.dtype).expand_as(mo)
        x = x + mo
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def _ensure_refs():
    """Reference means (24 rows): per-head y-means, mean patterns, MLP means."""
    if 'ymean' in _C:
        return
    rows = cl.fineweb_rows(24)[:, :T + 1].contiguous()
    ys = {L: [] for L in range(NL)}
    ps = {L: [] for L in range(NL)}
    ms = {L: [] for L in range(NL)}
    x = None
    for i in range(0, 24, 4):
        idx = rows[i:i + 4, :-1].to(DEV).contiguous()
        xx = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = xx; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * xx + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            cos, sin = at.rotary(at.c_q(xin).view(B, T, NH, 128))
            q = are(F.rms_norm(at.c_q(xin).view(B, T, NH, 128), (128,)), cos, sin)
            k = are(F.rms_norm(at.c_k(xin).view(B, T, NH, 128), (128,)), cos, sin)
            q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, NH, 128), (128,)), cos, sin)
            k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, NH, 128), (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            v = at.c_v(xin).view(B, T, NH, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            ys[L].append(y.float().mean((0, 1)).cpu())      # (9,128)? mean over B,T -> (9,128)
            ps[L].append(pat.float().mean(0).cpu())          # (9,T,T)
            xx = xm + at.c_proj(y.reshape(B, T, D))
            mo = blk.mlp(F.rms_norm(xx, (D,)))
            ms[L].append(mo.float().mean((0, 1)).cpu())
            xx = xx + mo
    _C['ymean'] = {L: torch.stack(ys[L]).mean(0).to(DEV) for L in range(NL)}
    _C['meanpat'] = {L: torch.stack(ps[L]).mean(0).to(DEV) for L in range(NL)}
    _C['mlpmean'] = {L: torch.stack(ms[L]).mean(0).to(DEV) for L in range(NL)}


@torch.no_grad()
def _rows(n=480, skip=400):
    key = ('rows', n, skip)
    if key not in _C:
        _C[key] = cl.fineweb_rows(n, skip=skip)[:, :T + 1].contiguous()
    return _C[key]


@torch.no_grad()
def _base_ce(rows):
    key = ('bce', rows.shape[0], int(rows[0, 0]))
    if key not in _C:
        out = []
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = _forward(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            out.append(ce.cpu())
        _C[key] = torch.cat(out)
    return _C[key]


def context(rows, r, p, back=14):
    e = cl.enc()
    pre = e.decode(rows[r, max(0, p - back):p + 1].tolist())
    tgt = e.decode([int(rows[r, p + 1])])
    return pre, tgt


@torch.no_grad()
def well_predicted(n=20, ce_max=0.05, seed=0, nrows=480):
    """Positions (past burn-in 64) the model predicts with CE < ce_max, sampled."""
    rows = _rows(nrows)
    bce = _base_ce(rows)
    mask = bce < ce_max
    mask[:, :64] = False
    pos = torch.nonzero(mask)
    g = torch.Generator().manual_seed(seed)
    sel = pos[torch.randperm(pos.shape[0], generator=g)[:n]]
    out = []
    for r, p in sel.tolist():
        pre, tgt = context(rows, r, p)
        out.append({'row': r, 'pos': p, 'ce': round(float(bce[r, p]), 4),
                    'context': pre, 'target': tgt})
    return out


@torch.no_grad()
def head_dce(row, pos, grain='ov', nrows=480):
    """dCE at ONE position for every head at the given grain ('ov'|'qk').
    Returns list of (L, h, dce) sorted desc. ~7s per grain."""
    _ensure_refs()
    rows = _rows(nrows)
    idx = rows[row:row + 1, :-1].to(DEV).contiguous()
    tgt = int(rows[row, pos + 1])
    base = _base_ce(rows)[row, pos]
    res = []
    kind = 'head_ov' if grain == 'ov' else 'head_qk'
    for L in range(NL):
        for h in range(NH):
            lo = _forward(idx, (kind, L, h)).float()
            ce = F.cross_entropy(lo[0, pos].unsqueeze(0), torch.tensor([tgt], device=DEV))
            res.append((L, h, round(float(ce - base), 4)))
    res.sort(key=lambda t: -t[2])
    return res


@torch.no_grad()
def mlp_dce(row, pos, nrows=480):
    """dCE at ONE position for each MLP layer (mean-ablated). Sorted desc."""
    _ensure_refs()
    rows = _rows(nrows)
    idx = rows[row:row + 1, :-1].to(DEV).contiguous()
    tgt = int(rows[row, pos + 1])
    base = _base_ce(rows)[row, pos]
    res = []
    for L in range(NL):
        lo = _forward(idx, ('mlp', L)).float()
        ce = F.cross_entropy(lo[0, pos].unsqueeze(0), torch.tensor([tgt], device=DEV))
        res.append((L, round(float(ce - base), 4)))
    res.sort(key=lambda t: -t[1])
    return res


@torch.no_grad()
def specificity_and_scope(spec, nrows=96, topk=30, skip=900):
    """Remove ONE part model-wide over FRESH rows (disjoint skip); return the
    category of what it serves: top-k hurt positions with contexts, plus stats.
    spec = ('head_ov'|'head_qk', L, h) or ('mlp', L)."""
    _ensure_refs()
    rows = cl.fineweb_rows(nrows, skip=skip)[:, :T + 1].contiguous()
    dces = []
    for i in range(0, nrows, 8):
        bb = rows[i:i + 8].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        lo_b = _forward(idx).float()
        ce_b = F.cross_entropy(lo_b.reshape(-1, lo_b.shape[-1]), tg.reshape(-1),
                               reduction='none').view(tg.shape)
        lo_a = _forward(idx, tuple(spec)).float()
        ce_a = F.cross_entropy(lo_a.reshape(-1, lo_a.shape[-1]), tg.reshape(-1),
                               reduction='none').view(tg.shape)
        dces.append((ce_a - ce_b).cpu())
    dce = torch.cat(dces)
    dce[:, :64] = 0.0
    flat = dce.flatten()
    top = flat.topk(topk)
    examples = []
    for v, fi in zip(top.values.tolist(), top.indices.tolist()):
        r, p = fi // dce.shape[1], fi % dce.shape[1]
        pre, tgt = context(rows, r, p)
        examples.append({'dce': round(v, 3), 'context': pre[-70:], 'target': tgt})
    n = int((dce != 0).sum())
    return {'spec': list(spec), 'n_positions': n,
            'mean_dce': round(float(dce.sum() / max(n, 1)), 5),
            'frac_gt_0.5': round(float((dce > 0.5).float().sum() / max(n, 1)), 5),
            'frac_lt_-0.5': round(float((dce < -0.5).float().sum() / max(n, 1)), 5),
            'top_examples': examples}
