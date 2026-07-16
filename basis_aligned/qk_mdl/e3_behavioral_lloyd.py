"""Method E, experiment 3 — BEHAVIORAL LLOYD PILOT (Logan-approved).
Bottom 12 streams (attn/mlp layers 0-5), k=64, windowed-D W=4 harness.
Loop (4 iterations):
  1. gradient pass through the PATCHED model: accumulate dLoss/d(table row)
     per stream row (leaf = creation-time gathered rows; lambda-rescale chain
     and all read layers aggregate into it), ~64k tokens;
  2. score every candidate move first-order: delta(t, c') = g_t . (C[c'] - C[a_t]);
     apply the most-improving moves only (damping: top 10% per stream, and
     predicted delta < 0);
  3. recompute centroids as activation-space means of members;
  4. held-out audit (early region); final iteration also audits the late region.
Start = best-of-3-seeds L2 partition (+0.1034, SEED_OFF=1555). Upper streams
(layers 6+) keep their plain L2 vq64 tables, fixed.
Known approximations (logged): first-order scores; greedy simultaneous moves vs
fixed centroids; Zipf-noisy gradients for rare rows (min-count filter 8)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/e3_behavioral_lloyd.json'
W = 4
K = 64
SEED_OFF = 1555
ITERS = 6
MOVE_FRAC = 0.02
MIN_COUNT = 8
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 6144, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:1044]
AUDIT_LATE = ALL[-16:]
PILOT = [f'{t}{l}' for l in range(6) for t in ('attn', 'mlp')]


def created_layer(nm):
    return int(nm[4:]) if nm.startswith('attn') else int(nm[3:])


RAW = torch.load(f'{QK}/stream_tables.pt')
X = {nm: t.float().to(DEV) for nm, t in RAW.items()}

# token counts (for the min-count gradient filter)
cnt = torch.zeros(V)
for i in range(0, len(TRAIN), 64):
    flat = TRAIN[i:i + 64, :-1].reshape(-1)
    cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
freq_ok = (cnt >= MIN_COUNT).to(DEV)


def kmeans_assign(nm):
    g = torch.Generator(); g.manual_seed((hash(nm) + SEED_OFF) % 2**31)
    Xs = X[nm]
    C = Xs[torch.randperm(len(Xs), generator=g)[:K].to(DEV)].clone()
    for _ in range(10):
        a_ = torch.empty(V, dtype=torch.long, device=DEV)
        for i in range(0, V, 2048):
            xx = Xs[i:i + 2048]
            a_[i:i + 2048] = ((xx * xx).sum(1, True) - 2 * xx @ C.T
                              + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        c2 = torch.zeros(K, device=DEV)
        Cn.index_add_(0, a_, Xs)
        c2.index_add_(0, a_, torch.ones(V, device=DEV))
        nz = c2 > 0
        C[nz] = Cn[nz] / c2[nz][:, None]
    return a_


def centroids_from(assign, nm):
    C = torch.zeros(K, D, device=DEV)
    c2 = torch.zeros(K, device=DEV)
    C.index_add_(0, assign, X[nm])
    c2.index_add_(0, assign, torch.ones(V, device=DEV))
    nz = c2 > 0
    C[nz] = C[nz] / c2[nz][:, None]
    return C

assigns, cents = {}, {}
for nm in RAW:
    assigns[nm] = kmeans_assign(nm)
    cents[nm] = centroids_from(assigns[nm], nm)
print('initial L2 vq64 partition built (seed3)', flush=True)


def table_rows(nm):
    # fp16 round-trip to match the e-series storage exactly (comparability)
    return cents[nm][assigns[nm]].half().float()         # (V, D) on DEV


def run_model(idx, grad_streams=None):
    """windowed-D W=4 forward; if grad_streams, creation-time gathered rows for
    those streams are grad leaves; returns (logits, leaves dict)."""
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    streams, tabs, leaves = {}, {}, {}
    for li, blk in enumerate(m.transformer.h):
        lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
        x = lam0 * x + lam1 * x0
        for nm in streams:
            streams[nm] = lam0 * streams[nm]
            tabs[nm] = lam0 * tabs[nm]
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        h_qk = h
        if li >= 1:
            old = [nm for nm in streams if created_layer(nm) < li - W]
            if old:
                xp = x
                for nm in old:
                    xp = xp - streams[nm] + tabs[nm]
                h_qk = F.rms_norm(xp, (xp.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h_qk).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + attn_out
        streams[f'attn{li}'] = attn_out
        gath = table_rows(f'attn{li}')[idx].detach()
        if grad_streams and f'attn{li}' in grad_streams:
            gath = gath.requires_grad_(True)
            leaves[f'attn{li}'] = gath
        tabs[f'attn{li}'] = gath
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        streams[f'mlp{li}'] = mlp_out
        gath = table_rows(f'mlp{li}')[idx].detach()
        if grad_streams and f'mlp{li}' in grad_streams:
            gath = gath.requires_grad_(True)
            leaves[f'mlp{li}'] = gath
        tabs[f'mlp{li}'] = gath
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30), leaves


@torch.no_grad()
def audit(aud):
    tot, n = 0.0, 0
    for i in range(0, len(aud), 4):
        b = aud[i:i + 4].to(DEV)
        logits, _ = run_model(b[:, :-1])
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


@torch.no_grad()
def base_ce(aud):
    tot, n = 0.0, 0
    for i in range(0, len(aud), 4):
        b = aud[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


base = base_ce(AUDIT)
base_late = base_ce(AUDIT_LATE)
d0 = audit(AUDIT) - base
res = {'baseline': base, 'iter0_l2_seed3': d0, 'iters': []}
print(f'baseline {base:.4f}; iter-0 (L2 seed3): dCE {d0:+.4f}', flush=True)

g_tr = torch.Generator(); g_tr.manual_seed(9)
best_d = d0
best_state = ({nm: t.clone() for nm, t in assigns.items()},
              {nm: t.clone() for nm, t in cents.items()})
for it in range(1, ITERS + 1):
    # --- gradient pass ---
    rowgrad = {nm: torch.zeros(V, D, device=DEV) for nm in PILOT}
    torch.cuda.empty_cache()
    NBAT = 64
    order = torch.randperm(len(TRAIN), generator=g_tr)[:NBAT]
    for bi in range(NBAT):
        b = TRAIN[order[bi:bi + 1]].to(DEV)
        idx = b[:, :-1]
        logits, leaves = run_model(idx, grad_streams=set(PILOT))
        loss = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        loss.backward()
        flat = idx.reshape(-1)
        for nm, leaf in leaves.items():
            rowgrad[nm].index_add_(0, flat, leaf.grad.float().reshape(-1, D))
    # --- score + damped moves ---
    n_moves, pred_sum = 0, 0.0
    for nm in PILOT:
        g = rowgrad[nm]
        C = cents[nm]
        cur = assigns[nm]
        # delta(t, c') = g_t . C[c'] - g_t . C[cur_t]
        gC = g @ C.T                                     # (V, K)
        cur_val = gC.gather(1, cur[:, None]).squeeze(1)
        best_val, best_c = gC.min(1)
        gain = best_val - cur_val                        # <= 0 predicted improvement
        gain = torch.where(freq_ok, gain, torch.zeros_like(gain))
        cand = (gain < 0)
        n_take = min(int(V * MOVE_FRAC), int(cand.sum()))
        if n_take > 0:
            take = gain.topk(n_take, largest=False).indices
            assigns[nm][take] = best_c[take]
            n_moves += n_take
            pred_sum += gain[take].sum().item()
        cents[nm] = centroids_from(assigns[nm], nm)
    del rowgrad
    torch.cuda.empty_cache()
    d = audit(AUDIT) - base
    if d > best_d:                       # trust region: revert + halve step
        assigns = {nm: t.clone() for nm, t in best_state[0].items()}
        cents = {nm: t.clone() for nm, t in best_state[1].items()}
        MOVE_FRAC = MOVE_FRAC / 2
        status = f'REVERTED (frac now {MOVE_FRAC:.3f})'
    else:
        best_d = d
        best_state = ({nm: t.clone() for nm, t in assigns.items()},
                      {nm: t.clone() for nm, t in cents.items()})
        status = 'kept'
    res['iters'].append({'iter': it, 'moves': n_moves,
                         'pred_gain_sum': pred_sum, 'heldout_dce': d,
                         'status': status})
    print(f'iter {it}: {n_moves} moves, predicted {pred_sum:+.1f}, '
          f'held-out dCE {d:+.4f} [{status}]', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
assigns = best_state[0]
cents = best_state[1]
res['best_heldout_dce'] = best_d
dl = audit(AUDIT_LATE) - base_late
res['final_late_region_dce'] = dl
print(f'final late-region dCE {dl:+.4f}', flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
torch.save({nm: assigns[nm].cpu() for nm in assigns}, f'{QK}/e3_assigns.pt')
print('e3 behavioral lloyd done', flush=True)
