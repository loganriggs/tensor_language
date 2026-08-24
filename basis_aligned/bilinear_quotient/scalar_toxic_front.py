# scalar_toxic_front: the §1233 scope note, executed. In bilin12, masking the front
# non-core heads ON TOP of the 12-head core mask RECOVERS 0.102 nats (§1226) — their
# long-range reads are net harmful when the core is blind. A per-head scalar on those
# heads' OUTPUTS can interpolate between "live" (s=1) and "gone" (s=0), so a learned s
# should recover at least what full suppression did — and possibly more (partial
# suppression / sign flips available). Folds into c_proj: zero added description cost.
#
# Setup: bilin12, prose rows, core-12 read-masked @W64 (§1225 CORE). Learn one scalar per
# FRONT NON-CORE head (L0-3, heads not in the core: ~22 scalars, init 1.0, L2 prior to 1.0,
# 40 steps, 32 fit rows), scaling those heads' pre-c_proj outputs. Eval on 16 disjoint rows.
#
# Registered predictions:
#   pred_a SCALARS >= MASKING: held-out CE(core-masked + scalars) <= CE(core+front-masked)
#          + 0.01 — the optimizer finds at least the s=0 solution's value.
#   pred_b SUPPRESSION IS THE MECHANISM: mean learned s over L0/L3 non-core heads < 0.7
#          (the §1227 toxic layers get suppressed), while L1/L2 non-core mean s >= 0.7.
#   pred_c CONTROL: same training with scalars on LATE non-core heads (L8-11) instead
#          recovers < 30% of what front scalars recover (§1226: late reads are HELPFUL —
#          nothing to suppress).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'scalar_toxic_front_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12', device=DEV, dtype=torch.float32); mdl.eval()
D = 768; NH = 6; HD = 128; NL = 12; T = 256; WIN = 64; QSTART = 128
NTRAIN = 32; NEVAL = 16; STEPS = 40; LR = 0.02; PRIOR = 0.5
V12 = int(mdl.lm_head.weight.shape[0])
CORE = [(2,1),(5,5),(5,1),(7,0),(7,5),(10,0),(11,3),(10,2),(11,2),(2,3),(8,0),(5,0)]
CORESET = set(CORE)

MASK_W = None
FULL = None


def forward_core_scaled(idx, scal_heads, scal):
    """Core heads read-masked @WIN; heads in scal_heads get output scalar from scal."""
    dt = mdl.transformer.wte.weight.dtype
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    core_by_layer = {}
    for L, h in CORE:
        core_by_layer.setdefault(L, set()).add(h)
    for L, blk in enumerate(mdl.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hs = F.rms_norm(x, (D,))
        q = apply_rot(F.rms_norm(a.c_q(hs).view(B, T, NH, HD), (HD,)), cos, sin)
        k = apply_rot(F.rms_norm(a.c_k(hs).view(B, T, NH, HD), (HD,)), cos, sin)
        s = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / HD
        heads = core_by_layer.get(L, None)
        if heads is None:
            msk = FULL.expand(NH, T, T)
        else:
            msk = torch.stack([MASK_W if h in heads else FULL for h in range(NH)], 0)
        pat = s.square().masked_fill(~msk.unsqueeze(0), 0.0)
        pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
        v = a.c_v(hs).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        vv = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        lidx = [j for j, (LL, hh) in enumerate(scal_heads) if LL == L]
        if lidx:
            hpos = torch.tensor([scal_heads[j][1] for j in lidx], device=y.device)
            sv = torch.ones(NH, device=y.device, dtype=scal.dtype)
            sv = sv.index_copy(0, hpos, scal[torch.tensor(lidx, device=y.device)])
            y = y * sv.view(1, 1, NH, 1).to(y.dtype)
        x = x + a.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)


def ce_rows(rows, scal_heads, scal):
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, rows.shape[0], 4):
            bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lo = forward_core_scaled(idx, scal_heads, scal).float()
            tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                         tgt[:, qp].reshape(-1), reduction='sum'))
            n += idx.shape[0] * len(qp)
    return tot / n


def train_scalars(train, scal_heads):
    qp = torch.arange(QSTART, T, device=DEV)
    scal = torch.ones(len(scal_heads), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([scal], lr=LR)
    for step in range(STEPS):
        i = (step * 4) % NTRAIN
        bb = train[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_core_scaled(idx, scal_heads, scal).float()
        loss = F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgt[:, qp].reshape(-1)) \
            + PRIOR * ((scal - 1.0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return scal.detach()


def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    for p in mdl.parameters():
        p.requires_grad_(False)
    ROWS = cl.fineweb_rows(NTRAIN + NEVAL)[:, :T + 1].contiguous().clamp_max(V12 - 1)
    train, ev = ROWS[:NTRAIN], ROWS[NTRAIN:]

    FRONT = [(L, h) for L in range(4) for h in range(NH) if (L, h) not in CORESET]
    LATE = [(L, h) for L in range(8, 12) for h in range(NH) if (L, h) not in CORESET]

    ce_core = ce_rows(ev, [], torch.ones(0, device=DEV))
    # core+front fully masked reference: emulate with scalars fixed at 0? No — §1226 masked READS,
    # scalars kill OUTPUTS. Use s=0 on front as the in-instrument reference (output-suppression).
    ce_front0 = ce_rows(ev, FRONT, torch.zeros(len(FRONT), device=DEV))
    print(f"eval core-mask {ce_core:.4f} | +front s=0 {ce_front0:.4f}", flush=True)

    s_front = train_scalars(train, FRONT)
    ce_front_s = ce_rows(ev, FRONT, s_front)
    s_late = train_scalars(train, LATE)
    ce_late_s = ce_rows(ev, LATE, s_late)

    rec_front = ce_core - ce_front_s
    rec_front0 = ce_core - ce_front0
    rec_late = ce_core - ce_late_s
    by = {}
    for j, (L, h) in enumerate(FRONT):
        by.setdefault(L, []).append(float(s_front[j]))
    mean_s = {f'L{L}': round(sum(v) / len(v), 3) for L, v in by.items()}

    out = {'model': 'bilin12', 'n_train': NTRAIN, 'n_eval': NEVAL, 'W': WIN,
           'ce': {'core_mask': round(ce_core, 4), 'front_s0': round(ce_front0, 4),
                  'front_learned': round(ce_front_s, 4), 'late_learned': round(ce_late_s, 4)},
           'recovery': {'front_learned': round(rec_front, 4), 'front_s0': round(rec_front0, 4),
                        'late_learned': round(rec_late, 4)},
           'front_mean_s_by_layer': mean_s,
           'front_scalars': {f'{L}.{h}': round(float(s_front[j]), 3) for j, (L, h) in enumerate(FRONT)},
           'pred_a_scalars_geq_masking': bool(ce_front_s <= ce_front0 + 0.01),
           'pred_b_suppression': bool((mean_s.get('L0', 1) + mean_s.get('L3', 1)) / 2 < 0.7 <=
                                      (mean_s.get('L1', 1) + mean_s.get('L2', 1)) / 2),
           'pred_c_late_control': bool(rec_late < 0.3 * max(rec_front, 1e-9)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recoveries {out['recovery']} | mean s {mean_s}")
    print(f"pred_a geq {out['pred_a_scalars_geq_masking']} | pred_b suppress {out['pred_b_suppression']} | pred_c late {out['pred_c_late_control']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
