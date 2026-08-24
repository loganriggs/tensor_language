# scalar_recalibrate2: the regularized retry of §1231's user-proposed instrument. v1
# (8 rows, no prior, 60 steps) overfit to NEGATIVE held-out recovery. v2: 32 fit rows,
# L2 prior pulling scalars to 1.0 (strength 1.0), LR 0.01, 40 steps — the prior means only
# heads with consistent gradient signal move. Same design otherwise (all-162 read-mask
# @W64 on prose; eval on 16 disjoint rows; permuted-scalar null).
#
# Registered predictions:
#   pred_a FREE RECOVERY: held-out gap shrinks >= 10% (bar lowered from v1's 20% —
#          stated plainly).
#   pred_b FIX AT THE CUTS: core-head mean |s-1| >= 2x other heads'.
#   pred_c PERM NULL <= 5% recovery.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'scalar_recalibrate2_results.json'
NTRAIN = 32; NEVAL = 16; WIN = 64; QSTART = 128; STEPS = 40; LR = 0.01; PRIOR = 1.0
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
CORE = {(2,5),(3,8),(5,5),(1,4),(7,0),(8,3),(7,3),(6,1),(6,7),(5,1),(13,0),(8,4)}

MASK_W = None
FULL_TRIL = None


def make_masks():
    global MASK_W, FULL_TRIL
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL_TRIL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL_TRIL & vis


def forward_masked_scaled(idx, scal):
    """All heads read-masked @WIN; per-head output scalars scal (18,9) or None."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        pat = pat.masked_fill(~MASK_W, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        if scal is not None:
            y = y * scal[L].view(1, 1, 9, 1)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def full_forward(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def ce_eval(rows, scal, masked=True):
    tot = 0.0; n = 0
    qp = torch.arange(QSTART, T, device=DEV)
    with torch.no_grad():
        for i in range(0, rows.shape[0], 4):
            bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lo = (forward_masked_scaled(idx, scal) if masked else full_forward(idx)).float()
            tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                         tgt[:, qp].reshape(-1), reduction='sum'))
            n += idx.shape[0] * len(qp)
    return tot / n


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    make_masks()
    for p in m.parameters():
        p.requires_grad_(False)
    ROWS = cl.fineweb_rows(NTRAIN + NEVAL)[:, :T + 1].contiguous()
    train, ev = ROWS[:NTRAIN], ROWS[NTRAIN:]
    qp = torch.arange(QSTART, T, device=DEV)

    base = ce_eval(ev, None, masked=False)
    masked0 = ce_eval(ev, None, masked=True)
    gap0 = masked0 - base
    print(f"eval base {base:.4f} | masked {masked0:.4f} | gap {gap0:.4f}", flush=True)

    scal = torch.ones(18, 9, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([scal], lr=LR)
    for step in range(STEPS):
        i = (step * 4) % NTRAIN
        bb = train[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_masked_scaled(idx, scal).float()
        loss = F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgt[:, qp].reshape(-1)) \
            + PRIOR * ((scal - 1.0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            print(f"step {step}: train CE {float(loss):.4f}", flush=True)
    S = scal.detach()

    masked_s = ce_eval(ev, S, masked=True)
    rec = (masked0 - masked_s) / gap0
    g = torch.Generator().manual_seed(3)
    recs_perm = []
    for _ in range(3):
        perm = torch.randperm(162, generator=g)
        Sp = S.view(-1)[perm].view(18, 9)
        recs_perm.append((masked0 - ce_eval(ev, Sp, masked=True)) / gap0)
    rec_perm = sum(recs_perm) / 3

    dev = (S - 1.0).abs()
    core_dev = float(torch.stack([dev[L, h] for L, h in CORE]).mean())
    other = [dev[L, h] for L in range(18) for h in range(9) if (L, h) not in CORE]
    other_dev = float(torch.stack(other).mean())

    out = {'n_train': NTRAIN, 'n_eval': NEVAL, 'W': WIN, 'steps': STEPS,
           'eval_base': round(base, 4), 'eval_masked': round(masked0, 4),
           'eval_masked_scaled': round(masked_s, 4),
           'gap': round(gap0, 4), 'recovery_frac': round(float(rec), 4),
           'recovery_perm_null': round(float(rec_perm), 4),
           'core_mean_abs_dev': round(core_dev, 4), 'other_mean_abs_dev': round(other_dev, 4),
           'scalar_extremes': {f'{L}.{h}': round(float(S[L, h]), 3)
                               for L in range(18) for h in range(9)
                               if abs(float(S[L, h]) - 1) > 0.15},
           'pred_a_free_recovery': bool(rec >= 0.10),
           'pred_b_fix_at_cuts': bool(core_dev >= 2 * other_dev),
           'pred_c_perm_null_fails': bool(rec_perm <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recovery {float(rec):.3f} (perm null {float(rec_perm):.3f}) | core dev {core_dev:.3f} vs other {other_dev:.3f}")
    print(f"extremes {out['scalar_extremes']}")
    print(f"pred_a rec {out['pred_a_free_recovery']} | pred_b cuts {out['pred_b_fix_at_cuts']} | pred_c null {out['pred_c_perm_null_fails']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
