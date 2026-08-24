# bucket_constants: USER-PROPOSED — the rate-distortion ladder for head constants.
# The own-mean stand-in (§1091/§1093) conditions on NOTHING (0 bits per head per position).
# Allow log2(k) bits: replace each head's per-position output y_h with its POSITION-BUCKET
# mean (k position bins over 0..T), fit on TRAIN rows, evaluated held-out. k=1 is the §1093
# all-static condition; k=2 is "early vs late text"; k up to 16. If a small number of bits
# closes much of the static gap, the crowd's collective value is largely a POSITION-INDEXED
# bias schedule — nameable, cheap understanding. Prior art: slice-conditioned constants
# carried ~85% of three circuits (§240); §1093 found all-static (3.67) WORSE than
# all-zero (3.42) — watch whether any k crosses below all-zero.
#
# Conditions (ALL 162 heads replaced simultaneously): base; zero (all attention outputs 0);
# k1; k2; k4; k16; k4rand (4 buckets assigned by position hash, not contiguous ranges —
# parameter-matched control that carries no positional information beyond noise).
#
# Registered predictions:
#   pred_a MONOTONE: CE(k1) > CE(k2) > CE(k4) > CE(k16) (each bit helps).
#   pred_b POSITION IS THE FIRST BIT AND IT MATTERS: k2 recovers >= 15% of the k1-vs-base
#          gap; and some k <= 16 drops BELOW the all-zero condition (buckets beat deletion
#          where the frozen global mean did not, §1093).
#   pred_c RANDOM BUCKETS ARE k1: |CE(k4rand) - CE(k1)| <= 0.08 (it is the positional
#          content of the buckets, not the extra parameters).
# Fit/eval split: bucket means from TRAIN rows (24), CE on DISJOINT EVAL rows (24).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bucket_constants_results.json'
NTRAIN = 24; NEVAL = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
KS = [1, 2, 4, 16]


def buckets_for(k, rand=False):
    pos = torch.arange(T)
    if rand:
        g = torch.Generator().manual_seed(5)
        return torch.randint(0, k, (T,), generator=g)
    return (pos * k // T).clamp_max(k - 1)


@torch.no_grad()
def head_outputs(idx):
    """Per-layer pre-c_proj head outputs y (B,T,9,128) under the TRUE model."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    outs = []
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
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        outs.append(y.detach())
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return outs


@torch.no_grad()
def forward_bucketed(idx, means, bidx):
    """All heads' y replaced by bucket means (means: list of (k,9,128); bidx: (T,) bucket index).
    means=None with bidx=None -> zero condition."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if means is None:
            yrep = torch.zeros(B, T, 9, 128, device=DEV)
        else:
            yrep = means[L][bidx].unsqueeze(0).expand(B, T, 9, 128)
        x = xm + at.c_proj(yrep.reshape(B, T, D).to(at.c_proj.weight.dtype))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NTRAIN + NEVAL)[:, :T + 1].contiguous()
    train, ev = ROWS[:NTRAIN], ROWS[NTRAIN:]
    qp = torch.arange(QSTART, T, device=DEV)

    # accumulate per-position mean head outputs on TRAIN
    sums = [torch.zeros(T, 9, 128, dtype=torch.float64) for _ in range(18)]
    cnt = 0
    for i in range(0, NTRAIN, 4):
        idx = train[i:i + 4, :-1].to(DEV).contiguous()
        outs = head_outputs(idx)
        for L in range(18):
            sums[L] += outs[L].double().mean(0).cpu()
        cnt += 1
    posmean = [(s / cnt) for s in sums]        # (T,9,128) per layer

    def bucket_means(k, rand=False):
        b = buckets_for(k, rand)
        ms = []
        for L in range(18):
            mk = torch.zeros(k, 9, 128, dtype=torch.float64)
            for j in range(k):
                sel = (b == j)
                mk[j] = posmean[L][sel].mean(0)
            ms.append(mk.float().to(DEV))
        return ms, b.to(DEV)

    def ce_cond(means, bidx):
        tot = 0.0; n = 0
        for i in range(0, NEVAL, 4):
            bb = ev[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lo = forward_bucketed(idx, means, bidx).float()
            tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                         tgt[:, qp].reshape(-1), reduction='sum'))
            n += idx.shape[0] * len(qp)
        return tot / n

    # base CE on eval
    tot = 0.0; n = 0
    for i in range(0, NEVAL, 4):
        bb = ev[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        tot += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    base = tot / n

    CE = {'base': round(base, 4), 'zero': round(ce_cond(None, None), 4)}
    for k in KS:
        ms, b = bucket_means(k)
        CE[f'k{k}'] = round(ce_cond(ms, b), 4)
        print(f"k={k}: CE {CE[f'k{k}']}", flush=True)
    ms, b = bucket_means(4, rand=True)
    CE['k4rand'] = round(ce_cond(ms, b), 4)

    gap1 = CE['k1'] - base
    rec2 = (CE['k1'] - CE['k2']) / gap1
    out = {'n_train': NTRAIN, 'n_eval': NEVAL, 'ce': CE,
           'k2_recovery_of_k1_gap': round(rec2, 4),
           'pred_a_monotone': bool(CE['k1'] > CE['k2'] > CE['k4'] > CE['k16']),
           'pred_b_position_first_bit': bool(rec2 >= 0.15 and
                                             any(CE[f'k{k}'] < CE['zero'] for k in KS)),
           'pred_c_random_is_k1': bool(abs(CE['k4rand'] - CE['k1']) <= 0.08),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | k2 recovery {rec2:.3f}")
    print(f"pred_a mono {out['pred_a_monotone']} | pred_b first-bit {out['pred_b_position_first_bit']} | pred_c rand {out['pred_c_random_is_k1']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
