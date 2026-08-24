# crowd_scaling: the pooling crowd's redundancy curve — cost of masking RANDOM k-head sets.
#
# §1222: the prose read budget (0.176 @W64) has a thin nameable core (12 heads = 43%) and a
# ~150-head tail. This quantifies the tail's structure: for k random heads drawn from the
# 150 non-core heads, how does joint read-mask cost scale with k? Linear in k = independent
# small contributions; superlinear = synergy (the crowd's value emerges jointly, §1093-style);
# sublinear = redundancy (heads cover for each other).
#
# Conditions: base; core12 (§1222's chosen set, anchor); pool150 (all non-core); random-k
# for k in {8, 16, 32, 64} x 3 seeds each (means reported); all18.
#
# Registered predictions:
#   pred_a THE TAIL CARRIES THE MAJORITY: cost(pool150) >= 0.5 x cost(all18).
#   pred_b SUBLINEAR PER HEAD (redundancy, not synergy): cost(k=64)/64 <= 0.7 x cost(k=8)/8
#          — as more of the crowd is blinded, survivors cover; per-head marginal value FALLS.
#          (§1093's 5.7x super-additivity was for STATIC-ablation of outputs; read-masking
#          leaves heads live on local context, so coverage should dominate — registered as
#          the discriminating test between the two redundancy pictures.)
#   pred_c PARTITION: cost(core12) + cost(pool150) within [0.7, 1.3] x cost(all18).
# Controls: sanity base = true model (±0.005); core12 must replicate §1222's 0.0761 (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'crowd_scaling_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
CORE = [(2,5),(3,8),(5,5),(1,4),(7,0),(8,3),(7,3),(6,1),(6,7),(5,1),(13,0),(8,4)]


MASK_W = None
FULL_TRIL = None


def make_mask():
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    return (torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)) & vis)


@torch.no_grad()
def forward_headmasked(idx, spec):
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
        heads = spec.get(L, None)
        if heads is None:
            pat = pat.masked_fill(~FULL_TRIL, 0.0)
        else:
            msk = torch.stack([MASK_W if h in heads else FULL_TRIL for h in range(9)], 0)
            pat = pat.masked_fill(~msk.unsqueeze(0), 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def spec_of(heads):
    sp = {}
    for L, h in heads:
        sp.setdefault(L, set()).add(h)
    return sp


@torch.no_grad()
def ce_of(spec, ROWS, qp):
    tot = 0.0; n = 0
    for i in range(0, NR, 8):
        bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_headmasked(idx, spec).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n




@torch.no_grad()
def main():
    global MASK_W, FULL_TRIL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    MASK_W = make_mask()
    FULL_TRIL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    qp = torch.arange(QSTART, T, device=DEV)
    base = ce_of({}, ROWS, qp)
    tot = 0.0; n = 0
    for i in range(0, NR, 8):
        bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        tot += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    ce_true = tot / n
    allheads = [(L, h) for L in range(18) for h in range(9)]
    pool = [x for x in allheads if x not in CORE]
    cost = {}
    cost['all18'] = round(ce_of(spec_of(allheads), ROWS, qp) - base, 4)
    cost['core12'] = round(ce_of(spec_of(CORE), ROWS, qp) - base, 4)
    cost['pool150'] = round(ce_of(spec_of(pool), ROWS, qp) - base, 4)
    print(f"base {base:.4f} true {ce_true:.4f} | all18 {cost['all18']} core12 {cost['core12']} pool150 {cost['pool150']}", flush=True)
    g = torch.Generator().manual_seed(11)
    rk = {}
    for k in (8, 16, 32, 64):
        vals = []
        for s in range(3):
            perm = torch.randperm(len(pool), generator=g).tolist()
            sel = [pool[j] for j in perm[:k]]
            vals.append(ce_of(spec_of(sel), ROWS, qp) - base)
        rk[k] = round(sum(vals) / 3, 5)
        print(f"k={k}: mean cost {rk[k]}", flush=True)
    per8 = rk[8] / 8; per64 = rk[64] / 64
    out = {'n_rows': NR, 'W': WIN, 'base': round(base, 4), 'cost': cost, 'random_k_mean': rk,
           'per_head': {'k8': round(per8, 6), 'k64': round(per64, 6)},
           'sanity': bool(abs(base - ce_true) <= 0.005),
           'core_replicates': bool(abs(cost['core12'] - 0.0761) <= 0.005),
           'pred_a_tail_majority': bool(cost['pool150'] >= 0.5 * cost['all18']),
           'pred_b_sublinear': bool(per64 <= 0.7 * per8),
           'pred_c_partition': bool(0.7 * cost['all18'] <= cost['core12'] + cost['pool150'] <= 1.3 * cost['all18']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"per-head k8 {per8:.6f} vs k64 {per64:.6f}")
    print(f"sanity {out['sanity']} | core repl {out['core_replicates']} | pred_a tail {out['pred_a_tail_majority']} | pred_b sublin {out['pred_b_sublinear']} | pred_c part {out['pred_c_partition']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
