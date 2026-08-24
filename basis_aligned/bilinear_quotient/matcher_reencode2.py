# matcher_reencode2: disambiguate §1243's blind_mlp45 confound. Hiding the MATCHER write
# from mlp4/5's inputs cost 2.34 — but that condition creates the §1242 inconsistency state
# (raw in stream, invisible to the massive-dims MLPs). Control: the SAME operation with the
# NULL heads' writes (2.1/3.3 — same layers, similar norms, no copy role).
#
# Registered predictions:
#   pred_a INCONSISTENCY-SENSITIVITY IS GENERIC-INPUT-SPECIFIC... decided by the number:
#          null blind_mlp45 <= 0.3 x matcher blind_mlp45 (2.34) -> mlp4/5 genuinely READ the
#          matcher vector (the §1243 pred_c failure was real signal);
#          null >= 0.5 x matcher -> generic inconsistency artifact; §1242's scaffolding
#          reading stands. Registered as the FIRST branch (pred_a = null <= 0.3x), with the
#          second explicitly the falsification outcome.
#   pred_b ANCHOR: matcher blind_mlp45 replicates §1243 (2.3425 ± 0.1).
#   pred_c SANITY: base = true model ±0.005.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_reencode2_results.json'
NR = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
NULLH = {2: [1], 3: [3]}
HSET = MATCHERS
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_reencode2_results.json'
NR = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_reencode2_results.json'
NR = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
NULLH = {2: [1], 3: [3]}


@torch.no_grad()
def forward_blind(idx, blind, zero):
    """blind: set of layer indices whose MLP input hides matcher contribs. zero: never write."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    hidden = torch.zeros_like(x)
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
        if L in HSET:
            if zero:
                for h in HSET[L]:
                    y[:, :, h, :] = 0.0
            else:
                for h in HSET[L]:
                    yh = torch.zeros_like(y)
                    yh[:, :, h, :] = y[:, :, h, :]
                    hidden = hidden + at.c_proj(yh.reshape(B, T, D))
        x = xm + at.c_proj(y.reshape(B, T, D))
        mlp_in = x - hidden if L in blind else x
        x = x + blk.mlp(F.rms_norm(mlp_in, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    qp = torch.arange(QSTART, T, device=DEV)
    global HSET
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {}; ce_true = 0.0; n = 0
    CONDS = [('base', MATCHERS, set(), False), ('m45', MATCHERS, {4, 5}, False),
             ('null45', NULLH, {4, 5}, False)]
    ce = {c[0]: 0.0 for c in CONDS}
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, hs, bl, z in CONDS:
            HSET = hs
            lo = forward_blind(idx, bl, z).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CE if c not in ('base', 'true_model')}
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_mlp45_reads_matcher': bool(cost['null45'] <= 0.3 * cost['m45']),
           'null_ratio': round(cost['null45'] / max(cost['m45'], 1e-6), 3),
           'pred_b_anchor': bool(abs(cost['m45'] - 2.3425) <= 0.1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost} | null ratio {out['null_ratio']}")
    print(f"sanity {out['sanity']} | pred_a reads {out['pred_a_mlp45_reads_matcher']} | pred_b anchor {out['pred_b_anchor']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
