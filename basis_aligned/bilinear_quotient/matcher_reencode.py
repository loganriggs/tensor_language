# matcher_reencode: WHO re-encodes the match evidence? §1242: the matchers' write is fully
# consumed by blocks 2-3 (keep-through-3-then-delete costs 0.03; never-write costs 2.23).
# The candidate consumers are those blocks' MLPs. Here the write stays in the ATTENTION
# path everywhere, but is HIDDEN from selected MLPs' inputs only (mlp input = rms(x −
# matcher_contribs) at the blinded layers; everything else untouched).
#
# Conditions (repeat rows, t>=128): base; blind_mlp2; blind_mlp3; blind_both;
# blind_mlp45 (control: hiding it from the L4-5 MLPs — §1242 says by then the raw vector
# is scaffolding, so this should be CHEAP if re-encoding is done at 2-3);
# zerowrite (anchor 2.225).
#
# Registered predictions:
#   pred_a THE RE-ENCODERS ARE MLP2/3: cost(blind_both) >= 0.6 x cost(zerowrite).
#   pred_b MLP3 IS THE MAIN ONE: cost(blind_mlp3) >= 2 x cost(blind_mlp2) (3.8 — the
#          dominant matcher — writes at L3; its own block's MLP is first consumer).
#   pred_c LATE BLINDING CHEAP: cost(blind_mlp45) <= 0.3 x cost(blind_both).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_reencode_results.json'
NR = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_reencode_results.json'
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
        if L in MATCHERS:
            if zero:
                for h in MATCHERS[L]:
                    y[:, :, h, :] = 0.0
            else:
                for h in MATCHERS[L]:
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
    CONDS = {'base': (set(), False), 'blind_mlp2': ({2}, False), 'blind_mlp3': ({3}, False),
             'blind_both': ({2, 3}, False), 'blind_mlp45': ({4, 5}, False),
             'zerowrite': (set(), True)}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, (bl, z) in CONDS.items():
            lo = forward_blind(idx, bl, z).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_mlp23_reencode': bool(cost['blind_both'] >= 0.6 * cost['zerowrite']),
           'pred_b_mlp3_main': bool(cost['blind_mlp3'] >= 2 * max(cost['blind_mlp2'], 1e-6)),
           'pred_c_late_cheap': bool(cost['blind_mlp45'] <= 0.3 * max(cost['blind_both'], 1e-6)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | pred_a mlp23 {out['pred_a_mlp23_reencode']} | pred_b mlp3 {out['pred_b_mlp3_main']} | pred_c late {out['pred_c_late_cheap']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
