# repeat_ablate_stations: does ZERO-ABLATION agree with READ-MASKING on the stations?
#
# §649/§952-53 found copying non-localizable under zero-ablation (prose); §1207 found the
# quad carrying 69% under read-masks (repeat rows). Two differences: instrument AND regime.
# This isolates the instrument: zero-ablate (output to zero) each station and the quad on
# THE SAME repeat rows, compare to the §1207 read-mask costs head by head.
#
# If ablation ≈ mask: the §649 discrepancy was regime-only (prose vs repeat), and either
# instrument names the stations. If ablation ≪ mask: the network dynamically COMPENSATES
# for a missing head but not a blinded one (ablation triggers backup routing; the read-mask
# is the sharper causal knife) — an instrument law worth recording.
#
# Registered predictions:
#   pred_a NO COMPENSATION ON REPEAT: abl(3.8) >= 0.8 x mask(3.8) (=0.86 nats) — on repeat
#          text the fetch/match chain has no backup (§1204 seriality).
#   pred_b QUAD AGREES ACROSS INSTRUMENTS: abl(quad) within [0.7, 1.5] x mask(quad) 2.2029.
#   pred_c SINK CONTROL: abl(5.7) >= 0.5 on repeat rows too (the constant-baseline head is
#          regime-independent; its read-mask cost was ~0 because pos-0 stayed visible —
#          ablation and mask MUST disagree here, the built-in instrument-difference control).
# Control: sanity base = true model ±0.005.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_ablate_stations_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL_TRIL = None
ALLH = set(range(9))


def make_mask():
    ar = torch.arange(T, device=DEV)
    near = (ar[:, None] - ar[None, :]) < WIN
    vis = near | (ar[None, :] == 0)
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
        q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
        k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
        q2 = F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,))
        k2 = F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,))
        q = are(q, cos, sin); k = are(k, cos, sin); q2 = are(q2, cos, sin); k2 = are(k2, cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        pat = pat.masked_fill(~FULL_TRIL, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        heads = spec.get(L, None)
        if heads is not None:
            for h in heads:
                y[:, :, h, :] = 0.0
        y = y.reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    global MASK_W, FULL_TRIL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    MASK_W = make_mask()
    FULL_TRIL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    CONDS = {'base': {},
             'a25': {2: {5}}, 'a38': {3: {8}}, 'a83': {8: {3}}, 'a84': {8: {4}},
             'quad': {2: {5}, 3: {8}, 8: {3, 4}}, 'sink57': {5: {7}}}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, spec in CONDS.items():
            lo = forward_headmasked(idx, spec).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    MASK_REF = {'a38': 1.0755, 'quad': 2.2029}
    out = {'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost, 'mask_refs': MASK_REF,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_no_compensation': bool(cost['a38'] >= 0.8 * MASK_REF['a38']),
           'pred_b_quad_agrees': bool(0.7 * MASK_REF['quad'] <= cost['quad'] <= 1.5 * MASK_REF['quad']),
           'pred_c_sink_control': bool(cost['sink57'] >= 0.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | pred_a nocomp {out['pred_a_no_compensation']} | pred_b quad {out['pred_b_quad_agrees']} | pred_c sink {out['pred_c_sink_control']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
