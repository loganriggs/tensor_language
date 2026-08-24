# matcher_consumer: WHERE is the matchers' -(matched value) write consumed? (§1240 follow-on.)
# Keep 2.5/3.8's stream writes intact up to layer L_cut, then SUBTRACT them from the
# residual before block L_cut runs (their influence on blocks < L_cut is left in place —
# a keep-until-then-remove design). If the write is consumed by the fetch band (aiming
# L8's queries / supplying match evidence to mid), removal at L>=9 should be nearly free;
# if the readout consumes it directly, cost persists to L15+.
#
# Conditions (repeat rows, scored t>=128): base; cut3 (remove before block 3 — 3.8's own
# write hasn't happened yet then, so cut4 is the true "immediate" for both; cut3 removes
# only 2.5's); cut4; cut6; cut9; cut12; cut15.
#
# Registered predictions:
#   pred_a CONSUMED BY THE FETCH BAND: cost(cut9) <= 0.2 x cost(cut4).
#   pred_b IMMEDIATE REMOVAL IS THE ANCHOR: cost(cut4) within [0.7, 1.6] x the §1207-scale
#          joint matcher damage (mask 2.5+3.8 ~ 1.7; instruments differ — write-removal vs
#          read-mask — so a loose band, logged not fitted).
#   pred_c MONOTONE: costs non-increasing in L_cut (later removal never hurts more).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_consumer_results.json'
NR = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}


@torch.no_grad()
def forward_cut(idx, lcut):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    contribs = []
    for L, blk in enumerate(m.transformer.h):
        if L == lcut:
            for c in contribs:
                x = x - c
            contribs = []
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
        if L in MATCHERS and lcut is not None and L < lcut:
            for h in MATCHERS[L]:
                yh = torch.zeros_like(y)
                yh[:, :, h, :] = y[:, :, h, :]
                contribs.append(at.c_proj(yh.reshape(B, T, D)))
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    qp = torch.arange(QSTART, T, device=DEV)
    CUTS = {'base': None, 'cut3': 3, 'cut4': 4, 'cut6': 6, 'cut9': 9, 'cut12': 12, 'cut15': 15}
    ce = {c: 0.0 for c in CUTS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, lc in CUTS.items():
            lo = forward_cut(idx, lc).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CUTS if c != 'base'}
    seq = [cost['cut4'], cost['cut6'], cost['cut9'], cost['cut12'], cost['cut15']]
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_fetchband_consumes': bool(cost['cut9'] <= 0.2 * max(cost['cut4'], 1e-6)),
           'pred_b_anchor_band': bool(0.7 * 1.7 <= cost['cut4'] <= 1.6 * 1.7),
           'pred_c_monotone': bool(all(seq[i] >= seq[i + 1] - 0.02 for i in range(len(seq) - 1))),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | pred_a fetch {out['pred_a_fetchband_consumes']} | pred_b anchor {out['pred_b_anchor_band']} | pred_c mono {out['pred_c_monotone']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
