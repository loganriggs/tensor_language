# builders_split: WHICH builder feeds which station? attn0 writes prev-token/copy-source
# (§841/§877); attn1 is the lexical/identity infrastructure (§239). §1228 showed zeroing
# both at source positions collapses the stations' reads. Split them: zero attn0 alone vs
# attn1 alone (source half), measure each station's SIGNATURE offset share (matchers 2.5/3.8
# at o=128; fetchers 8.3/8.4 at o=127) and CE.
#
# Textbook prediction: matching compares token IDENTITY (attn1's write) at query vs key;
# fetching keys on "my predecessor is X" (attn0's prev-token write) at the successor
# position. If bilin18 follows it:
#
# Registered predictions:
#   pred_a MATCHERS TRACK attn1: mean matcher o=128 drop under src1 >= 2x drop under src0.
#   pred_b FETCHERS TRACK attn0: mean fetcher o=127 drop under src0 >= 2x drop under src1.
#   pred_c ANCHOR: cost(src01) replicates §1228's 5.6011 (±0.15); both singles cost less
#          than the pair (sanity of the split).
# Control: sanity base = true model (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'builders_split_results.json'
NR = 24; QSTART = 128; QOFF = 160
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def forward_zeroed(idx, zero_layers, zero_slice, cap_layer=None):
    """Full model; attn OUTPUT zeroed on zero_slice positions at zero_layers.
    If cap_layer is not None, return |pattern| (B,9,T,T) at that layer instead of logits."""
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
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        if L == cap_layer:
            return pat.abs()
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        yo = at.c_proj(y)
        if L in zero_layers:
            yo = yo.clone()
            yo[:, zero_slice] = 0.0
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def offset_share(idx, zero_layers, zero_slice, L, h, o):
    pat = forward_zeroed(idx, zero_layers, zero_slice, cap_layer=L)
    qs = torch.arange(QOFF, T, device=DEV)
    p = pat[:, h, qs, 1:]
    keys = torch.arange(1, T, device=DEV)
    off = qs[:, None] - keys[None, :]
    far = off > 64
    mo = float((p * ((off == o) & far)).sum())
    mfar = float((p * far).sum())
    return mo / max(mfar, 1e-9)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    src = slice(0, 128)
    CONDS = {'base': (set(), src), 'src0': ({0}, src), 'src1': ({1}, src),
             'src01': ({0, 1}, src)}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; n = 0
    HOFF = {'2.5': 128, '3.8': 128, '8.3': 127, '8.4': 127}
    sh = {c: {k: 0.0 for k in HOFF} for c in ('base', 'src0', 'src1')}
    nb = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, (zl, zs) in CONDS.items():
            lo = forward_zeroed(idx, zl, zs).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        for cname in ('base', 'src0', 'src1'):
            zl, zs = CONDS[cname]
            for hh, o in HOFF.items():
                L, h = int(hh.split('.')[0]), int(hh.split('.')[1])
                sh[cname][hh] += offset_share(idx, zl, zs, L, h, o)
        nb += 1
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    SH = {c: {k: round(v / nb, 4) for k, v in d.items()} for c, d in sh.items()}
    def drop(c, hh):
        return 1 - SH[c][hh] / max(SH['base'][hh], 1e-9)
    m_d0 = (drop('src0', '2.5') + drop('src0', '3.8')) / 2
    m_d1 = (drop('src1', '2.5') + drop('src1', '3.8')) / 2
    f_d0 = (drop('src0', '8.3') + drop('src0', '8.4')) / 2
    f_d1 = (drop('src1', '8.3') + drop('src1', '8.4')) / 2
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost, 'signature_share': SH,
           'drops': {'matchers_src0': round(m_d0, 3), 'matchers_src1': round(m_d1, 3),
                     'fetchers_src0': round(f_d0, 3), 'fetchers_src1': round(f_d1, 3)},
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_matchers_track_1': bool(m_d1 >= 2 * m_d0),
           'pred_b_fetchers_track_0': bool(f_d0 >= 2 * f_d1),
           'pred_c_anchor': bool(abs(cost['src01'] - 5.6011) <= 0.15 and
                                 cost['src0'] < cost['src01'] and cost['src1'] < cost['src01']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"signature shares {SH} | drops {out['drops']}")
    print(f"sanity {out['sanity']} | pred_a m->1 {out['pred_a_matchers_track_1']} | pred_b f->0 {out['pred_b_fetchers_track_0']} | pred_c anchor {out['pred_c_anchor']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
