# station_source_builders: RECONCILIATION of §239's name-circuit source-builders (attn0/1
# "build the token identity representations at the name's earlier mention that downstream
# copying reads as its source") with the §1204-18 copy stations. The pipeline hypothesis:
# attn0/1 WRITE identity at the source position -> source keys carry it -> matchers 2.5/3.8
# match on it (o=128 reads, §1215) -> fetchers 8.3/8.4 collect successors.
#
# Design (repeat rows, tokens[128:256]=tokens[0:128], scored t>=128): zero attn0+attn1
# OUTPUTS at the SOURCE HALF only (positions < 128); the destination half keeps them. If the
# source-builder story is right, the matchers' offset-128 pattern mass collapses and CE
# spikes, even though nothing at the scored positions was touched directly.
#
# Conditions: base; src01 (attn0+1 zeroed at pos<128); dst01 (attn0+1 zeroed at pos>=128,
# site-specificity control); src10 (attn10 zeroed at pos<128, layer-placebo control).
# Measured: CE at t>=128 (all conditions) + matcher (2.5, 3.8) and fetcher (8.3, 8.4)
# offset-shares under base and src01 (§1215 instrument: |pattern| mass beyond offset 64,
# key 0 excluded, queries t>=160).
#
# Registered predictions:
#   pred_a MATCH COLLAPSES AT THE READ: matcher o=128 share drops >= 50% under src01
#          (2.5: 0.302, 3.8: 0.325 at base — §1215).
#   pred_b SOURCE >> DESTINATION: CE(src01) - base >= 3 x (CE(dst01) - base) — the §239
#          antecedent-side pattern, now on synthetic repeat.
#   pred_c PLACEBO SMALL: CE(src10) - base <= 0.3 x (CE(src01) - base).
# Control: sanity base = true model (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'station_source_builders_results.json'
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
def offset128_share(idx, zero_layers, zero_slice, L, h):
    pat = forward_zeroed(idx, zero_layers, zero_slice, cap_layer=L)
    qs = torch.arange(QOFF, T, device=DEV)
    p = pat[:, h, qs, 1:]
    keys = torch.arange(1, T, device=DEV)
    off = qs[:, None] - keys[None, :]
    far = off > 64
    m128 = float((p * ((off == 128) & far)).sum())
    mfar = float((p * far).sum())
    return m128 / max(mfar, 1e-9)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    src = slice(0, 128); dst = slice(128, T)
    CONDS = {'base': (set(), src), 'src01': ({0, 1}, src), 'dst01': ({0, 1}, dst),
             'src10': ({10}, src)}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; n = 0
    sh = {c: {'2.5': 0.0, '3.8': 0.0, '8.3': 0.0, '8.4': 0.0} for c in ('base', 'src01')}
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
        for cname in ('base', 'src01'):
            zl, zs = CONDS[cname]
            for hh in ('2.5', '3.8', '8.3', '8.4'):
                L, h = int(hh.split('.')[0]), int(hh.split('.')[1])
                sh[cname][hh] += offset128_share(idx, zl, zs, L, h)
        nb += 1
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    SH = {c: {k: round(v / nb, 4) for k, v in d.items()} for c, d in sh.items()}
    drop25 = 1 - SH['src01']['2.5'] / max(SH['base']['2.5'], 1e-9)
    drop38 = 1 - SH['src01']['3.8'] / max(SH['base']['3.8'], 1e-9)
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost, 'o128_share': SH,
           'matcher_drops': {'2.5': round(drop25, 3), '3.8': round(drop38, 3)},
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_match_collapses': bool(drop25 >= 0.5 and drop38 >= 0.5),
           'pred_b_source_side': bool(cost['src01'] >= 3 * cost['dst01']),
           'pred_c_placebo_small': bool(cost['src10'] <= 0.3 * cost['src01']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"o128 shares {SH} | drops {out['matcher_drops']}")
    print(f"sanity {out['sanity']} | pred_a collapse {out['pred_a_match_collapses']} | pred_b source {out['pred_b_source_side']} | pred_c placebo {out['pred_c_placebo_small']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
