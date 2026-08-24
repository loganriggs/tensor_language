# prose_offsets_family: the criterion story, family side. bilin18's matchers self-match on
# prose (6x layer baseline, §1219). swiglu18 has NO matchers (§1217) — so NO head should
# show a token-identity criterion on prose, and its stations' prose double duty (§1216,
# 0.036 nats) must be ordinary recency/content pooling instead.
#
# Instrument: §1219's self-match share (far mass on keys whose token equals the query
# token) + recency shape (far-mass share at offsets 65-96 vs 129-160), softmax patterns,
# natural rows, queries t>=160. Heads: stations 4.4, 5.2, 8.0, 8.8 + full L4/L5/L8 maps.
#
# Registered predictions:
#   pred_a NO MATCHER ANYWHERE: every station's self-match ratio vs its layer's
#          non-station mean < 3.0 (bilin18's matchers: 5.9-6.0).
#   pred_b RECENCY POOLING: each station puts >= 2x more far mass at offsets 65-96 than at
#          129-160 (decaying pool, §1182 shape).
#   pred_c NO POSITIONAL PEAK on prose: every station's max single-offset share <= 0.05.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()

import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'prose_offsets_family_results.json'
NR = 24; QSTART = 160  # scored queries: deep enough that offsets up to 130 exist beyond 64
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb
HEADS = [(4, 4), (5, 2), (8, 0), (8, 8)]


@torch.no_grad()
def patterns_for_layer(idx, L):
    """softmax pattern (B,9,T,T) at layer L under the TRUE swiglu18 model."""
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for li, blk in enumerate(mdl.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if li == L:
            q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
            k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
            cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
            q = are(q, cos, sin); k = are(k, cos, sin)
            scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            scores = scores.masked_fill(~tril, float('-inf'))
            return F.softmax(scores, dim=-1)
        # advance the true block
        q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
        k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(q, cos, sin); k = are(k, cos, sin)
        scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = F.softmax(scores.masked_fill(~tril, float('-inf')), dim=-1)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    layers = [4, 5, 8]
    accs = {L: torch.zeros(9, 4, dtype=torch.float64) for L in layers}  # [selfmatch, near65_96, far129_160, far_total]
    peak_off = {L: torch.zeros(9, T, dtype=torch.float64) for L in layers}
    qs = torch.arange(QSTART, T, device=DEV)
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        for L in layers:
            pat = patterns_for_layer(idx, L)
            p = pat[:, :, qs, 1:]
            keys = torch.arange(1, T, device=DEV)
            off = qs[:, None] - keys[None, :]
            far = off > 64
            qtok = idx[:, qs]; ktok = idx[:, 1:]
            match = (qtok[:, :, None] == ktok[:, None, :])
            pm = p * far
            accs[L][:, 0] += (pm * match[:, None]).sum((0, 2, 3)).double().cpu()
            accs[L][:, 1] += (p * ((off >= 65) & (off <= 96))).sum((0, 2, 3)).double().cpu()
            accs[L][:, 2] += (p * ((off >= 129) & (off <= 160))).sum((0, 2, 3)).double().cpu()
            accs[L][:, 3] += pm.sum((0, 2, 3)).double().cpu()
            for o in range(65, T):
                sel = (off == o)
                peak_off[L][:, o] += (p * sel).sum((0, 2, 3)).double().cpu()
    res = {}
    for L in layers:
        a = accs[L]; mx = peak_off[L].max(dim=1).values
        for h in range(9):
            tot = float(a[h, 3])
            res[f'{L}.{h}'] = {'selfmatch_share': round(float(a[h, 0]) / tot, 4) if tot > 0 else None,
                               'near_share': round(float(a[h, 1]) / tot, 4) if tot > 0 else None,
                               'deep_share': round(float(a[h, 2]) / tot, 4) if tot > 0 else None,
                               'max_offset_share': round(float(mx[h]) / tot, 4) if tot > 0 else None,
                               'far_mass': round(tot, 2)}
    ST = {'4.4': 4, '5.2': 5, '8.0': 8, '8.8': 8}
    ratios = {}
    for s, L in ST.items():
        sh = int(s.split('.')[1])
        ns = [res[f'{L}.{h}']['selfmatch_share'] for h in range(9)
              if not (L == int(s.split('.')[0]) and h == sh) and not (L == 8 and h in (0, 8))]
        base = sum(ns) / len(ns)
        ratios[s] = round(res[s]['selfmatch_share'] / max(base, 1e-6), 2)
    pa = all(v < 3.0 for v in ratios.values())
    pb = all(res[s]['near_share'] >= 2 * res[s]['deep_share'] for s in ST)
    pc = all(res[s]['max_offset_share'] <= 0.05 for s in ST)
    out = {'model': 'swiglu18', 'n_rows': NR, 'qstart': QSTART, 'shares': res, 'ratios': ratios,
           'pred_a_no_matcher': bool(pa), 'pred_b_recency': bool(pb), 'pred_c_no_peak': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stations {json.dumps({k: res[k] for k in ST})}")
    print(f"ratios {ratios}")
    print(f"pred_a nomatcher {pa} | pred_b recency {pb} | pred_c nopeak {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
