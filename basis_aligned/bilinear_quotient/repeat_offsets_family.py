# repeat_offsets_family: swiglu18's station mechanism — are its copy heads source-matchers
# (o=128) or successor-fetchers (o=127)? bilin18 splits the roles ACROSS DEPTH (§1215):
# front stations 2.5/3.8 read the source, mid stations 8.3/8.4 (+5.5) fetch the successor.
# swiglu18's stations are ALL mid-stack (L4H4, L5H2, L8H0/H8). If depth-role coupling is the
# law, they should be successor-fetchers; if the front role migrated with them, one should
# be a source-matcher.
#
# Softmax patterns are normalized — offset shares computed over pattern mass at offsets > 64
# (key 0 excluded), scored queries t>=160. Heads reported: 4.4, 5.2, 8.0, 8.8 + layer means.
#
# Registered predictions:
#   pred_a MECHANISM PRESENT: each of the four stations puts >= 0.15 of its beyond-64 mass
#          in the 126..130 band (weaker bar than bilin18's measured 0.27-0.37 — softmax
#          normalization spreads mass differently; direction is what matters).
#   pred_b AT LEAST ONE SOURCE-MATCHER: some station peaks at o=128 (the front ROLE exists
#          even without front placement) — else the depth-role law wins and roles here are
#          all successor; logged either way.
#   pred_c CONTROLS FLAT: mean 126-130 share over L4's non-station heads <= 0.10.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()

import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_offsets_family_results.json'
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
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    layers = sorted({L for L, _ in HEADS})
    # accumulate per (layer, head): mass at o=127, o=128, band 126-130, total beyond-64
    acc = {L: torch.zeros(9, 4, dtype=torch.float64) for L in layers}
    qs = torch.arange(QSTART, T, device=DEV)
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        for L in layers:
            pat = patterns_for_layer(idx, L)          # (B,9,T,T)
            p = pat[:, :, qs, 1:]                     # exclude key 0 (sink constant)
            keys = torch.arange(1, T, device=DEV)
            off = qs[:, None] - keys[None, :]         # (Q,K) offsets
            far = off > 64
            b127 = off == 127; b128 = off == 128
            band = (off >= 126) & (off <= 130)
            m127 = (p * (b127 & far)).sum((0, 2, 3)); m128 = (p * (b128 & far)).sum((0, 2, 3))
            mband = (p * (band & far)).sum((0, 2, 3)); mfar = (p * far).sum((0, 2, 3))
            acc[L] += torch.stack([m127, m128, mband, mfar], 1).double().cpu()
    res = {}
    for L in layers:
        a = acc[L]
        for h in range(9):
            tot = float(a[h, 3])
            res[f'{L}.{h}'] = {'share_127': round(float(a[h, 0]) / tot, 4) if tot > 0 else None,
                               'share_128': round(float(a[h, 1]) / tot, 4) if tot > 0 else None,
                               'share_band': round(float(a[h, 2]) / tot, 4) if tot > 0 else None}
    stations = ['4.4', '5.2', '8.0', '8.8']
    pa = all(res[s]['share_band'] is not None and res[s]['share_band'] >= 0.15 for s in stations)
    peaks = {s: ('128' if res[s]['share_128'] >= res[s]['share_127'] else '127') for s in stations}
    pb = any(v == '128' for v in peaks.values())
    non_station = [res[f'4.{h}']['share_band'] for h in range(9) if h != 4]
    pc = (sum(non_station) / len(non_station)) <= 0.10
    out = {'model': 'swiglu18', 'n_rows': NR, 'qstart': QSTART, 'shares': res, 'peaks': peaks,
           'pred_a_mechanism': bool(pa), 'pred_b_source_matcher': bool(pb),
           'pred_c_controls_flat': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    named = {k: res[k] for k in stations}
    print(f"stations {json.dumps(named)} | peaks {peaks}")
    print(f"pred_a mech {out['pred_a_mechanism']} | pred_b source {out['pred_b_source_matcher']} | pred_c flat {out['pred_c_controls_flat']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
