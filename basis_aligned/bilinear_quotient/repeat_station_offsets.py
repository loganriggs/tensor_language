# repeat_station_offsets: WHAT do the station heads read at distance — the classic
# induction split (predecessor t−127 vs copy-source t−128), or something else?
#
# §1207/§1213: bilin18's copy front end = heads 2.5, 3.8 (fetcher) + 3.1 (auxiliary,
# toxic without the fetcher) + 8.3/8.4 (redundant pair). This measures, on repeat rows
# (tokens[128:256]=tokens[0:128]; at query t the same token sits at t−128, its successor at
# t−127), each head's attention-pattern mass by offset: share at o=128 (copy-source), o=127
# (successor = the induction fetch target), o in 126..130 (neighborhood), vs all other
# offsets > 64 (the beyond-window remainder). Squared-bilinear patterns are unnormalized —
# use |pattern| mass shares over offsets > 64 only (the long-range part; within-window mass
# is not at issue). Heads: the five named + controls 5.7 (sink; expect mass at position 0,
# excluded by construction here since offset ≠ t), 5.5 (induction head per dossier), and
# layer means for L3/L8 non-station heads.
#
# Registered predictions:
#   pred_a FETCHERS READ THE SUCCESSOR: heads 2.5, 3.8, 8.3, 8.4 each put >= 40% of their
#          beyond-64 mass in the 126..130 band (mechanistically: induction-style reads).
#   pred_b ROLE SPLIT AT L3: H8 and H1 peak at DIFFERENT offsets within {127, 128}
#          (fetcher reads the successor o=127, auxiliary the source o=128, or vice versa —
#          registered as a difference, direction logged).
#   pred_c NON-STATION CONTROLS FLAT: mean over L3's other seven heads of the 126..130
#          share <= 0.15 (the offset structure is station-specific, not layer-wide).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_station_offsets_results.json'
NR = 24; QSTART = 160  # scored queries: deep enough that offsets up to 130 exist beyond 64
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
HEADS = [(2, 5), (3, 8), (3, 1), (8, 3), (8, 4), (5, 5), (5, 7)]


@torch.no_grad()
def patterns_for_layer(idx, L):
    """|pattern| (B,9,T,T) at layer L under the TRUE model (hooks-free manual forward)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for li, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if li == L:
            xin = F.rms_norm(xm, (D,))
            cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
            q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
            q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            return pat.masked_fill(~tril, 0.0).abs()
        x, v1 = blk(x, v1, x0)
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
    stations = ['2.5', '3.8', '8.3', '8.4']
    pa = all(res[s]['share_band'] is not None and res[s]['share_band'] >= 0.40 for s in stations)
    h8_peak = '127' if res['3.8']['share_127'] >= res['3.8']['share_128'] else '128'
    h1_peak = '127' if res['3.1']['share_127'] >= res['3.1']['share_128'] else '128'
    pb = h8_peak != h1_peak
    non_station = [res[f'3.{h}']['share_band'] for h in range(9) if h not in (8, 1)]
    pc = (sum(non_station) / len(non_station)) <= 0.15
    out = {'n_rows': NR, 'qstart': QSTART, 'shares': res,
           'h8_peak': h8_peak, 'h1_peak': h1_peak,
           'pred_a_fetchers_successor': bool(pa),
           'pred_b_role_split': bool(pb),
           'pred_c_controls_flat': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    named = {k: res[k] for k in ('2.5', '3.8', '3.1', '8.3', '8.4', '5.5', '5.7')}
    print(f"named heads {json.dumps(named)}")
    print(f"H8 peak o={h8_peak} | H1 peak o={h1_peak}")
    print(f"pred_a fetchers {out['pred_a_fetchers_successor']} | pred_b split {out['pred_b_role_split']} | pred_c flat {out['pred_c_controls_flat']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
