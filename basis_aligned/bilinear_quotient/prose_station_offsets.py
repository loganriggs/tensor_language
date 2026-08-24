# prose_station_offsets: what do bilin18's copy stations read on ORDINARY PROSE?
# (§1211/§1216: they carry ~0.038 nats = 22% of the prose read budget — doing what?)
#
# Same offset instrument as §1215 but NATURAL FineWeb rows (no repeat). On prose there is
# no privileged offset 127/128; the diagnostic is the SHAPE of each station's far mass:
# a recency-decaying content read (like the §1182 pooling curve) vs residual match-seeking
# (mass on whatever far tokens repeat the query token — measured as the share of far mass
# on keys equal to the query token, "self-match share").
#
# Registered predictions:
#   pred_a NO OFFSET PEAK: every station's single largest far-offset share <= 0.05 on prose
#          (the 127/128 peaks are repeat-structure creations, not positional habits).
#   pred_b MATCHERS STILL MATCH: front stations 2.5/3.8 put >= 3x more of their far mass on
#          self-match keys (key token == query token) than the L3 non-station mean — the
#          matching CRITERION is content (token identity), engaged whenever a match exists.
#   pred_c FETCHERS DON'T: 8.3/8.4's self-match ratio vs their layer's non-station mean
#          < the matchers' ratio (fetchers key on the PREDECESSOR, not the token itself).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'prose_station_offsets_results.json'
NR = 24; QSTART = 160
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


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
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    layers = [2, 3, 8]
    accs = {L: torch.zeros(9, 3, dtype=torch.float64) for L in layers}   # [max_offset_mass, selfmatch, far_total]
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
            qtok = idx[:, qs]                                # (B,Q)
            ktok = idx[:, 1:]                                # (B,K)
            match = (qtok[:, :, None] == ktok[:, None, :])   # (B,Q,K)
            pm = p * far
            accs[L][:, 2] += pm.sum((0, 2, 3)).double().cpu()
            accs[L][:, 1] += (pm * match[:, None]).sum((0, 2, 3)).double().cpu()
            # per-offset far-mass histogram (offsets index 65..T-1)
            for o in range(65, T):
                sel = (off == o) & far
                if sel.any():
                    peak_off[L][:, o] += (p * sel).sum((0, 2, 3)).double().cpu()
    res = {}
    for L in layers:
        tot = accs[L][:, 2]
        mx = peak_off[L].max(dim=1).values
        for h in range(9):
            t_ = float(tot[h])
            res[f'{L}.{h}'] = {'max_offset_share': round(float(mx[h]) / t_, 4) if t_ > 0 else None,
                               'selfmatch_share': round(float(accs[L][h, 1]) / t_, 4) if t_ > 0 else None,
                               'far_mass': round(t_, 2)}
    stations = ['2.5', '3.8', '8.3', '8.4']
    pa = all(res[s]['max_offset_share'] is not None and res[s]['max_offset_share'] <= 0.05 for s in stations)
    l3_ns = [res[f'3.{h}']['selfmatch_share'] for h in range(9) if h not in (8, 1)]
    l3_mean = sum(l3_ns) / len(l3_ns)
    l8_ns = [res[f'8.{h}']['selfmatch_share'] for h in range(9) if h not in (3, 4)]
    l8_mean = sum(l8_ns) / len(l8_ns)
    r25 = res['2.5']['selfmatch_share'] / max(l3_mean, 1e-6)   # cross-layer ref: use L3 mean for both front
    r38 = res['3.8']['selfmatch_share'] / max(l3_mean, 1e-6)
    r83 = res['8.3']['selfmatch_share'] / max(l8_mean, 1e-6)
    r84 = res['8.4']['selfmatch_share'] / max(l8_mean, 1e-6)
    pb = (r25 >= 3.0 and r38 >= 3.0)
    pc = (max(r83, r84) < min(r25, r38))
    out = {'n_rows': NR, 'qstart': QSTART, 'shares': res,
           'ratios': {'2.5': round(r25, 2), '3.8': round(r38, 2), '8.3': round(r83, 2), '8.4': round(r84, 2)},
           'l3_nonstation_mean_selfmatch': round(l3_mean, 4), 'l8_nonstation_mean_selfmatch': round(l8_mean, 4),
           'pred_a_no_peak': bool(pa), 'pred_b_matchers_match': bool(pb), 'pred_c_fetchers_dont': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    named = {k: res[k] for k in stations}
    print(f"stations {json.dumps(named)} | ratios {out['ratios']}")
    print(f"pred_a nopeak {pa} | pred_b matchers {pb} | pred_c fetchers {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
