# shared_variable: do the COPY PIPELINE and the CONTENT POOL share ONE intermediate
# variable — block-0's static per-token c_v code (§1076) — consumed via two separable
# routes? Route A: block-0's own attention output writes token identity into the stream
# (what the matchers 2.5/3.8 compare at range, §1228-30). Route B: the value-residual
# (v = (1-λ)v + λ·v1 at every later layer) broadcasts the same v1 = block-0 c_v values
# into the mid-band pool that becomes the content manifold (§1076; ablation +3.3 nats
# content-tilted).
#
# Intervention: IDENTITY SCRAMBLE — v1 replaced by v1 of the same row with positions
# permuted (wrong tokens' values, right norms). Applied to: routeA (block-0's own y only),
# routeB (the v1 handed to layers 1-17 only), both (at the source). Identity permutation
# (no-op) = sanity.
#
# Measures (NR=48 rows each — doubled data per the standing more-data rule):
#   COPY: CE at t>=128 on verbatim-repeat rows (base 0.36, §1204).
#   CONTENT: CE on natural prose split by target class — RARE targets (outside top-128
#   corpus tokens; the content words, §1151 convention) vs FREQUENT targets (grammar).
#
# Registered predictions:
#   pred_a DISSOCIATION, COPY SIDE: routeA raises repeat-CE >= 3x routeB's rise.
#          (Risk registered: fetched values also carry λ·v1, so routeB may hurt copying
#          through the fetch payload — if pred_a fails THAT way, the variable is shared
#          but the routes are not separable on the copy side; report which.)
#   pred_b DISSOCIATION, CONTENT SIDE: routeB raises prose RARE-target CE >= 2x routeA's
#          rise, and routeB's rare-target rise >= 2x its own frequent-target rise
#          (content-tilt, §FINDINGS 4a).
#   pred_c CONTROLS: identity permutation = base exactly (±0.005 all metrics); both-routes
#          >= max(routeA, routeB) on every metric (no cancellation).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'shared_variable_results.json'
NR = 48; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def forward_scramble(idx, perm, route):
    """route in {'none','A','B','both'}; perm permutes positions of v1."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None; v1s = None
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
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
            v1s = v[:, perm]                                     # scrambled block-0 values
            use = v1s if route in ('A', 'both') else v1          # block 0's own output
            vv = use
        else:
            carry = v1s if route in ('B', 'both') else v1        # what later layers inherit
            vv = (1 - at.lamb) * v + at.lamb * carry.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_split(rows, perm, route, is_freq):
    """Returns (ce_all, ce_rare, ce_freq) at t>=QSTART."""
    qp = torch.arange(QSTART, T, device=DEV)
    tots = [0.0, 0.0, 0.0]; ns = [0, 0, 0]
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_scramble(idx, perm, route).float()
        lse = F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                              tgt[:, qp].reshape(-1), reduction='none')
        tq = tgt[:, qp].reshape(-1)
        fr = is_freq[tq]
        tots[0] += float(lse.sum()); ns[0] += len(lse)
        tots[1] += float(lse[~fr].sum()); ns[1] += int((~fr).sum())
        tots[2] += float(lse[fr].sum()); ns[2] += int(fr.sum())
    return tuple(t / max(n, 1) for t, n in zip(tots, ns))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    PROSE = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    V = int(m.lm_head.weight.shape[0])
    cnts = torch.bincount(PROSE[:, :T].reshape(-1), minlength=V)
    is_freq = torch.zeros(V, dtype=torch.bool, device=DEV)
    is_freq[torch.topk(cnts, 128).indices.to(DEV)] = True

    g = torch.Generator().manual_seed(9)
    perm = torch.randperm(T, generator=g).to(DEV)
    ident = torch.arange(T, device=DEV)

    res = {}
    for route in ('none', 'A', 'B', 'both'):
        pm = ident if route == 'none' else perm
        rep_all, _, _ = ce_split(REP, pm, route, is_freq)
        pr_all, pr_rare, pr_freq = ce_split(PROSE, pm, route, is_freq)
        res[route] = {'repeat_ce': round(rep_all, 4), 'prose_ce': round(pr_all, 4),
                      'prose_rare_ce': round(pr_rare, 4), 'prose_freq_ce': round(pr_freq, 4)}
        print(f"{route:>5}: repeat {res[route]['repeat_ce']} | prose {res[route]['prose_ce']} "
              f"(rare {res[route]['prose_rare_ce']} freq {res[route]['prose_freq_ce']})", flush=True)
    # identity-perm sanity vs true forward
    x_rows = REP[:4].to(DEV); idx = x_rows[:, :-1].contiguous()
    xa = forward_scramble(idx, ident, 'both').float()
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h:
        x, v1 = blk(x, v1, x0)
    xb = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)).float()
    sanity_exact = bool(float((xa - xb).abs().max()) < 0.01)

    b = res['none']
    dA_rep = res['A']['repeat_ce'] - b['repeat_ce']
    dB_rep = res['B']['repeat_ce'] - b['repeat_ce']
    dA_rare = res['A']['prose_rare_ce'] - b['prose_rare_ce']
    dB_rare = res['B']['prose_rare_ce'] - b['prose_rare_ce']
    dB_freq = res['B']['prose_freq_ce'] - b['prose_freq_ce']
    out = {'n_rows': NR, 'results': res, 'sanity_identity_exact': sanity_exact,
           'deltas': {'A_repeat': round(dA_rep, 4), 'B_repeat': round(dB_rep, 4),
                      'A_rare': round(dA_rare, 4), 'B_rare': round(dB_rare, 4),
                      'B_freq': round(dB_freq, 4)},
           'pred_a_copy_routeA': bool(dA_rep >= 3 * max(dB_rep, 1e-6)),
           'pred_b_content_routeB': bool(dB_rare >= 2 * max(dA_rare, 1e-6) and
                                         dB_rare >= 2 * max(dB_freq, 1e-6)),
           'pred_c_controls': bool(sanity_exact and
                                   res['both']['repeat_ce'] >= max(res['A']['repeat_ce'], res['B']['repeat_ce']) - 0.02 and
                                   res['both']['prose_rare_ce'] >= max(res['A']['prose_rare_ce'], res['B']['prose_rare_ce']) - 0.02),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"deltas {out['deltas']}")
    print(f"pred_a copyA {out['pred_a_copy_routeA']} | pred_b contentB {out['pred_b_content_routeB']} | pred_c controls {out['pred_c_controls']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
