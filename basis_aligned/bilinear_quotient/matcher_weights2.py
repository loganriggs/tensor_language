# matcher_weights2: the SIGN STRUCTURE of the anti-matching criterion (§1238 follow-up).
# On raw token codes the matchers 2.5/3.8 give same-token pairs the LOWEST s1*s2 scores
# (AUC ~ 0, perfect inverted separation). Two questions:
#  (1) FACTOR STRUCTURE (weights side): at same-token pairs, do the two bilinear factors
#      have opposite signs (s1>0, s2<0 or vice versa), or is one factor negative-definite?
#  (2) IN VIVO (live model on repeat rows): §1215 measured |pattern| — is the actual signed
#      pattern value at offset 128 NEGATIVE for the matchers (delivering -1x the matched
#      token's value), and positive at offset 127 for the fetchers?
#
# Registered predictions:
#   pred_a IN-VIVO ANTI-MATCH: mean signed pattern at o=128 (queries t>=160, repeat rows)
#          is NEGATIVE for both 2.5 and 3.8, and |mean| >= 0.5 x mean |pattern| there
#          (a consistent sign, not cancellation).
#   pred_b FETCHERS POSITIVE: mean signed pattern at o=127 is POSITIVE for 8.3 and 8.4.
#   pred_c FACTOR CONSISTENCY: for each matcher, same-token weight-side scores have one
#          factor with consistent sign across >= 80% of tokens (the anti-ness is structured,
#          not noise) — factor sign shares reported.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_weights2_results.json'
NTOK = 512; QPOS = 200; KPOS = 72; NR = 24; QOFF = 160
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def factor_stats(L, h, toks):
    at = m.transformer.h[L].attn
    x = F.rms_norm(m.transformer.wte(toks), (D,))
    N = x.shape[0]
    dummy = torch.zeros(1, QPOS + 1, 9, 128, device=DEV)
    cos_t, sin_t = at.rotary(dummy)

    def pipe(lin, pos):
        z = F.rms_norm(lin(x).view(N, 9, 128), (128,)).view(1, N, 9, 128)
        return are(z, cos_t[:, pos:pos + 1], sin_t[:, pos:pos + 1])

    q = pipe(at.c_q, QPOS)[0, :, h]; k = pipe(at.c_k, KPOS)[0, :, h]
    q2 = pipe(at.c_q2, QPOS)[0, :, h]; k2 = pipe(at.c_k2, KPOS)[0, :, h]
    s1 = (q.float() * k.float()).sum(-1) / 128.0          # same-token diagonal only
    s2 = (q2.float() * k2.float()).sum(-1) / 128.0
    return {'s1_pos_share': round(float((s1 > 0).float().mean()), 4),
            's2_pos_share': round(float((s2 > 0).float().mean()), 4),
            's1_mean': round(float(s1.mean()), 5), 's2_mean': round(float(s2.mean()), 5),
            'prod_mean': round(float((s1 * s2).mean()), 6),
            'prod_neg_share': round(float(((s1 * s2) < 0).float().mean()), 4)}


@torch.no_grad()
def invivo_signed(idx, L, h, off):
    """Mean signed pattern and mean |pattern| at offset off, queries >= QOFF."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for li, blk in enumerate(m.transformer.h):
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
        if li == L:
            qs = torch.arange(QOFF, T, device=DEV)
            keys = qs - off
            p = pat[:, h, qs, :].gather(2, keys.view(1, -1, 1).expand(B, -1, 1)).squeeze(-1)
            return float(p.mean()), float(p.abs().mean())
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
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
    rows = cl.fineweb_rows(8)[:, :256].reshape(-1)
    uniq = torch.unique(rows)
    g = torch.Generator().manual_seed(4)
    sel = uniq[torch.randperm(len(uniq), generator=g)[:NTOK]].to(DEV)

    fac = {hh: factor_stats(int(hh.split('.')[0]), int(hh.split('.')[1]), sel)
           for hh in ('2.5', '3.8')}
    print(f"factors {json.dumps(fac)}", flush=True)

    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    vivo = {}
    for hh, off in (('2.5', 128), ('3.8', 128), ('8.3', 127), ('8.4', 127)):
        L, h = int(hh.split('.')[0]), int(hh.split('.')[1])
        ms, mabs = 0.0, 0.0; nb = 0
        for i in range(0, NR, 4):
            idx = REP[i:i + 4, :-1].to(DEV).contiguous()
            a, b = invivo_signed(idx, L, h, off)
            ms += a; mabs += b; nb += 1
        vivo[hh] = {'mean_signed': round(ms / nb, 5), 'mean_abs': round(mabs / nb, 5)}
        print(f"{hh} @o={off}: signed {vivo[hh]['mean_signed']} | abs {vivo[hh]['mean_abs']}", flush=True)

    pa = all(vivo[hh]['mean_signed'] < 0 and
             abs(vivo[hh]['mean_signed']) >= 0.5 * vivo[hh]['mean_abs'] for hh in ('2.5', '3.8'))
    pb = all(vivo[hh]['mean_signed'] > 0 for hh in ('8.3', '8.4'))
    pc = all(max(f['s1_pos_share'], 1 - f['s1_pos_share'],
                 f['s2_pos_share'], 1 - f['s2_pos_share']) >= 0.8 for f in fac.values())
    out = {'n_tokens': NTOK, 'factors_same_token': fac, 'invivo': vivo,
           'pred_a_invivo_anti': bool(pa), 'pred_b_fetchers_pos': bool(pb),
           'pred_c_factor_consistent': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a anti {pa} | pred_b fetchpos {pb} | pred_c factor {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
