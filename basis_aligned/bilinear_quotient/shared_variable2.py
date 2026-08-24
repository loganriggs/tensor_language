# shared_variable2: DEPTH MAP of the broadcast's consumers. §1236: scrambling v1 as
# inherited by layers 1-17 kills copy (+3.65) and content (+0.72). WHICH layers' v1-mix
# carries which function? Scramble the inherited v1 one BAND at a time: front (L1-4),
# mid (L5-9), late (L10-17). Block-0's own write stays clean everywhere.
#
# Expected consumer geography from the circuit map: the matchers' comparison substrate
# is stream content built by FRONT layers' value-mixing; the fetch payload is delivered
# at L8 (mid); the content bag is pooled by the mid crowd.
#
# Registered predictions (NR=48):
#   pred_a CONTENT IS MID: B-mid's rare-target CE rise >= 50% of B-all's (+0.72).
#   pred_b COPY IS FRONT+MID: (B-front + B-mid repeat-CE rises) >= 0.8 x B-all's (+3.65),
#          and B-late's <= 15% of B-all's.
#   pred_c ANCHORS: B-all replicates §1236 (repeat +3.65 ±0.15, rare +0.72 ±0.1);
#          identity permutation exact.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'shared_variable2_results.json'
NR = 48; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
BANDS = {'front': set(range(1, 5)), 'mid': set(range(5, 10)), 'late': set(range(10, 18)),
         'all': set(range(1, 18))}
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
def forward_scramble(idx, perm, band):
    """band: set of layers whose INHERITED v1 is scrambled (block-0's own write clean)."""
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
            v1s = v[:, perm]
            vv = v1
        else:
            carry = v1s if L in band else v1
            vv = (1 - at.lamb) * v + at.lamb * carry.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_split(rows, perm, band, is_freq):
    """Returns (ce_all, ce_rare, ce_freq) at t>=QSTART."""
    qp = torch.arange(QSTART, T, device=DEV)
    tots = [0.0, 0.0, 0.0]; ns = [0, 0, 0]
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_scramble(idx, perm, band).float()
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
    for name in ('none', 'front', 'mid', 'late', 'all'):
        pm = ident if name == 'none' else perm
        band = set() if name == 'none' else BANDS[name]
        rep_all, _, _ = ce_split(REP, pm, band, is_freq)
        pr_all, pr_rare, pr_freq = ce_split(PROSE, pm, band, is_freq)
        res[name] = {'repeat_ce': round(rep_all, 4), 'prose_rare_ce': round(pr_rare, 4),
                     'prose_freq_ce': round(pr_freq, 4)}
        print(f"{name:>6}: repeat {res[name]['repeat_ce']} | rare {res[name]['prose_rare_ce']} | freq {res[name]['prose_freq_ce']}", flush=True)
    b = res['none']
    d = {k: {'rep': round(res[k]['repeat_ce'] - b['repeat_ce'], 4),
             'rare': round(res[k]['prose_rare_ce'] - b['prose_rare_ce'], 4)}
         for k in ('front', 'mid', 'late', 'all')}
    out = {'n_rows': NR, 'results': res, 'deltas': d,
           'pred_a_content_mid': bool(d['mid']['rare'] >= 0.5 * d['all']['rare']),
           'pred_b_copy_frontmid': bool(d['front']['rep'] + d['mid']['rep'] >= 0.8 * d['all']['rep'] and
                                        d['late']['rep'] <= 0.15 * d['all']['rep']),
           'pred_c_anchor': bool(abs(d['all']['rep'] - 3.649) <= 0.15 and
                                 abs(d['all']['rare'] - 0.7226) <= 0.1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"deltas {d}")
    print(f"pred_a mid {out['pred_a_content_mid']} | pred_b front+mid {out['pred_b_copy_frontmid']} | pred_c anchor {out['pred_c_anchor']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
