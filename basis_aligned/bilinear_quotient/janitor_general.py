# janitor_general: is block 5's cleanup a SUBSPACE-KEYED service or provenance-bound?
# §1245: the matchers' own writes are canceled 86% across block 5 (4x passive decay).
# Test: INJECT vectors the model did not generate — at block-4 entry, add alpha * d to the
# stream, where d is (i) a real matcher-write direction harvested from a DIFFERENT batch
# (matcher-shaped, wrong provenance), (ii) a null-head write direction (2.1/3.3 harvested
# the same way), (iii) a random direction of matched norm. Track the injected component's
# projection at block entries 4..8. If (i) is canceled fast while (ii)/(iii) merely decay,
# the janitor keys on the SUBSPACE, and cleanup is a standing service.
#
# alpha sized to the real writes' median norm (matched perturbation scale).
#
# Registered predictions:
#   pred_a SUBSPACE-KEYED: injected matcher-direction projection at L6-entry <= 0.5 x its
#          L4-entry value, AND <= 0.6 x the random-direction ratio (actively cleaned).
#   pred_b CONTROLS PASSIVE: null-direction and random-direction L4->L6 ratios within
#          [0.4, 1.1] of each other (both merely decay).
#   pred_c LOCUS: the matcher-direction's largest single-block drop is across block 5
#          (L5-entry -> L6-entry), matching §1245.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'janitor_general_results.json'
NR = 24; QSTART = 160
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
NULLH = {2: [1], 3: [3]}


@torch.no_grad()
def harvest(idx, HS):
    """Per-position write directions (B,T,D), summed over the group's heads."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    wsum = torch.zeros(B, T, D, device=DEV)
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
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        if L in HS:
            for h in HS[L]:
                yh = torch.zeros_like(y)
                yh[:, :, h, :] = y[:, :, h, :]
                wsum = wsum + at.c_proj(yh.reshape(B, T, D)).float()
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
        if L >= 4:
            break
    return wsum


@torch.no_grad()
def inject_profile(idx, d):
    """Add d at block-4 entry; return projections of stream onto d at entries 4..8."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    dn = (d * d).sum(-1).clamp_min(1e-9)
    proj = {}
    for L, blk in enumerate(m.transformer.h):
        if L == 4:
            x = x + d
        if 4 <= L <= 8:
            p = (x.float() * d).sum(-1) / dn
            proj[L] = float(p[:, QSTART:].mean())
        if L > 8:
            break
        x, v1 = blk(x, v1, x0)
    return proj


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR + 4)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    # harvest directions from a SEPARATE batch (wrong provenance for injection rows)
    hidx = ROWS[NR:NR + 4, :-1].to(DEV).contiguous()
    wm = harvest(hidx, MATCHERS)
    wn = harvest(hidx, NULLH)
    scale = wm.norm(dim=-1).median()
    g = torch.Generator(device=DEV).manual_seed(6)
    wr = torch.randn(wm.shape, device=DEV, generator=g)
    wr = wr / wr.norm(dim=-1, keepdim=True) * scale
    wn = wn / wn.norm(dim=-1, keepdim=True).clamp_min(1e-6) * scale
    wm = wm / wm.norm(dim=-1, keepdim=True).clamp_min(1e-6) * scale

    prof = {k: {} for k in ('matcher_dir', 'null_dir', 'random_dir')}
    nb = 0
    for i in range(0, NR, 4):
        idx = ROWS[i:i + 4, :-1].to(DEV).contiguous()
        for name, d in (('matcher_dir', wm), ('null_dir', wn), ('random_dir', wr)):
            pr = inject_profile(idx, d)
            for L, v in pr.items():
                prof[name][L] = prof[name].get(L, 0.0) + v
        nb += 1
    P = {k: {str(L): round(v / nb, 4) for L, v in d.items()} for k, d in prof.items()}
    def ratio(k):
        return P[k]['6'] / max(abs(P[k]['4']), 1e-6)
    rm, rn, rr = ratio('matcher_dir'), ratio('null_dir'), ratio('random_dir')
    drops = {L: P['matcher_dir'][str(L)] - P['matcher_dir'][str(L + 1)] for L in (4, 5, 6, 7)}
    locus = max(drops, key=drops.get)
    out = {'n_rows': NR, 'profiles': P,
           'L4_to_L6_ratio': {'matcher': round(rm, 3), 'null': round(rn, 3), 'random': round(rr, 3)},
           'largest_drop_block': locus,
           'pred_a_subspace_keyed': bool(rm <= 0.5 and rm <= 0.6 * rr),
           'pred_b_controls_passive': bool(0.4 <= rn / max(rr, 1e-6) <= 1.1),
           'pred_c_locus_block5': bool(locus == 5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"profiles {json.dumps(P)}")
    print(f"ratios m {rm:.3f} n {rn:.3f} r {rr:.3f} | locus block {locus}")
    print(f"pred_a subspace {out['pred_a_subspace_keyed']} | pred_b passive {out['pred_b_controls_passive']} | pred_c locus {out['pred_c_locus_block5']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
