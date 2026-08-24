# matcher_janitor: the §1244 janitor hypothesis, measured in the BASE model. If mlp4/5
# neutralize the matchers' raw −(matched value) vector, the stream's projection onto each
# position's own recorded write direction should DROP sharply across blocks 4-5 — beyond
# the passive decay that null-head writes (2.1/3.3) show over the same span.
#
# Measurement (repeat rows, positions t>=160 where matches fire): record each matcher/null
# head's per-position write w (c_proj of its head slice) during the normal forward; at each
# later block entry L, projection p_L = <x_L, w>/|w|^2 (per position, then averaged over
# positions with |w| in the top half — active writes). Report the p_L profile L=3..8.
#
# Registered predictions:
#   pred_a ACTIVE CANCELLATION: matcher projection at L6-entry <= 0.4 x its L4-entry value.
#   pred_b PASSIVE NULL: null-head projection at L6-entry >= 0.6 x its L4-entry value.
#   pred_c SPECIFICITY: the L4->L6 drop ratio (matcher / null) <= 0.6.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_janitor_results.json'
NR = 24; QSTART = 160
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
GROUPS = {'matcher': {2: [5], 3: [8]}, 'null': {2: [1], 3: [3]}}


@torch.no_grad()
def profile(idx, HS):
    """Returns dict L -> mean projection of x at block-L entry onto each write direction."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    writes = []                     # list of (B,T,D) write vectors
    proj = {}
    for L, blk in enumerate(m.transformer.h):
        if writes and L in (3, 4, 5, 6, 7, 8):
            ps = []
            for w in writes:
                wn = (w * w).sum(-1).clamp_min(1e-9)
                p = (x * w).sum(-1) / wn                     # (B,T)
                mask = wn > wn.median()
                ps.append(float(p[:, QSTART:][mask[:, QSTART:]].mean()))
            proj[L] = round(sum(ps) / len(ps), 4)
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
                writes.append(at.c_proj(yh.reshape(B, T, D)).float())
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return proj


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    acc = {g: {} for g in GROUPS}
    nb = 0
    for i in range(0, NR, 4):
        idx = ROWS[i:i + 4, :-1].to(DEV).contiguous()
        for g, HS in GROUPS.items():
            pr = profile(idx, HS)
            for L, v in pr.items():
                acc[g][L] = acc[g].get(L, 0.0) + v
        nb += 1
    prof = {g: {str(L): round(v / nb, 4) for L, v in d.items()} for g, d in acc.items()}
    m4 = prof['matcher']['4']; m6 = prof['matcher']['6']
    n4 = prof['null']['4']; n6 = prof['null']['6']
    ratio_m = m6 / max(abs(m4), 1e-6); ratio_n = n6 / max(abs(n4), 1e-6)
    out = {'n_rows': NR, 'profiles': prof,
           'L4_to_L6': {'matcher': round(ratio_m, 3), 'null': round(ratio_n, 3)},
           'pred_a_active_cancel': bool(ratio_m <= 0.4),
           'pred_b_passive_null': bool(ratio_n >= 0.6),
           'pred_c_specific': bool(ratio_m / max(ratio_n, 1e-6) <= 0.6),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"profiles {json.dumps(prof)}")
    print(f"L4->L6 matcher {ratio_m:.3f} vs null {ratio_n:.3f}")
    print(f"pred_a cancel {out['pred_a_active_cancel']} | pred_b passive {out['pred_b_passive_null']} | pred_c specific {out['pred_c_specific']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
