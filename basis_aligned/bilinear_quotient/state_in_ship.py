# state_in_ship: DO THE COMPUTATIONS SURVIVE THE FULL ATTENTION REPLACEMENT?
# (S1550: no single layer matters; the strong test is all 18 at once.) All
# attention layers at whitened rank-32 simultaneously (no exempt heads), scored on
# agreement accuracy AND the parity CE gap.
#
# Registered predictions:
#   pred_a is/are accuracy under the full replacement >= .90 (clean .957).
#   pred_b the parity gap retains >= .70 of its clean value.
#   pred_c global CE lands 3.2-3.7 (all-r32-no-roster composite, consistent with
#          the S1477 composite minus roster benefits).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'state_in_ship_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')


def trunc_perhead(W, r_, Wh, Whi):
    Wf = W.float().to(DEV).view(9, 128, D)
    out = torch.zeros_like(Wf)
    for h in range(9):
        U, S, Vt = torch.linalg.svd(Wf[h] @ Wh, full_matrices=False)
        out[h] = ((U[:, :r_] * S[:r_]) @ Vt[:r_]) @ Whi
    return out.view(9 * 128, D)


@torch.no_grad()
def fwd_arm(idx, LT, TWALL):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if LT == 'all':
            TW = TWALL[L]
            qp = (xin.float() @ TW['q'].T).view(B, T, 9, 128)
            kp = (xin.float() @ TW['k'].T).view(B, T, 9, 128)
            q2p = (xin.float() @ TW['q2'].T).view(B, T, 9, 128)
            k2p = (xin.float() @ TW['k2'].T).view(B, T, 9, 128)
        else:
            qp = at.c_q(xin).view(B, T, 9, 128).float()
            kp = at.c_k(xin).view(B, T, 9, 128).float()
            q2p = at.c_q2(xin).view(B, T, 9, 128).float()
            k2p = at.c_k2(xin).view(B, T, 9, 128).float()
        cos, sin = at.rotary(qp)
        q = are(F.rms_norm(qp, (128,)), cos, sin)
        k = are(F.rms_norm(kp, (128,)), cos, sin)
        q2 = are(F.rms_norm(q2p, (128,)), cos, sin)
        k2 = are(F.rms_norm(k2p, (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    CR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    XACC = {L: torch.zeros(D, D, device=DEV) for L in range(18)}
    ncov = 0
    for i in range(0, 96, 8):
        idx = CR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1_ = None
        for L, blk in enumerate(H):
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            Xf = xin.float().reshape(-1, D)
            XACC[L] += Xf.T @ Xf
            x, v1_ = blk(x, v1_, x0)
        ncov += idx.shape[0] * T
    TWALL = {}
    for L in range(18):
        Sg = XACC[L] / ncov
        ev, Vv = torch.linalg.eigh(Sg)
        ev = ev.clamp_min(1e-6)
        Wh = Vv @ torch.diag(ev.sqrt()) @ Vv.T
        Whi = Vv @ torch.diag(ev.rsqrt()) @ Vv.T
        at = H[L].attn
        TWALL[L] = {'q': trunc_perhead(at.c_q.weight, 32, Wh, Whi),
                    'k': trunc_perhead(at.c_k.weight, 32, Wh, Whi),
                    'q2': trunc_perhead(at.c_q2.weight, 32, Wh, Whi),
                    'k2': trunc_perhead(at.c_k2.weight, 32, Wh, Whi)}
        print(f"maps L{L}", flush=True)

    tid_is = ENC.encode(' is')[0]
    tid_are = ENC.encode(' are')[0]
    import re
    QMASK = torch.zeros(50257, dtype=torch.bool)
    QCOUNT = torch.zeros(50257)
    for t in range(50257):
        s = ENC.decode([t])
        QCOUNT[t] = s.count('"') + s.count('\u201c') + s.count('\u201d')
        if re.match(r'^["\u201d]$|^ ?"$', s):
            QMASK[t] = True

    def metric(LT):
        correct = 0; total = 0
        gs = 0.0; gn = 0
        sp = 0.0; np_ = 0; si = 0.0; ni = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, LT, TWALL).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk].sum()); gn += int(mk.sum())
            cls = ((tg == tid_is) | (tg == tid_are)) & mk
            if int(cls.sum()):
                li = lo[..., tid_is][cls]
                la = lo[..., tid_are][cls]
                correct += int(((li > la) == (tg[cls] == tid_is)).sum())
                total += int(cls.sum())
            qc = QCOUNT.to(DEV)[idx]
            odd = (torch.cumsum(qc, dim=1) % 2) == 1
            qcls = QMASK.to(DEV)[tg] & mk
            proper = qcls & odd; improper = qcls & ~odd
            sp += float(ce[proper].sum()); np_ += int(proper.sum())
            si += float(ce[improper].sum()); ni += int(improper.sum())
        return (correct / max(total, 1),
                (si / max(ni, 1)) - (sp / max(np_, 1)), gs / max(gn, 1))

    acc0, gap0, g0 = metric(None)
    acc1, gap1, g1 = metric('all')
    pa = acc1 >= 0.90
    pb = gap1 >= 0.70 * gap0
    pc = 3.2 <= g1 <= 3.7
    out = {'clean': {'acc': round(acc0, 4), 'gap': round(gap0, 4),
                     'global': round(g0, 4)},
           'all_r32': {'acc': round(acc1, 4), 'gap': round(gap1, 4),
                       'global': round(g1, 4)},
           'pred_a_acc_90': bool(pa), 'pred_b_gap_70': bool(pb),
           'pred_c_global_32_37': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
