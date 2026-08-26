# state_localize_A: WHERE DOES THE AGREEMENT STATE LIVE? (S1548: the copula circuit
# announces; the number-agreement computation survives its removal.) Per-layer probe:
# replace ONE attention layer at a time with its whitened rank-32 approximation
# (S1474 class) and measure the is/are restricted accuracy — the layer(s) whose
# approximation destroys agreement carry the state.
#
# Registered predictions:
#   pred_a at least one layer drops restricted accuracy by >= .05.
#   pred_b the largest-drop layer is <= 11 (state computed before the L11/15
#          announcer reads it).
#   pred_c median accuracy drop across layers <= .02 (the state is localized, not
#          smeared).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'state_localize_A_results.json'
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
        if L == LT:
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

    def metric(LT):
        correct = 0; total = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, LT, TWALL).float()
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cls = ((tg == tid_is) | (tg == tid_are)) & mk
            if int(cls.sum()):
                li = lo[..., tid_is][cls]
                la = lo[..., tid_are][cls]
                correct += int(((li > la) == (tg[cls] == tid_is)).sum())
                total += int(cls.sum())
        return correct / max(total, 1)

    acc0 = metric(-1)
    res = {'clean_acc': round(acc0, 4)}
    drops = {}
    for L in range(18):
        a = metric(L)
        drops[L] = round(acc0 - a, 4)
        res[f'L{L}'] = {'acc': round(a, 4), 'drop': drops[L]}
        print(f"L{L}: acc {a:.4f} drop {drops[L]:+.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    import statistics
    worst = max(drops, key=lambda L: drops[L])
    pa = drops[worst] >= 0.05
    pb = worst <= 11
    pc = statistics.median(drops.values()) <= 0.02
    out = {'res': res, 'worst_layer': worst, 'worst_drop': drops[worst],
           'median_drop': round(statistics.median(drops.values()), 4),
           'pred_a_some_05': bool(pa), 'pred_b_worst_le_11': bool(pb),
           'pred_c_median_le_02': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"worst L{worst} ({drops[worst]:+.4f})")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
