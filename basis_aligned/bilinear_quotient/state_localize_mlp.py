# state_localize_mlp: THE MLP SIDE OF STATE LOCALIZATION (S1550: no single
# attention layer carries agreement or parity). Replace ONE MLP at a time with its
# top-2048-unit approximation (mlps 4-17) or its verified plank (mlp0/1: tier
# table; mlp2: lin2 ridge — approximated here as K=2048 units too for uniformity,
# registered assumption) and score BOTH metrics.
#
# Registered predictions:
#   pred_a max agreement-accuracy drop across all 18 MLPs <= .02 (distributed).
#   pred_b the largest parity-gap shrink is >= .10 AND occurs at mlp0 or mlp1
#          (parity rides token-identity features).
#   pred_c median effects ~0 (agreement <= .005, parity <= .02).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'state_localize_mlp_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')


UNITS = {}


@torch.no_grad()
def fwd_arm(idx, LT, TWALL=None):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(-1, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(-1, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        x = xm + at.c_proj(y.reshape(-1, T, D))
        z = F.rms_norm(x, (D,))
        if L == LT:
            U = UNITS[L]
            h = (z.float() @ U['l'].T) * (z.float() @ U['r'].T)
            x = x + (h @ U['d'].T + U['b']).to(x.dtype)
        else:
            x = x + blk.mlp(z)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    # unit approximations for every MLP (K=2048, ranked on 96 rows)
    HR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    store = {}
    def mk_pre(L):
        def hk(mod, args):
            store.setdefault(L, []).append(args[0].detach())
            return None
        return hk
    pre_hooks = [H[L].mlp.register_forward_pre_hook(mk_pre(L)) for L in range(18)]
    acc1 = {L: 0 for L in range(18)}; acc2 = {L: 0 for L in range(18)}; n0 = 0
    for i in range(0, 96, 8):
        store.clear()
        idxh = HR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idxh), (D,)); x0 = x; v1_ = None
        for L, blk in enumerate(H):
            x, v1_ = blk(x, v1_, x0)
        for L in range(18):
            zz = store[L][0]
            hh_ = (H[L].mlp.Left(zz).float() * H[L].mlp.Right(zz).float()) \
                .reshape(-1, H[L].mlp.Left.weight.shape[0])
            acc1[L] = acc1[L] + hh_.sum(0); acc2[L] = acc2[L] + (hh_ * hh_).sum(0)
        n0 += 8 * T
    for hk in pre_hooks:
        hk.remove()
    for L in range(18):
        mu = acc1[L] / n0
        hsd = (acc2[L] / n0 - mu * mu).clamp_min(0).sqrt()
        score = hsd * H[L].mlp.Down.weight.float().norm(dim=0)
        topu = score.argsort(descending=True)[:2048]
        UNITS[L] = {'l': H[L].mlp.Left.weight.float()[topu].clone(),
                    'r': H[L].mlp.Right.weight.float()[topu].clone(),
                    'd': H[L].mlp.Down.weight.float()[:, topu].clone(),
                    'b': H[L].mlp.Down_bias.detach().float().clone()}
    print("unit planks built", flush=True)
    TWALL = None
    import re
    QMASK = torch.zeros(50257, dtype=torch.bool)
    QCOUNT = torch.zeros(50257)
    for t in range(50257):
        s = ENC.decode([t])
        QCOUNT[t] = s.count('"') + s.count('\u201c') + s.count('\u201d')
        if re.match(r'^["\u201d]$|^ ?"$', s):
            QMASK[t] = True
    tid_is = ENC.encode(' is')[0]
    tid_are = ENC.encode(' are')[0]

    tid_are = ENC.encode(' are')[0]

    def metric(LT):
        correct = 0; total = 0
        sp = 0.0; np_ = 0; si = 0.0; ni = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, LT).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
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
                (si / max(ni, 1)) - (sp / max(np_, 1)))

    acc0, gap0 = metric(-1)
    res = {'clean': {'acc': round(acc0, 4), 'gap': round(gap0, 4)}}
    adrops = {}; gshr = {}
    for L in range(18):
        a, g = metric(L)
        adrops[L] = round(acc0 - a, 4); gshr[L] = round(gap0 - g, 4)
        res[f'mlp{L}'] = {'acc_drop': adrops[L], 'gap_shrink': gshr[L]}
        print(f"mlp{L}: acc {adrops[L]:+.4f} gap {gshr[L]:+.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    import statistics
    wa = max(adrops, key=lambda L: adrops[L])
    wg = max(gshr, key=lambda L: gshr[L])
    pa = adrops[wa] <= 0.02
    pb = gshr[wg] >= 0.10 and wg in (0, 1)
    pc = statistics.median(adrops.values()) <= 0.005 \
        and statistics.median(gshr.values()) <= 0.02
    out = {'res': res, 'worst_agreement': [wa, adrops[wa]],
           'worst_parity': [wg, gshr[wg]],
           'pred_a_agreement_distributed': bool(pa),
           'pred_b_parity_token_modules': bool(pb),
           'pred_c_medians_flat': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
