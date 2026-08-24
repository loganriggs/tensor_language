# extraction_v3: GOAL-1 RUNG 3 (user correction §1314: count DESCRIPTION LENGTH, not
# heads). The crowd's induction service is route-carried (§1290/93), and the route is
# nearly free in bits: keep each removed head's lambda*v1 term (existing per-layer scalar
# x block-0 values, which the extraction keeps anyway) and mean-replace only its FRESH
# values. Patterns stay live for all heads — themselves known weights-functions (§1161-66:
# all 162 window-foldable at +0.014 nats), so they count as cheap kept code.
#
# Conditions: full | allmean (v1/v2 anchor, y-level mean) | circ_route (7 circuit heads
# fully live + v1-route through all others) | closure_route (33-head closure fully live +
# v1-route through all others).
#
# Registered predictions:
#   pred_a ROUTE RESTORES THE CROWD: closure_route ident damage <= 40% of allmean's.
#   pred_b SHARED-VARIABLE LEAK (predicted, not feared): elsewhere also recovers >= 40%
#          of its gap — the same broadcast is the content pool's substrate (§1076).
#   pred_c BAND STILL NEEDED: circ_route trails closure_route by >= 0.3 nats on ident.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'extraction_v3_results.json'
NMEAN = 24; NR = 960; W = 128
H = m.transformer.h
CIRCUIT7 = {(2, 5), (3, 8), (8, 3), (8, 4), (5, 7), (1, 1), (1, 8)}
CLOSURE33 = CIRCUIT7 | {(L, h) for L in (0, 1, 2) for h in range(9)} | {(5, 7), (8, 3), (8, 4)}


def stem(s):
    s = s.strip().lower()
    for suf in ('ing', 'es', 'ed', 's', 'd'):
        if s.endswith(suf) and len(s) - len(suf) > 3:
            return s[:-len(suf)]
    return s


@torch.no_grad()
def fwd_route(idx, keep, vmeans, ymeans, mode):
    """mode 'full': normal. mode 'ymean': non-kept heads' y-slices -> per-head mean
    (v1/v2 instrument). mode 'route': non-kept heads keep lambda*v1 but fresh values ->
    per-head mean vector (the route extraction)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        are = sys.modules[type(at).__module__].apply_rotary_emb
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
        v1v = v1.view_as(v)
        vv = (1 - at.lamb) * v + at.lamb * v1v
        if mode == 'route':
            sel = torch.tensor([(L, h) not in keep for h in range(9)], device=DEV).view(1, 1, 9, 1)
            vfixed = (1 - at.lamb) * vmeans[L].view(1, 1, 9, 128).to(vv.dtype) + at.lamb * v1v
            vv = torch.where(sel, vfixed, vv)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        if mode == 'ymean':
            y = y.clone()
            for h in range(9):
                if (L, h) not in keep:
                    y[:, :, h * 128:(h + 1) * 128] = ymeans[L][h * 128:(h + 1) * 128].to(y.dtype)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    stem_id = torch.zeros(50257, dtype=torch.long); smap = {}
    for tok in range(50257):
        try:
            s = stem(enc.decode([tok]))
        except Exception:
            s = f'<{tok}>'
        if s not in smap:
            smap[s] = len(smap)
        stem_id[tok] = smap[s]

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    stems = stem_id[toks]
    IDENT = torch.zeros_like(toks, dtype=torch.bool)
    VAR = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, NR, 64):
        tb = toks[b0:b0 + 64]; gb = tgt[b0:b0 + 64]; sb = stems[b0:b0 + 64]
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - W)
        cont = (gb.unsqueeze(1) == gb.unsqueeze(2))
        IDENT[b0:b0 + 64] = ((tb.unsqueeze(1) == tb.unsqueeze(2)) & cont & band).any(1)
        VAR[b0:b0 + 64] = ((sb.unsqueeze(1) == sb.unsqueeze(2)) &
                           ~(tb.unsqueeze(1) == tb.unsqueeze(2)) & cont & band).any(1)
    VAR &= ~IDENT
    IDENT[:, :16] = False; VAR[:, :16] = False
    ELSE = ~IDENT & ~VAR; ELSE[:, :16] = False
    print(f"ident {int(IDENT.sum())} | var {int(VAR.sum())}", flush=True)

    # capture per-head fresh-value means (c_v) and y-level means on MEANR
    vcaps = {L: [] for L in range(18)}; ycaps = {L: [] for L in range(18)}; hooks = []
    for L in range(18):
        def mkv(L):
            def h(mod, args, out):
                vcaps[L].append(out.detach().float().view(out.shape[0], -1, 9, 128).mean((0, 1)))
                return out
            return h
        def mky(L):
            def h(mod, args):
                ycaps[L].append(args[0].detach().float().mean((0, 1)))
            return h
        hooks.append(H[L].attn.c_v.register_forward_hook(mkv(L)))
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mky(L)))
    for i in range(0, NMEAN, 4):
        fwd_route(MEANR[i:i + 4, :-1].to(DEV).contiguous(), set(), None, None, 'full')
    for h in hooks:
        h.remove()
    vmeans = {L: torch.stack(v).mean(0).to(DEV) for L, v in vcaps.items()}
    ymeans = {L: torch.stack(v).mean(0).to(DEV) for L, v in ycaps.items()}

    NAMES = ('ident', 'var', 'els'); SETS = (IDENT, VAR, ELSE)

    def ce_sets(keep, mode):
        tots = {k: 0.0 for k in NAMES}; ns = {k: 0 for k in NAMES}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_route(idx, keep, vmeans, ymeans, mode).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in zip(NAMES, SETS):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    res = {}
    for cname, keep, mode in (('full', set(), 'full'), ('allmean', set(), 'ymean'),
                              ('circ_route', CIRCUIT7, 'route'),
                              ('closure_route', CLOSURE33, 'route')):
        r = ce_sets(keep, mode)
        res[cname] = {k: round(v, 4) for k, v in r.items()}
        print(f"{cname}: {res[cname]}", flush=True)

    d = {c: {k: res[c][k] - res['full'][k] for k in NAMES} for c in res if c != 'full'}
    pa = d['closure_route']['ident'] <= 0.4 * max(d['allmean']['ident'], 1e-4)
    pb = d['closure_route']['els'] <= 0.6 * d['allmean']['els']
    pc = d['circ_route']['ident'] - d['closure_route']['ident'] >= 0.3
    out = {'n_rows': NR, 'ce': res,
           'dmg_vs_full': {c: {k: round(v, 4) for k, v in dd.items()} for c, dd in d.items()},
           'pred_a_route_restores': bool(pa), 'pred_b_shared_leak': bool(pb),
           'pred_c_band_needed': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a route {pa} | pred_b leak {pb} | pred_c band {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
