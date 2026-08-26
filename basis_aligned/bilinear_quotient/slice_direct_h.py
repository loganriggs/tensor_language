# slice_direct_h: HELD-OUT replication of slice_direct (skip=15000).
# slice_direct: IS THE QUESTION SLICE AN OUTPUT CHANNEL? S1599 showed the
# span(v1,v2) source cut (+.812 class) is NOT consumed by any computational
# reader (joint reader-input cut = -.205, it HELPS) — the damage must ride
# the direct path residual -> final rms_norm -> unembedding. Certify:
#   1. LOGIT-LENS RANKINGS: decode the top-10 tokens of WU @ (+-v1), (+-v2).
#      Is '?' itself at the top of the channel?
#   2. FINAL-RESIDUAL CUT: mean-substitute the span(v1,v2) coords of x only
#      at the final readout (before the last rms_norm), leaving all internal
#      computation intact.
#   3. COMPLETENESS: final cut + all-reader input cuts vs the source cut —
#      do the two decompositions meet?
# NR=960 eval (skip=SKIP), 96 fit rows, question class.
# Registered predictions:
#   pred_a a question-class token ('?' or ' ?') appears in the top-10 of
#          50257 by WU projection for at least one signed direction.
#   pred_b the final-residual cut alone costs >= .5 x the source cut's class
#          rise, with global rise <= .02.
#   pred_c final + joint-reader cut reproduces the source cut within 25%
#          (ratio in [.75, 1.25]).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'slice_direct_h_results.json'
SKIP = 15000
NR = 960
SITE = 11
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
WRITERS = ['attn10', 'attn9', 'mlp9', 'mlp10']
RD = {'set': set(), 'V': None, 'mu': None}
WR = {'on': False, 'V': None, 'mu': None}
FIN = {'on': False, 'V': None, 'mu': None}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def edit_span(t, mu):
    tf = t.float()
    pv = tf @ RD['V']
    return (tf - (pv - mu) @ RD['V'].T).to(t.dtype)


def mk_reader_mlp_pre(L):
    def hook(mod, args):
        nm = f'mlp{L}'
        if nm not in RD['set']:
            return None
        return (edit_span(args[0], RD['mu'][nm]),) + tuple(args[1:])
    return hook


def mk_reader_attn_pre(L):
    def hook(mod, args):
        nm = f'attn{L}'
        if nm not in RD['set']:
            return None
        return (edit_span(args[0], RD['mu'][nm]),) + tuple(args[1:])
    return hook


def mk_writer_hook(nm):
    def hook(mod, args, output):
        if not WR['on']:
            return None
        o = output.float()
        pv = o @ WR['V']
        return (o - (pv - WR['mu'][nm]) @ WR['V'].T).to(output.dtype)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    if FIN['on']:
        xf = x.float()
        pv = xf @ FIN['V']
        x = (xf - (pv - FIN['mu']) @ FIN['V'].T).to(x.dtype)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=SKIP)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    mask_v = rx(r'^\?$| \?$')
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()

    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    wdir = u @ Dw
    Q = Lw.T @ (wdir[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    order = lam.abs().argsort(descending=True)[:2]
    V2 = V[:, order].contiguous()
    RD['V'] = V2; WR['V'] = V2; FIN['V'] = V2

    # logit-lens rankings of the channel directions
    rank_report = {}
    qtoks = set(torch.nonzero(mask_v).flatten().tolist())
    best_rank = 50257
    for i, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        sc = WU @ (sgn * V2[:, i])
        top = sc.argsort(descending=True)[:10].tolist()
        rk = 50257
        for pos, tk in enumerate(sc.argsort(descending=True)[:200].tolist()):
            if tk in qtoks:
                rk = pos + 1
                break
        best_rank = min(best_rank, rk)
        rank_report[f'v{i + 1}{"+" if sgn > 0 else "-"}'] = {
            'top10': [ENC.decode([t]) for t in top],
            'first_question_rank': rk if rk <= 200 else '>200'}
    print(json.dumps(rank_report, indent=1), flush=True)

    readers = [f'mlp{L}' for L in range(SITE, 18)] \
        + [f'attn{L}' for L in range(SITE, 18)]
    cap = {}
    def mk_cap_pre(nm):
        def hook(mod, args):
            cap[nm]['s'] = cap[nm]['s'] + (args[0].float().reshape(-1, D)
                                           @ V2).sum(0)
            cap[nm]['n'] += args[0].shape[0] * args[0].shape[1]
            return None
        return hook
    def mk_cap_out(nm):
        def hook(mod, args, output):
            cap[nm]['s'] = cap[nm]['s'] + (output.float().reshape(-1, D)
                                           @ V2).sum(0)
            cap[nm]['n'] += output.shape[0] * output.shape[1]
            return None
        return hook
    tmp = []
    for nm in readers + [w + '_out' for w in WRITERS]:
        cap[nm] = {'s': torch.zeros(2, device=DEV), 'n': 0}
    for L in range(SITE, 18):
        tmp.append(H[L].mlp.register_forward_pre_hook(mk_cap_pre(f'mlp{L}')))
        tmp.append(H[L].attn.c_q.register_forward_pre_hook(
            mk_cap_pre(f'attn{L}')))
    for w in WRITERS:
        L = int(w[4:]) if w.startswith('attn') else int(w[3:])
        mod = H[L].attn.c_proj if w.startswith('attn') else H[L].mlp
        tmp.append(mod.register_forward_hook(mk_cap_out(w + '_out')))
    for i in range(0, 96, 8):
        fwd(FR[i:i + 8, :-1].to(DEV).contiguous())
    for hk in tmp:
        hk.remove()
    RD['mu'] = {nm: cap[nm]['s'] / max(cap[nm]['n'], 1) for nm in readers}
    WR['mu'] = {w: cap[w + '_out']['s'] / max(cap[w + '_out']['n'], 1)
                for w in WRITERS}
    # FIN mean must be fit on the PRE-norm final x (where the edit applies):
    finacc = torch.zeros(2, device=DEV); finn = 0
    for i in range(0, 96, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1_ = None
        for blk in H:
            x, v1_ = blk(x, v1_, x0)
        finacc += (x.float().reshape(-1, D) @ V2).sum(0)
        finn += x.shape[0] * x.shape[1]
    FIN['mu'] = finacc / max(finn, 1)
    print('fit means done', flush=True)

    hooks = []
    for L in range(SITE, 18):
        hooks.append(H[L].mlp.register_forward_pre_hook(mk_reader_mlp_pre(L)))
        for sub in ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v'):
            hooks.append(getattr(H[L].attn, sub).register_forward_pre_hook(
                mk_reader_attn_pre(L)))
    for w in WRITERS:
        L = int(w[4:]) if w.startswith('attn') else int(w[3:])
        mod = H[L].attn.c_proj if w.startswith('attn') else H[L].mlp
        hooks.append(mod.register_forward_hook(mk_writer_hook(w)))

    def measure():
        gs = 0.0; gn = 0; cs = 0.0; cn_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn_ += int(cm.sum())
        return gs / max(gn, 1), cs / max(cn_, 1)

    g0, c0 = measure()
    WR['on'] = True
    gS, cS = measure()
    WR['on'] = False
    src = cS - c0
    FIN['on'] = True
    gF, cF = measure()
    RD['set'] = set(readers)
    gA, cA = measure()
    RD['set'] = set(); FIN['on'] = False
    res = {'rankings': rank_report, 'best_question_rank':
           best_rank if best_rank <= 200 else '>200',
           'clean': {'global': round(g0, 4), 'class': round(c0, 4)},
           'source_cut': {'rise_class': round(src, 4),
                          'rise_global': round(gS - g0, 4)},
           'final_cut': {'rise_class': round(cF - c0, 4),
                         'rise_global': round(gF - g0, 4),
                         'frac_of_source': round((cF - c0)
                                                 / max(src, 1e-9), 3)},
           'final_plus_readers': {'rise_class': round(cA - c0, 4),
                                  'rise_global': round(gA - g0, 4),
                                  'ratio_to_source': round((cA - c0)
                                                           / max(src, 1e-9),
                                                           3)}}
    print(json.dumps(res, indent=1), flush=True)
    for hk in hooks:
        hk.remove()

    pa = best_rank <= 10
    pb = (cF - c0) >= 0.5 * src and (gF - g0) <= 0.02
    ratio = (cA - c0) / max(src, 1e-9)
    pc = 0.75 <= ratio <= 1.25
    out = {'res': res, 'pred_a_channel_top10': bool(pa),
           'pred_b_final_cut_half': bool(pb),
           'pred_c_decomposition_complete': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
