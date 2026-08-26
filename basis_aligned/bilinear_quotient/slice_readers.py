# slice_readers: WHO READS THE SHARED VARIABLE? S1597 certified the question
# slice writer graph: cutting span(v1,v2) at its 4 sources (attn10, attn9,
# mlp9, mlp10) costs .814 class CE at ZERO global cost — but mlp11's own
# rank-2 readout accounts for only .178. So ~.64 of the damage is consumed by
# OTHER downstream readers of the same 2-dim subspace. Map them:
#   READER-SIDE EDIT: mean-substitute the span(v1,v2) coordinates in the
#   INPUT of one downstream component at a time (mlp11..17 via their z; attn
#   11..17 via xin fed to all five projections c_q/c_k/c_q2/c_k2/c_v), and
#   measure the question class rise each edit causes. Directions transfer
#   across layers because rms_norm is a per-position scalar (span is
#   direction-preserved in every normalized input).
#   Completeness: the JOINT all-reader input edit vs the S1597 source cut
#   (re-measured in the same rows).
# NR=960 eval, 96 fit rows, question class.
# Registered predictions:
#   pred_a SHARED: mlp11 is the largest single reader but carries < .50 of
#          the summed individual reader rises (the variable has real fan-out).
#   pred_b COMPLETE: the joint all-reader edit reproduces >= .70 of the
#          source-cut class rise in the same rows.
#   pred_c FREE: the joint reader edit's global rise <= .01 (the subspace is
#          class-private for every reader, not just mlp11).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'slice_readers_results.json'
NR = 960
SITE = 11
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
WRITERS = ['attn10', 'attn9', 'mlp9', 'mlp10']       # S1597 causal set
RD = {'set': set(), 'V': None, 'mu': None}           # reader input edits
WR = {'on': False, 'V': None, 'mu': None}            # writer output edits


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
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
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
    RD['V'] = V2; WR['V'] = V2

    mlp_readers = [f'mlp{L}' for L in range(SITE, 18)]
    attn_readers = [f'attn{L}' for L in range(SITE, 18)]
    readers = mlp_readers + attn_readers

    # fit pass: mean input coords per reader, mean output coords per writer
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
    print('fit means done', flush=True)

    # edit hooks (readers input-side; writers output-side for the reference)
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
    source_rise = cS - c0
    res = {'clean': {'global': round(g0, 4), 'class': round(c0, 4)},
           'source_cut': {'rise_class': round(source_rise, 4),
                          'rise_global': round(gS - g0, 4)},
           'readers': {}}
    print('source cut', res['source_cut'], flush=True)

    for nm in readers:
        RD['set'] = {nm}
        g1, c1 = measure()
        RD['set'] = set()
        res['readers'][nm] = {'rise_class': round(c1 - c0, 4),
                              'rise_global': round(g1 - g0, 4)}
        print(nm, res['readers'][nm], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    RD['set'] = set(readers)
    gJ, cJ = measure()
    RD['set'] = set()
    res['joint_readers'] = {'rise_class': round(cJ - c0, 4),
                            'rise_global': round(gJ - g0, 4),
                            'frac_of_source': round((cJ - c0)
                                                    / max(source_rise, 1e-9),
                                                    3)}
    print('joint', res['joint_readers'], flush=True)

    rises = {nm: res['readers'][nm]['rise_class'] for nm in readers}
    tot_pos = sum(v for v in rises.values() if v > 0)
    top_reader = max(rises, key=lambda k: rises[k])
    res['reader_shares'] = {nm: round(max(rises[nm], 0.0)
                                      / max(tot_pos, 1e-9), 3)
                            for nm in sorted(rises, key=lambda k: -rises[k])[:8]}
    pa = (top_reader == 'mlp11' and rises['mlp11'] > 0
          and rises['mlp11'] / max(tot_pos, 1e-9) < 0.50)
    pb = res['joint_readers']['frac_of_source'] >= 0.70
    pc = res['joint_readers']['rise_global'] <= 0.01
    for hk in hooks:
        hk.remove()
    out = {'res': res, 'top_reader': top_reader,
           'pred_a_shared_fanout': bool(pa), 'pred_b_complete_70': bool(pb),
           'pred_c_free_global': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
