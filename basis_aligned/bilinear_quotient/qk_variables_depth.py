"""QK VARIABLE-READING PROFILE across all attention layers (783 found attn L1 reads
token-class, attn L5 reads position + a further variable -- map which early variable
EACH attention layer reads: the amortized-composition graph). For every attention
layer, restrict its INPUT to token-class / token-class+position, measure CE-recovery;
classify each head as class-reader, position-reader, or reads-something-else. Emits a
stacked-bar figure (class share / position share / remainder) weighted by benefit.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating each attention output raises CE (benefit >= 0);
  (a) HEAD-SPECIFIC VARIABLE READING: attention layers differ in which early variable
      they read -- some are class-readers (token-only recovery high), some position-
      readers (position adds a lot), some need a further variable (combined < 0.6);
      report the per-layer profile;
  (b) most of the BIG-benefit attention layers read mostly class+position (combined
      >= 0.6), with a minority (e.g. L5) needing a further variable;
  NULL: random same-rank input subspace recovers far less at every layer."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

D = 1152; NL = 18
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_variables_depth_results.json'; FIG = PT + 'qk_variables_depth.png'
NEVAL = 48; MINCOUNT = 5; RTOK = 64; RPOS = 32
IN = {'U': None, 'op': None, 'L': -1}; OUTABL = {'L': -1}
BLUE, GREEN, MUTEDC = '#3987e5', '#2e8b57', '#c9c7bf'


def pre_hook_factory(L):
    def pre(mo, args):
        if IN['op'] is None or IN['L'] != L: return None
        x = args[0]; sh = x.shape; v = x.reshape(-1, D).float()
        U = IN['U']; v2 = (v @ U) @ U.T
        return (v2.reshape(sh).to(x.dtype),) + tuple(args[1:])
    return pre


def out_hook_factory(L):
    def h(mo, i_, o_):
        if OUTABL['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
        return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture_attn_input(rows, n, L):
    cap = []; toks = []; pos = []
    def pre(mo, args): cap.append(args[0].detach().float().reshape(-1, D))
    h = m.transformer.h[L].attn.register_forward_pre_hook(pre)
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    h.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    pres = [m.transformer.h[L].attn.register_forward_pre_hook(pre_hook_factory(L)) for L in range(NL)]
    outs = [m.transformer.h[L].attn.register_forward_hook(out_hook_factory(L)) for L in range(NL)]
    IN['op'] = None; OUTABL['L'] = -1; ce_full = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0); res = {}
    for L in range(NL):
        Ain, toks, pos = capture_attn_input(rows, NEVAL, L)
        Utok = mean_subspace(Ain, toks, RTOK); Upos = mean_subspace(Ain, pos, RPOS)
        Ucomb = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
        OUTABL['L'] = L; ce_abl = ce_on(rows, NEVAL); OUTABL['L'] = -1; ben = ce_abl - ce_full
        def keeprec(U):
            IN['op'] = 'keep'; IN['U'] = U; IN['L'] = L; c = ce_on(rows, NEVAL); IN['op'] = None; IN['U'] = None; IN['L'] = -1
            return float((ce_abl - c)/max(ben, 1e-6))
        rt = keeprec(Utok); rc = keeprec(Ucomb)
        Ur = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]; rr = keeprec(Ur)
        res[str(L)] = {'benefit': round(ben, 4), 'keep_token': round(rt, 4), 'keep_combined': round(rc, 4), 'keep_random': round(rr, 4)}
        print(f'attn L{L}: ben {ben:.2f} | class {rt:.2f} +pos {rc:.2f} rand {rr:.2f}', flush=True)
    for h in pres + outs: h.remove()

    xs = np.arange(NL)
    cls = np.array([np.clip(res[str(L)]['keep_token'], 0, 1) for L in xs])
    comb = np.array([np.clip(res[str(L)]['keep_combined'], 0, 1) for L in xs])
    posadd = np.clip(comb - cls, 0, 1); rem = np.clip(1 - comb, 0, 1)
    fig, ax = plt.subplots(figsize=(11, 5.4)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.bar(xs, cls, color=BLUE, label='token-class (grammatical category)')
    ax.bar(xs, posadd, bottom=cls, color=GREEN, label='+ position')
    ax.bar(xs, rem, bottom=comb, color=MUTEDC, label='other variable (unread by class+position)')
    for L in xs: ax.text(L, 1.02, f'{res[str(L)]["benefit"]:.1f}', ha='center', fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 1.13); ax.set_xlabel('attention layer'); ax.set_ylabel('share of layer function readable from input variable')
    ax.set_title('Which early variable each attention layer reads (amortized composition)\n'
                 'number above bar = layer benefit (nats); grey = a variable beyond token-class + position',
                 color=INK, fontsize=12.5, loc='left')
    ax.set_xticks(xs); ax.legend(fontsize=9, loc='lower center'); ax.grid(True, axis='y', color=GRID, lw=0.6)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(SECONDARY)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE); print('wrote', FIG, flush=True)

    real = [L for L in range(NL) if res[str(L)]['benefit'] > 0.2]
    pb = sum(res[str(L)]['keep_combined'] >= 0.6 for L in real) >= 0.6*len(real)
    out = {'results': res, 'pred_b_most_read_class_position': bool(pb), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(b) most big-benefit attention layers read mostly class+position: {pb}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
