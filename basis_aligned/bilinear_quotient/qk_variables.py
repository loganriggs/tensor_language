"""QK reads EARLY VARIABLES (user's compositional vision: later layers' computation
defined in variables from earlier layers, amortized). Test whether a later attention
layer's INPUT can be restricted to the token-class + position subspace (the "variables"
the front computes) without losing the layer's function. If a head still does its job
seeing only the grammatical-class + position of each position, then its computation is
EXPRESSIBLE in the early variables -- attention attends on class + position, reading
amortized quantities rather than recomputing features.

For target attention layers (L1, L5 -- the big-benefit heads), build the token-class
+ position subspace of the layer's INPUT (rms-normed residual it reads), then measure
CE-recovery when the input is projected onto ONLY that variable subspace, vs a random
same-rank subspace.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating the attention output raises CE (benefit > 0);
  (a) READS VARIABLES: restricting the attention INPUT to the token-class+position
      subspace recovers >= 0.6 of the layer's CE benefit (>> random same-rank), so the
      head's computation is largely expressible in the early class+position variables
      (amortized composition);
  (b) report keep-only-variables recovery vs random per target layer, and token-only
      vs token+position;
  NULL: random same-rank input subspace recovers far less."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_variables_results.json'
NEVAL = 48; MINCOUNT = 5; RTOK = 64; RPOS = 32; TARGETS = [1, 5]
IN = {'U': None, 'op': None, 'L': -1}      # input-projection pre-hook state
OUTABL = {'L': -1}                          # output-ablate state


def pre_hook_factory(L):
    def pre(mo, args):
        if IN['op'] is None or IN['L'] != L: return None
        x = args[0]; sh = x.shape; v = x.reshape(-1, D).float()
        U = IN['U']; v2 = (v @ U) @ U.T if IN['op'] == 'keep' else v - (v @ U) @ U.T
        return (v2.reshape(sh).to(x.dtype),) + tuple(args[1:])
    return pre


def out_hook_factory(L):
    def h(mo, i_, o_):
        if OUTABL['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_
        z = torch.zeros_like(y)
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
    pres = [m.transformer.h[L].attn.register_forward_pre_hook(pre_hook_factory(L)) for L in TARGETS]
    outs = [m.transformer.h[L].attn.register_forward_hook(out_hook_factory(L)) for L in TARGETS]
    IN['op'] = None; OUTABL['L'] = -1; ce_full = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0); res = {}
    for L in TARGETS:
        Ain, toks, pos = capture_attn_input(rows, NEVAL, L)
        Utok = mean_subspace(Ain, toks, RTOK); Upos = mean_subspace(Ain, pos, RPOS)
        Ucomb = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
        OUTABL['L'] = L; ce_abl = ce_on(rows, NEVAL); OUTABL['L'] = -1; ben = ce_abl - ce_full
        def keeprec(U, op='keep'):
            IN['op'] = op; IN['U'] = U; IN['L'] = L; c = ce_on(rows, NEVAL); IN['op'] = None; IN['U'] = None; IN['L'] = -1
            return float((ce_abl - c)/max(ben, 1e-6))
        rc = keeprec(Ucomb); rt = keeprec(Utok)
        Ur = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]; rr = keeprec(Ur)
        res[str(L)] = {'benefit': round(ben, 4), 'keep_combined': round(rc, 4), 'keep_token': round(rt, 4), 'keep_random': round(rr, 4)}
        print(f'attn L{L}: benefit {ben:.3f} | keep-variables(token+pos) {rc:.3f} | token-only {rt:.3f} | random {rr:.3f}', flush=True)
    for h in pres + outs: h.remove()

    pa = all(res[str(L)]['keep_combined'] >= 0.6 and res[str(L)]['keep_combined'] > 1.5*res[str(L)]['keep_random'] for L in TARGETS)
    out = {'targets': TARGETS, 'results': res, 'pred_a_reads_variables': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) attention reads mostly the early token-class+position variables (>=0.6 & >>random): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
