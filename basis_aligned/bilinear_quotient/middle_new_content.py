"""WHAT is the ~43% NEW class+position the middle adds — finer CLASS or finer POSITION?
(§818 follow-up). §818 showed the middle adds class+position in mostly-new directions (own
0.65 vs front-subspace 0.37). Decompose that 'new' content: for the middle band, measure
keep-only recovery of its collective benefit when keeping (i) the FRONT's class+position
subspace alone (baseline, ~0.37); (ii) front + middle's TOKEN-driven directions; (iii) front
+ middle's POSITION-driven directions; (iv) middle's OWN full class+position (~0.65). The
incremental recovery from adding middle-TOKEN vs middle-POSITION says which variable the
middle newly refines. Matched-rank random-add null for fairness.

REGISTERED PREDICTIONS:
  (0) SANITY: (i) reproduces ~0.37, (iv) ~0.65; random-add null gains little over (i);
  (a) report which of token / position, added to the front subspace, closes more of the gap
      to 0.65 -> the middle newly refines CLASS (token) or POSITION;
  (b) if both add and neither dominates, the middle refines both; if token+position together
      ~= own, the new content is fully token/position (not interaction);
  NULL: matched-rank random directions added to front recover far less than token/position."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_new_content_results.json'
NEVAL = 160; MINCOUNT = 5; RTOK = 64; RPOS = 32
FRONT = list(range(0, 6)); MID = list(range(6, 12))
MEANS = {}; TOKD = {}; POSD = {}; MODE = {'op': None, 'U': None, 'active': set()}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    name = (w, L)
    def hook(mo, i_, o_):
        if MODE['op'] is None or name not in MODE['active']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: mu = MEANS[name]; U = MODE['U']; v2 = mu + ((v - mu) @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


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
def capture(rows, n, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r, gmean):
    rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - gmean[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def orth(*mats):
    C = torch.cat([x for x in mats if x is not None], 1)
    return torch.linalg.svd(C, full_matrices=False)[0][:, :C.shape[1]].contiguous()


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    MODE['op'] = None
    front_dirs = []; mid_tok = []; mid_pos = []
    for L in FRONT + MID:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture(rows, NEVAL, w, L)
            MEANS[(w, L)] = O.mean(0, keepdim=True)
            Ut = mean_subspace(O, toks, RTOK, MEANS[(w, L)]); Up = mean_subspace(O, pos, RPOS, MEANS[(w, L)])
            if L in FRONT:
                front_dirs.append(torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous())
            else:
                TOKD[(w, L)] = Ut; POSD[(w, L)] = Up; mid_tok.append(Ut); mid_pos.append(Up)
    # reduce each aggregate to a fixed rank (BUGFIX: orth(*dirs) kept all 1152 cols = full rank)
    U_front = torch.linalg.svd(torch.cat(front_dirs, 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()  # rank 96
    U_mtok = torch.linalg.svd(torch.cat(mid_tok, 1), full_matrices=False)[0][:, :RTOK].contiguous()           # rank 64
    U_mpos = torch.linalg.svd(torch.cat(mid_pos, 1), full_matrices=False)[0][:, :RPOS].contiguous()           # rank 32
    U_own = orth(U_mtok, U_mpos)                     # middle own class+position (rank 96)
    g = torch.Generator(device=DEV).manual_seed(0)
    U_rtok = torch.linalg.qr(torch.randn(D, RTOK, generator=g, device=DEV))[0]
    U_rpos = torch.linalg.qr(torch.randn(D, RPOS, generator=g, device=DEV))[0]

    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for L in MID for w in ('attn', 'mlp')]
    MODE['active'] = {(w, L) for L in MID for w in ('attn', 'mlp')}
    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
    def rec(U): MODE['op'] = 'keep'; MODE['U'] = U; c = ce_on(rows, NEVAL); MODE['op'] = None; return round(float((ce_abl-c)/max(ben, 1e-6)), 4)
    res = {
        'front_only': rec(U_front),
        'front_plus_midtoken': rec(orth(U_front, U_mtok)),
        'front_plus_midpos': rec(orth(U_front, U_mpos)),
        'front_plus_randtoken': rec(orth(U_front, U_rtok)),
        'front_plus_randpos': rec(orth(U_front, U_rpos)),
        'middle_own': rec(U_own),
    }
    MODE['active'] = set()
    for h in hooks: h.remove()
    tok_gain = res['front_plus_midtoken'] - res['front_only'] - max(res['front_plus_randtoken'] - res['front_only'], 0)
    pos_gain = res['front_plus_midpos'] - res['front_only'] - max(res['front_plus_randpos'] - res['front_only'], 0)
    verdict = ('finer CLASS (token)' if tok_gain > 1.5*max(pos_gain, 0.01) else
               'finer POSITION' if pos_gain > 1.5*max(tok_gain, 0.01) else 'both class and position')
    out = {'middle_benefit': round(ben, 4), 'keep': res, 'net_token_gain': round(tok_gain, 4),
           'net_position_gain': round(pos_gain, 4), 'verdict': verdict, 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'middle benefit {ben:.3f} | keep {res}', flush=True)
    print(f'net token gain {tok_gain:+.3f} | net position gain {pos_gain:+.3f} -> middle newly refines: {verdict}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
