"""STRESS-TEST THE CAUSAL PILLAR: is class steering CLASS-SPECIFIC? (§823 verification at the
§836 null standard). After §836 retracted the keep-only magnitude, the surviving evidence that
class+position is REAL and CAUSAL is naming (§825) + steering (§823). §823 used a random-direction
null; the decisive check is specificity: inject class B's deviation at the front components
(amplified, as §823) and measure how much the prediction moves toward EACH class's typical
continuation p_C. If steering toward B reduces KL to p_B MORE than to other classes' continuations
(off-diagonal), the steering is class-specific — real causal control, not a generic perturbation.

REGISTERED PREDICTIONS:
  (0) SANITY: unsteered KL(·‖p_C) matrix is baseline;
  (a) SPECIFIC: injecting class B most reduces KL to p_B (the diagonal drop is the largest in its
      row) — steering moves predictions toward the INJECTED class specifically; averaged, the
      diagonal drop >> mean off-diagonal drop -> class steering is causally class-specific;
  (b) if the diagonal is not special (injecting B moves toward all classes equally), steering is a
      generic effect, not class-specific (would undercut §823).
  Uses token-class sources (frequent tokens standing for their class)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'steering_specificity_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
SRC = [262, 257, 290, 13]        # " the"(det), " a"(det), " and"(conj), "."(punct)
FRONT = list(range(0, 6)); ALPHA = 16.0
ST = {'on': False, 'delta': {}}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    key = (w, L)
    def hook(mo, i_, o_):
        if not ST['on']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        v2 = v + ALPHA * ST['delta'][key]
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def avg_pred(rows, tok_id=None):
    ps = []
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        lg = forward_logits(idx).float(); p = F.softmax(lg, -1).reshape(-1, lg.shape[-1])
        if tok_id is None: ps.append(p.mean(0).cpu())
        else:
            mk = (idx.reshape(-1) == tok_id)
            if mk.any(): ps.append(p[mk].cpu())
    return (torch.cat(ps, 0) if tok_id is not None else torch.stack(ps, 0)).mean(0)


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p = p/p.sum(); q = q/q.sum()
    return float((p*(p/q).log()).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    # per-component: class+position subspace, global mean, per-source deviation delta
    subs = {}; gm = {}; tmeanbytok = {}
    for L in FRONT:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture(rows, w, L); g = O.mean(0, keepdim=True)
            Ut = mean_subspace(O, toks, RTOK); Up = mean_subspace(O, pos.astype(np.int64), RPOS)
            U = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
            subs[(w, L)] = U; gm[(w, L)] = g
            for b in SRC:
                mk = toks == b
                if mk.sum() >= MINCOUNT: tmeanbytok[(b, w, L)] = O[mk].mean(0, keepdim=True).to(DEV)
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for L in FRONT for w in ('attn', 'mlp')]
    ST['on'] = False
    pC = {b: avg_pred(rows, b) for b in SRC}                 # each source's typical continuation
    base = {b: kl(avg_pred(rows), pC[b]) for b in SRC}       # unsteered avg pred vs each p_C
    ST['on'] = False
    mat = {}                                                 # inject B -> KL to each p_C
    for b in SRC:
        for (w, L) in [(w, L) for L in FRONT for w in ('attn', 'mlp')]:
            key = (w, L)
            if (b, w, L) in tmeanbytok:
                dev = tmeanbytok[(b, w, L)] - gm[key]; U = subs[key]; ST['delta'][key] = (dev @ U) @ U.T
            else: ST['delta'][key] = torch.zeros(1, D, device=DEV)
        ST['on'] = True; sp = avg_pred(rows); ST['on'] = False
        mat[b] = {c: round(kl(sp, pC[c]) - base[c], 4) for c in SRC}   # DROP in KL toward each class (negative = moved toward)
    for h in hooks: h.remove()
    # diagonal (toward injected) vs off-diagonal drops
    diag = [mat[b][b] for b in SRC]; off = [mat[b][c] for b in SRC for c in SRC if c != b]
    md = float(np.mean(diag)); mo = float(np.mean(off))
    out = {'alpha': ALPHA, 'sources': SRC, 'kl_drop_matrix': mat, 'base_kl': {str(b): round(base[b], 4) for b in SRC},
           'mean_diagonal_drop': round(md, 4), 'mean_offdiag_drop': round(mo, 4),
           'pred_a_class_specific': bool(md < mo - 0.1 and md < 0), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    for b in SRC: print(f'inject {b}: KL-drop toward each class {mat[b]} (diagonal={mat[b][b]})', flush=True)
    print(f'\nmean diagonal (toward injected) KL-drop {md:+.3f} | mean off-diagonal {mo:+.3f}', flush=True)
    print(f'(a) steering is class-SPECIFIC (diagonal drop >> off-diagonal): {out["pred_a_class_specific"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
