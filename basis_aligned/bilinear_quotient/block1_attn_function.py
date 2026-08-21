"""BLOCK1.ATTN FUNCTION (correct the 701/726 name). block1.attn's rank-1
core has a big benefit (2.06 nats) and FIRES on newline, but 726 showed it
is NOT boundary-selective -- ablating it hurts non-boundary MORE. So what
does it ACTUALLY contribute to? Ablate its rank-1 write direction and
measure the CE increase by next-token CATEGORY (the 710 framework). The
category it most helps = its real function.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating the rank-1 dir reproduces ~its 2-nat benefit overall;
  (a) FUNCTION: report the per-category dCE. Register the expectation (from
      726: broad, non-boundary) that it most helps the HARD open-vocab
      categories (content/subword) -- i.e. it is a GENERAL continuation
      writer, not a boundary circuit;
  (b) report dCE by category + top category;
  NULL: ablating a RANDOM same-norm direction in block1.attn output is
      category-flat (much smaller, no strong top category)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'block1_attn_function_results.json'
NFIT = 64; NEVAL = 96
FUNC = {'the','a','an','of','to','in','and','is','was','are','for','with','that','as','on',
        'at','by','it','be','or','this','from','his','her','their','they','he','she','i','you',
        'we','but','not','have','has','had','were','been','which','who','will','would'}
CATS = ['newline','punct','digit','cap_word','func_word','content','subword']
ABL = {'d': None}


def categorize(t):
    if '\n' in t: return 'newline'
    s = t.strip()
    if s == '': return 'punct'
    if not t.startswith(' ') and not t.startswith('\n'):
        if s[0].isdigit(): return 'digit'
        if not s[0].isalnum(): return 'punct'
        return 'subword'
    core = t.lstrip()
    if core == '': return 'punct'
    if core[0].isdigit(): return 'digit'
    if not core[0].isalnum(): return 'punct'
    if core[0].isupper(): return 'cap_word'
    if core.lower() in FUNC: return 'func_word'
    return 'content'


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


def hook(mo, i_, o_):
    if ABL['d'] is None: return o_
    of = o_.float(); return (of - (of @ ABL['d'])[..., None] * ABL['d']).to(o_.dtype)


@torch.no_grad()
def capture_in(mod, rows, n):
    cap = []
    h = mod.register_forward_pre_hook(lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, D).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


@torch.no_grad()
def per_tok_ce(rows, n):
    ce = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    return torch.cat(ce).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT); ev = cl.fineweb_rows(NEVAL)
    cats = [categorize(cl.d1(int(t))) for r in range(NEVAL) for t in ev[r, 1:257].tolist()][:NEVAL*256]
    cats = np.array(cats)

    b1 = m.transformer.h[1].attn.c_proj
    X = capture_in(b1, rows, NFIT).to(DEV)
    A, _ = asvd_fast(b1.weight.data.float().to(DEV), X)
    d1dir = (A[:, 0] / A[:, 0].norm())

    h = b1.register_forward_hook(hook)
    ABL['d'] = None; base = per_tok_ce(ev, NEVAL)
    ABL['d'] = d1dir; abl = per_tok_ce(ev, NEVAL)
    g = torch.Generator().manual_seed(0); rd = torch.randn(D, generator=g); rd = (rd/rd.norm()).to(DEV)
    ABL['d'] = rd; ablr = per_tok_ce(ev, NEVAL); ABL['d'] = None
    h.remove()

    dc = abl - base; dcr = ablr - base
    res = {}; resr = {}
    for c in CATS:
        sel = cats == c
        res[c] = round(float(dc[sel].mean()), 3); resr[c] = round(float(dcr[sel].mean()), 3)
    top = max(CATS, key=lambda c: res[c])
    print(f'block1.attn rank-1 ablation dCE by category:', flush=True)
    for c in CATS:
        print(f'  {c:9s}: real {res[c]:+.3f}   random {resr[c]:+.3f}', flush=True)
    print(f'top category (real): {top} ({res[top]:+.3f})', flush=True)

    pa = top in ('content', 'subword')
    null_flat = max(abs(v) for v in resr.values()) < 0.3 * res[top]
    print(f'\n(a) general open-vocab (top in content/subword): {pa}', flush=True)
    print(f'NULL random flat: {null_flat}', flush=True)
    out = {'dCE_by_cat': res, 'random_dCE_by_cat': resr, 'top_category': top,
           'pred_a_openvocab': bool(pa), 'null_flat': bool(null_flat), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
