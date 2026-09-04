"""MLP17 CORE VERIFY (apply the 726 causal lens to a circuit named by
READOUT, not causal test). 696 named mlp17's rank-4 core directions by
their unembedding readout: dir0 = frequency-calibration (cos 0.878 w_freq),
dir1 = subword writer, dir2 = proper-noun writer, dir3 = topical writer.
Those are FIRING/geometric names. Verify CAUSALLY: ablate each core
direction (project out of mlp17 output) and measure which next-token
CATEGORY + frequency the CE-increase concentrates on. Does the causal
selectivity match the readout name?

REGISTERED PREDICTIONS:
  (0) SANITY: ablating all 4 reproduces much of mlp17's benefit;
  (a) CALIBRATION dir0: ablating it should hurt FREQUENT-token positions
      (high log-freq targets) more than rare -- a frequency effect (its
      readout is the calibration axis). Report frequency-selectivity
      (dCE on frequent vs rare targets) per direction;
  (b) CONTENT dirs (2,3): should hurt content/open-vocab categories more;
      report per-direction category dCE; state whether causal selectivity
      MATCHES the 696 readout names or not (correct if not);
  NULL: a random direction ablated in mlp17 output is category/frequency
      flat."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_core_verify_results.json'
NFIT = 64; NEVAL = 128
FUNC = {'the','a','an','of','to','in','and','is','was','are','for','with','that','as','on',
        'at','by','it','be','or','this','from','he','she','they','you','we','not','have','has'}
CATS = ['punct','digit','cap_word','func_word','content','subword','newline']
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
def capture_gate(rows, n):
    cap = []
    h = m.transformer.h[17].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def per_tok_ce(rows, n):
    ce = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    return torch.cat(ce).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT); ev = cl.fineweb_rows(NEVAL)
    V = m.lm_head.weight.shape[0]
    # target categories + log-freq
    freq = np.zeros(V)
    for r in range(NEVAL):
        for t in ev[r, 1:257].tolist(): freq[t] += 1
    logf = np.log(freq + 1)
    cats = np.array([categorize(cl.d1(int(t))) for r in range(NEVAL) for t in ev[r,1:257].tolist()][:NEVAL*256])
    tgt_lf = np.array([logf[t] for r in range(NEVAL) for t in ev[r,1:257].tolist()][:NEVAL*256])
    freq_hi = tgt_lf > np.median(tgt_lf)

    d17 = m.transformer.h[17].mlp.Down
    X = capture_gate(rows, NFIT).to(DEV)
    A, _ = asvd_fast(d17.weight.data.float().to(DEV), X)
    dirs = [A[:, k] / A[:, k].norm() for k in range(4)]
    g = torch.Generator().manual_seed(0); rd = torch.randn(D, generator=g); rd = (rd/rd.norm()).to(DEV)

    h = m.transformer.h[17].mlp.register_forward_hook(hook)
    ABL['d'] = None; base = per_tok_ce(ev, NEVAL)
    def prof(d):
        ABL['d'] = d; ce = per_tok_ce(ev, NEVAL); ABL['d'] = None; dc = ce - base
        cat = {c: round(float(dc[cats==c].mean()),3) for c in CATS}
        fsel = round(float(dc[freq_hi].mean() - dc[~freq_hi].mean()),3)  # freq - rare
        return cat, fsel
    res = {}
    names = {0:'calibration(696)',1:'subword(696)',2:'proper-noun(696)',3:'topical(696)'}
    for k in range(4):
        cat, fsel = prof(dirs[k])
        top = max(CATS, key=lambda c: cat[c])
        res[f'dir{k}'] = {'name_696': names[k], 'cat_dCE': cat, 'freq_selectivity': fsel, 'causal_top_cat': top}
        print(f'dir{k} [{names[k]:16s}]: top-cat {top}  freq-sel {fsel:+.3f}  cats {cat}', flush=True)
    catr, fselr = prof(rd)
    print(f'\nNULL random: top {max(CATS,key=lambda c:catr[c])}  freq-sel {fselr:+.3f}', flush=True)
    h.remove()

    # verdicts
    cal_ok = res['dir0']['freq_selectivity'] > 0.02   # calibration hurts frequent more
    print(f'\n(a) dir0 calibration = frequency-selective (freq-sel>0.02): {cal_ok}', flush=True)
    out = {'directions': res, 'null_freq_sel': round(fselr,3),
           'null_top_cat': max(CATS,key=lambda c:catr[c]), 'dir0_calibration_confirmed': bool(cal_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
