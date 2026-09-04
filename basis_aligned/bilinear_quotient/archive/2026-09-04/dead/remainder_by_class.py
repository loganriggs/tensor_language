"""WHERE THE ~22% REMAINDER LIVES (follows 794: whole model is ~78% class+position;
what is the other ~22%?). With ALL 36 components projected onto class+position, resolve
the CE-recovery by the CURRENT token's grammatical class. If the model is near-100%
class+position on FUNCTION words (determiner/punct/prep/aux) but much lower on content
/ 'other' tokens (nouns, verbs, rare words), the distributed remainder is specifically
CONTENT prediction -- the model is a class+position machine for grammar and a
distributed machine for content.

Runs on 128 rows.

REGISTERED PREDICTIONS:
  (0) SANITY: overall keep-recovery ~ 0.78 (reproduce 794);
  (a) REMAINDER IS CONTENT: class+position CE-recovery is HIGH (>= 0.85) on function-
      word classes (det/punct/prep/aux/conj) and LOWER on 'other' (content) tokens
      (gap >= 0.15) -- so the ~22% distributed remainder is concentrated on predicting
      after content words, not function words;
  (b) report keep-recovery per current-token-class;
  NULL: n/a (descriptive)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; NL = 18
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'remainder_by_class_results.json'
NEVAL = 128; MINCOUNT = 5; RTOK = 64; RPOS = 32
SUBS = {}; MODE = {'op': None}
CLASSES = {
    'det': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your'},
    'punct': {'.', ',', '!', '?', ';', ':', '(', ')', '"', "'", '--', '-'},
    'prep': {'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from', 'into', 'about'},
    'aux': {'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'will', 'would', 'can'},
    'conj': {'and', 'or', 'but', 'if', 'when', 'so', 'because'},
    'pron': {'it', 'he', 'she', 'they', 'we', 'you', 'i', 'him', 'them'},
    'num': {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'one', 'two', 'three'},
    'other': set(),
}


def comp(which, L): return m.transformer.h[L].mlp if which == 'mlp' else m.transformer.h[L].attn


def hook_factory(which, L):
    key = (which, L)
    def h(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = SUBS[key]; v2 = (v @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def per_token_nll(rows, n):
    nlls = []; toks = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        nlls.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
        toks.append(idx.reshape(-1).cpu().numpy())
    return torch.cat(nlls).numpy(), np.concatenate(toks)


@torch.no_grad()
def capture_out(rows, n, which, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(which, L).register_forward_hook(h)
    for i in range(0, n, 4):
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


def cls(tokid):
    try: w = cl.d1(int(tokid)).strip().lower()
    except Exception: return 'other'
    for c, mem in CLASSES.items():
        if w in mem: return c
    return 'other'


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    MODE['op'] = None
    for L in range(NL):
        for w in ('attn', 'mlp'):
            O, toks, pos = capture_out(rows, NEVAL, w, L)
            Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
            SUBS[(w, L)] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    hooks = [comp(w, L).register_forward_hook(hook_factory(w, L)) for L in range(NL) for w in ('attn', 'mlp')]

    MODE['op'] = None; nf, toks = per_token_nll(rows, NEVAL)
    MODE['op'] = 'ablate'; na, _ = per_token_nll(rows, NEVAL)
    MODE['op'] = 'keep'; nk, _ = per_token_nll(rows, NEVAL); MODE['op'] = None
    for h in hooks: h.remove()
    clslab = np.array([cls(t) for t in toks])

    overall = float((na.mean()-nk.mean())/max(na.mean()-nf.mean(), 1e-9))
    by_class = {}
    for c in CLASSES:
        mk = clslab == c
        if mk.sum() < 30: continue
        num = na[mk].mean()-nk[mk].mean(); den = na[mk].mean()-nf[mk].mean()
        by_class[c] = round(float(num/max(den, 1e-9)), 4)
    print(f'overall keep-recovery {overall:.3f} (reproduce 794 ~0.78)', flush=True)
    print(f'by current-token-class: {by_class}', flush=True)
    func = [c for c in ['det', 'punct', 'prep', 'aux', 'conj'] if c in by_class]
    func_mean = float(np.mean([by_class[c] for c in func])) if func else 0.0
    other = by_class.get('other', 0.0)
    print(f'function-word classes mean {func_mean:.3f}  vs  other(content) {other:.3f}  gap {func_mean-other:.3f}', flush=True)

    p0 = abs(overall-0.78) < 0.15
    pa = func_mean >= 0.85 and (func_mean - other) >= 0.15
    out = {'overall_recovery': round(overall, 4), 'by_class': by_class, 'function_mean': round(func_mean, 4),
           'other_content': round(other, 4), 'func_minus_content': round(func_mean-other, 4),
           'pred_0': bool(p0), 'pred_a_remainder_is_content': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) the ~22% remainder is CONTENT (function-words high, content lower): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
