"""CAUSAL VERIFICATION of the folded head names (791): attn1 = DETERMINER head,
attn5 = POSITIONAL head. The fold predicts attn1's benefit concentrates on
determiner queries and attn5's on position (class-flat). Verify by ablating each
head's output and measuring the CE increase RESOLVED by the current token's
grammatical class (attn1) and by position bin (attn5).

REGISTERED PREDICTIONS:
  (0) SANITY: ablating attn1 / attn5 raises CE;
  (a) attn1 = DETERMINER head: the CE increase from ablating attn1 is LARGEST at
      DETERMINER query positions (>= 1.5x the all-token mean, and the top class),
      confirming its benefit is determiner-concentrated (the fold's class-attention
      is causal);
  (b) attn5 = POSITIONAL head: attn5's CE increase is FLAT across grammatical classes
      (max-class / mean < 1.5, unlike attn1) but VARIES by position -- consistent with
      a positional, not class, computation;
  NULL: a random head's CE increase is flat across both class and position (control:
      report attn1's class-profile vs attn5's)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_verify_results.json'
NEVAL = 96
ABL = {'L': -1}
CLASSES = {
    'det': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your'},
    'num': {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'one', 'two', 'three', 'four', 'five', 'ten'},
    'punct': {'.', ',', '!', '?', ';', ':', '(', ')', '"', "'", '--', '-'},
    'pron': {'it', 'he', 'she', 'they', 'we', 'you', 'i', 'him', 'them'},
    'prep': {'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from'},
    'aux': {'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'will', 'would', 'can'},
    'conj': {'and', 'or', 'but', 'if', 'when', 'so', 'because'},
    'other': set(),
}


def hook_factory(L):
    def h(mo, i_, o_):
        if ABL['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
        return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def cls(tokid):
    try: w = cl.d1(int(tokid)).strip().lower()
    except Exception: return 'other'
    for c, mem in CLASSES.items():
        if w in mem: return c
    return 'other'


@torch.no_grad()
def per_token_nll(rows, n):
    nlls = []; toks = []; poss = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        nll = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none')
        nlls.append(nll.cpu()); toks.append(idx.reshape(-1).cpu().numpy())
        poss.append(np.broadcast_to(np.arange(idx.shape[1]), idx.shape).reshape(-1))
    return torch.cat(nlls).numpy(), np.concatenate(toks), np.concatenate(poss)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hooks = [m.transformer.h[L].attn.register_forward_hook(hook_factory(L)) for L in [1, 5]]
    ABL['L'] = -1; nll_full, toks, pos = per_token_nll(rows, NEVAL)
    clslab = np.array([cls(t) for t in toks])

    res = {}
    for L in [1, 5]:
        ABL['L'] = L; nll_abl, _, _ = per_token_nll(rows, NEVAL); ABL['L'] = -1
        d = nll_abl - nll_full                                    # per-token CE increase
        mean_d = float(d.mean())
        by_class = {c: round(float(d[clslab == c].mean()), 4) for c in CLASSES if (clslab == c).sum() >= 20}
        # position profile: mean d in 8 position bins
        nb = 8; edges = np.linspace(0, pos.max()+1, nb+1)
        by_pos = [round(float(d[(pos >= edges[b]) & (pos < edges[b+1])].mean()), 4) for b in range(nb)]
        top_class = max(by_class, key=by_class.get)
        class_ratio = by_class[top_class]/max(mean_d, 1e-9)
        pos_ratio = max(by_pos)/max(np.mean(by_pos), 1e-9)
        res[str(L)] = {'mean_dCE': round(mean_d, 4), 'by_class': by_class, 'top_class': top_class,
                       'class_concentration': round(class_ratio, 2), 'by_position_bin': by_pos, 'pos_concentration': round(float(pos_ratio), 2)}
        print(f'attn{L}: mean dCE {mean_d:.3f} | top class {top_class} (x{class_ratio:.1f} mean) | by-class {by_class}', flush=True)
        print(f'        by-position-bin {by_pos} (pos concentration x{pos_ratio:.1f})', flush=True)
    for h in hooks: h.remove()

    a1 = res['1']; a5 = res['5']
    pa = a1['top_class'] == 'det' and a1['class_concentration'] >= 1.5
    pb = a5['class_concentration'] < 1.5 and a5['pos_concentration'] >= 1.5
    out = {'results': res, 'pred_a_attn1_determiner': bool(pa), 'pred_b_attn5_positional': bool(pb), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) attn1 = determiner head (CE-benefit determiner-concentrated): {pa}; (b) attn5 = positional (class-flat, position-varying): {pb}', flush=True)
    print(f'wrote {OUT} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
