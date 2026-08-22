"""WHAT DOES attn5 HELP PREDICT? (effect-based lens; §852 — input-decodes all failed). attn5 (1.97 nats)
is not captured by any grammatical/token input-probe, so characterize it by its EFFECT: ablate attn5 and
measure WHERE the loss increases — which TARGET grammatical class it helps predict, and which specific
target tokens degrade most. Content-word (word) damage => attn5 helps content/semantic prediction;
function-word/punct damage => grammatical; a few specific tokens => a targeted signal.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating attn5 raises total CE ~1.97 nats;
  (a) report per-target-class CE increase (normalized by class frequency) — which next-token class does
      attn5 most help? and the top specific target tokens whose CE rises most;
  (b) if content 'word' dominates -> semantic/content role; if punct/function -> grammatical; if a small
      token set -> a targeted predictor."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn5_effect_results.json'
NEVAL = 200
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
ABL = {'on': False}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def hook(mo, i_, o_):
    if not ABL['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
    return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def per_token_nll(rows):
    nlls = []; tgts = []
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        nll = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.numel(), device=DEV), tgt.reshape(-1)]
        nlls.append(nll.cpu().numpy()); tgts.append(tgt.reshape(-1).cpu().numpy())
    return np.concatenate(nlls), np.concatenate(tgts)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    h = m.transformer.h[5].attn.register_forward_hook(hook)
    ABL['on'] = False; nll0, tgt = per_token_nll(rows)
    ABL['on'] = True; nll1, _ = per_token_nll(rows)
    h.remove()
    delta = nll1 - nll0                                    # CE increase per position when attn5 ablated
    tgtcls = np.array([CLASSES.index(classify(d(int(t)))) for t in tgt])
    per_class = {}
    for c in range(len(CLASSES)):
        mk = tgtcls == c
        if mk.sum() > 20: per_class[CLASSES[c]] = {'mean_delta': round(float(delta[mk].mean()), 4), 'frac_of_total': round(float(delta[mk].sum()/delta.sum()), 3), 'n': int(mk.sum())}
    # top specific target tokens by total delta
    order = {}
    for t in np.unique(tgt):
        mk = tgt == t
        if mk.sum() >= 5: order[int(t)] = float(delta[mk].sum())
    top = sorted(order.items(), key=lambda kv: -kv[1])[:15]
    out = {'total_ce_increase': round(float(delta.mean()), 4), 'per_target_class': per_class,
           'top_helped_target_tokens': [(repr(d(t)), round(v, 2)) for t, v in top], 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"attn5 ablation total CE increase {out['total_ce_increase']}", flush=True)
    print("per-target-class mean CE increase (what attn5 helps predict):", flush=True)
    for c, r in sorted(per_class.items(), key=lambda kv: -kv[1]['mean_delta']): print(f"  {c}: mean +{r['mean_delta']} (frac {r['frac_of_total']}, n {r['n']})", flush=True)
    print("top helped target tokens:", out['top_helped_target_tokens'][:10], flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
