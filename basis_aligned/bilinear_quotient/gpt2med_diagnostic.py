"""GPT2-MEDIUM mlp0 DIAGNOSTIC (settle 801: is the negative keep -0.14 a MASSIVE-
ACTIVATION confound or genuine?). gpt2-medium's low whole-model class+position score
(0.12) is driven entirely by mlp0 (82% of benefit) having keep -0.14 -- keeping only
its token-class+position projection is WORSE than ablating it, the signature of a
massive-activation direction dominating the token-mean subspace. Diagnose mlp0 alone:
  - is the token-mean subspace dominated by a huge-singular-value direction (massive
    activation)? report the norm/singular-value concentration.
  - keep-only recovery at ranks 16/64/128/256, AND with the top-1 direction of the
    token-mean subspace REMOVED (de-massive). If keep goes POSITIVE with more dims or
    with the top direction removed, the -0.14 is a MASSIVE-ACTIVATION CONFOUND (mis-
    measurement), not genuine.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating gpt2-medium mlp0 raises CE (benefit > 0, ~4.8);
  (a) CONFOUND CONFIRMED: removing the top-1 token-mean direction (or using rank 256)
      makes keep POSITIVE and substantial (>= 0.5), so the -0.14 was a massive-
      activation artifact -- gpt2-medium mlp0 IS class+position once the massive
      direction is handled;
  ALT: if keep stays <= 0 at all ranks and de-massive, gpt2-medium mlp0 is genuinely
      NOT class+position (a real scale/family difference)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gpt2med_diagnostic_results.json'
MODEL = 'gpt2-medium'; SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RS = [16, 64, 128, 256]
HK = {'U': None, 'op': None, 'd': None}


def hook(mo, i_, o_):
    if HK['op'] is None: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; d = HK['d']; sh = y.shape; v = y.reshape(-1, d).float()
    if HK['op'] == 'ablate': v2 = torch.zeros_like(v)
    else: U = HK['U']; v2 = (v @ U) @ U.T
    yn = v2.reshape(sh).to(y.dtype)
    return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn


def get_text(nc=2000, npass=48):
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
    t = []
    for i, ex in enumerate(ds):
        t.append(ex['text'][:nc])
        if i >= npass: break
    return '\n\n'.join(t)


@torch.no_grad()
def ce(mdl, blocks):
    tot = 0.0; nn = 0
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i+4].to(DEV); tot += float(mdl(b, labels=b).loss)*b.shape[0]; nn += b.shape[0]
    return tot/nn


@torch.no_grad()
def capture(mdl, mod, blocks, d):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, d))
    hh = mod.register_forward_hook(h)
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i+4].to(DEV); mdl(b); toks.append(b.reshape(-1).cpu().numpy())
        pos.append(np.broadcast_to(np.arange(b.shape[1]), b.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    return _, S                                          # return Vh (dirs), singular values


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL); mdl = AutoModelForCausalLM.from_pretrained(MODEL).to(DEV).eval()
    d = mdl.config.n_embd; HK['d'] = d
    mod = mdl.transformer.h[0].mlp
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    O, toks, pos = capture(mdl, mod, blocks, d)
    # massive-activation check: norm concentration of mlp0 output
    on = O.norm(dim=1); print(f'mlp0 output norm: mean {on.mean():.1f} max {on.max():.1f} (max/mean {on.max()/on.mean():.1f})', flush=True)
    Vtok, Stok = mean_subspace(O, toks, 256)
    print(f'token-mean subspace top singular values: {[round(float(x),1) for x in Stok[:5]]} (s0/s1 {float(Stok[0]/Stok[1]):.1f})', flush=True)

    h = mod.register_forward_hook(hook)
    HK['op'] = None; ce_full = ce(mdl, blocks)
    HK['op'] = 'ablate'; ce_abl = ce(mdl, blocks); ben = ce_abl - ce_full
    print(f'mlp0 benefit {ben:.3f}', flush=True)
    Vpos = mean_subspace(O, pos, 32)[0]

    def keeprec(U): HK['op'] = 'keep'; HK['U'] = U; c = ce(mdl, blocks); HK['op'] = None; HK['U'] = None; return float((ce_abl-c)/max(ben, 1e-6))
    res = {}
    for r in RS:
        Ucp = torch.linalg.svd(torch.cat([Vtok[:r].T, Vpos.T], 1), full_matrices=False)[0][:, :r+32].contiguous()
        res[f'keep_r{r}'] = round(keeprec(Ucp), 4)
    # de-massive: token subspace with top-1 removed
    Ucp_nomass = torch.linalg.svd(torch.cat([Vtok[1:65].T, Vpos.T], 1), full_matrices=False)[0][:, :96].contiguous()
    res['keep_64tok_noTopPC'] = round(keeprec(Ucp_nomass), 4)
    # keep only the top-1 direction (is it harmful alone?)
    res['keep_top1_only'] = round(keeprec(Vtok[:1].T.contiguous()), 4)
    h.remove()
    print(f'keep-only recovery: {res}', flush=True)

    pos_when_fixed = max(res['keep_r256'], res['keep_64tok_noTopPC'])
    pa = pos_when_fixed >= 0.5
    out = {'model': MODEL, 'mlp0_benefit': round(ben, 4), 'norm_max_over_mean': round(float(on.max()/on.mean()), 2),
           's0_over_s1': round(float(Stok[0]/Stok[1]), 2), 'recovery': res,
           'pred_a_confound_confirmed': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) gpt2-medium mlp0 negative-keep is a MASSIVE-ACTIVATION CONFOUND (fixed by more dims / de-massive -> keep>=0.5): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
