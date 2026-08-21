"""WHY IS GPT2-MEDIUM's mlp0 DIFFERENT? (the one open question from 802/803). gpt2-
small/large mlp0 reduce to class+position (keep 0.77-0.92); gpt2-medium mlp0 does not
(keep < 0). Diagnose the difference by comparing all three GPT-2 mlp0's on:
  (a) EFFECTIVE RANK of the output (is medium's higher / more distributed?);
  (b) keep-only recovery using the output's OWN top-r principal directions
      (UNSUPERVISED SVD -- is the output low-rank CAUSALLY at all?), r=16/64/128;
  (c) keep-only recovery using the token-mean directions (SUPERVISED class, for
      reference).
This separates two hypotheses: (H1) gpt2-medium mlp0 is genuinely HIGH-RANK/distributed
(own-SVD keep also low) -> it computes something not low-rank; (H2) it is LOW-RANK but
NOT TOKEN-ORGANISED (own-SVD keep HIGH, token-mean keep low) -> its low-rank structure
is content/context, not class.

REGISTERED PREDICTIONS:
  (0) SANITY: gpt2-small/large mlp0 own-SVD keep is high (>=0.7 at r=64);
  (a) DIAGNOSIS: report which hypothesis holds for gpt2-medium -- if own-SVD keep is
      HIGH (>=0.6) but token-mean keep is low/negative, its mlp0 is low-rank but not
      class-organised (H2); if own-SVD keep is ALSO low, it is genuinely high-rank (H1);
  (b) report eff-rank + own-SVD keep(r) + token-mean keep for all three;
  NULL: random same-rank subspace keep is far lower than own-SVD keep."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gpt2_mlp0_compare_results.json'
MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RS = [16, 64, 128]
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
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def eff_rank(X):
    s2 = torch.linalg.svdvals(X - X.mean(0, keepdim=True))**2
    return float((s2.sum()**2)/(s2**2).sum())


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.n_embd; HK['d'] = d; mod = mdl.transformer.h[0].mlp
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    O, toks, pos = capture(mdl, mod, blocks, d)
    er = eff_rank(O)
    Vown = torch.linalg.svd(O - O.mean(0, keepdim=True), full_matrices=False)[2]     # own principal dirs
    Utok = mean_subspace(O, toks, 64); Upos = mean_subspace(O, pos, 32)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :96].contiguous()
    h = mod.register_forward_hook(hook)
    HK['op'] = None; ce_full = ce(mdl, blocks)
    HK['op'] = 'ablate'; ce_abl = ce(mdl, blocks); ben = ce_abl - ce_full
    def keeprec(U): HK['op'] = 'keep'; HK['U'] = U; c = ce(mdl, blocks); HK['op'] = None; HK['U'] = None; return float((ce_abl-c)/max(ben, 1e-6))
    own = {f'r{r}': round(keeprec(Vown[:r].T.contiguous()), 4) for r in RS}
    g = torch.Generator(device=DEV).manual_seed(0); Ur = torch.linalg.qr(torch.randn(d, 64, generator=g, device=DEV))[0]
    own_rand = round(keeprec(Ur), 4)
    tokkeep = round(keeprec(Ucp), 4)
    h.remove(); del mdl; torch.cuda.empty_cache()
    return {'d': d, 'benefit': round(ben, 4), 'eff_rank': round(er, 2), 'own_svd_keep': own,
            'own_rand64_keep': own_rand, 'token_mean_keep': tokkeep}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        try:
            r = run(mid); out[mid] = r
            print(f'{mid} (d{r["d"]}): eff-rank {r["eff_rank"]} | own-SVD keep {r["own_svd_keep"]} (rand64 {r["own_rand64_keep"]}) | token-mean keep {r["token_mean_keep"]}', flush=True)
        except Exception as e:
            out[mid] = {'error': str(e)[:200]}; print(f'{mid} FAILED: {e}', flush=True)
    # diagnosis for gpt2-medium
    m = out.get('gpt2-medium', {})
    if 'own_svd_keep' in m:
        h2 = m['own_svd_keep']['r64'] >= 0.6 and m['token_mean_keep'] < 0.3
        h1 = m['own_svd_keep']['r64'] < 0.6
        out['diagnosis'] = 'H2_low-rank-not-token-organised' if h2 else ('H1_high-rank-distributed' if h1 else 'mixed')
        print(f"\ngpt2-medium diagnosis: {out['diagnosis']} (own-SVD-r64 {m['own_svd_keep']['r64']}, token-mean {m['token_mean_keep']})", flush=True)
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
