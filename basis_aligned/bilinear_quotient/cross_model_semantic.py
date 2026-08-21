"""CROSS-MODEL token-class SUBSPACE (does bilin18's 767-770 finding generalise to
REAL pretrained models? user ask). For GPT-2 (small) and Pythia-410m, replicate the
token-conditional-mean subspace analysis on an early MLP's output: is a low-rank
token-class subspace NECESSARY (remove -> CE up vs random) and SUFFICIENT (keep-only
top-r -> recover most of the layer's loss benefit), like bilin18 (keep-only-64 = 0.92)?
If yes, the "front sorts what it writes by token-class in a canonical low-rank
subspace" result is a GENERAL property of transformers, not a bilin18 quirk.

Concatenate + chunk FineWeb text (no padding); hook each model's MLP output at an
early layer; token-conditional-mean subspace; sweep keep-only r vs random.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating the MLP output raises CE (benefit > 0);
  (a) GENERALISES: for BOTH GPT-2 and Pythia, keep-only the top-64 token-class
      subspace recovers >= 0.6 of the early MLP's CE benefit, and >> a random 64-dim
      subspace -- the canonical low-rank token-class structure is general;
  (b) report keep-only recovery vs r (16/64/128) per model, with random baseline;
  NULL: random same-rank subspace recovers far less."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_semantic_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RS = [16, 64, 128]
HK = {'U': None, 'op': None, 'd': None}


def get_mlp(mdl, L):
    if hasattr(mdl, 'transformer'): return mdl.transformer.h[L].mlp       # gpt2
    if hasattr(mdl, 'gpt_neox'): return mdl.gpt_neox.layers[L].mlp        # pythia
    raise ValueError('unknown arch')


def hook(mo, i_, o_):
    if HK['op'] is None: return o_
    out = o_[0] if isinstance(o_, tuple) else o_
    d = HK['d']; sh = out.shape; v = out.reshape(-1, d).float()
    if HK['op'] == 'ablate': v2 = torch.zeros_like(v)
    else:
        U = HK['U']; v2 = (v @ U) @ U.T if HK['op'] == 'keep' else v - (v @ U) @ U.T
    vn = v2.reshape(sh).to(out.dtype)
    return (vn,) + tuple(o_[1:]) if isinstance(o_, tuple) else vn


def get_text(n_chars=2000, npass=40):
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
    txt = []
    for i, ex in enumerate(ds):
        txt.append(ex['text'][:n_chars])
        if i >= npass: break
    return '\n\n'.join(txt)


def make_blocks(tok, text):
    ids = tok(text, return_tensors='pt')['input_ids'][0]
    nb = min(NBLOCK, ids.shape[0] // SEQ)
    return ids[:nb*SEQ].reshape(nb, SEQ)


@torch.no_grad()
def ce(mdl, blocks):
    tot = 0.0; nn = 0
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i+4].to(DEV)
        out = mdl(b, labels=b)
        tot += float(out.loss) * b.shape[0]; nn += b.shape[0]
    return tot / nn


@torch.no_grad()
def capture(mdl, mlp, blocks, d):
    cap = []; toks = []
    h = mlp.register_forward_hook(lambda mo, i_, o_: cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, d)))
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i+4].to(DEV); mdl(b); toks.append(b.reshape(-1).cpu())
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy()


def token_subspace(O, toks, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def analyze_layer(mdl, L, blocks, d):
    mlp = get_mlp(mdl, L); HK['d'] = d
    O, toks = capture(mdl, mlp, blocks, d)
    h = mlp.register_forward_hook(hook)
    HK['op'] = None; ce_full = ce(mdl, blocks)
    HK['op'] = 'ablate'; ce_abl = ce(mdl, blocks); ben = ce_abl - ce_full
    g = torch.Generator(device=DEV).manual_seed(0); res = {}
    for r in RS:
        U = token_subspace(O, toks, r); HK['op'] = 'keep'; HK['U'] = U; ce_keep = ce(mdl, blocks)
        Ur = torch.linalg.qr(torch.randn(d, r, generator=g, device=DEV))[0]; HK['U'] = Ur; ce_rand = ce(mdl, blocks)
        HK['op'] = None; HK['U'] = None
        res[str(r)] = {'keep': round(float((ce_abl-ce_keep)/max(ben, 1e-6)), 4),
                       'keep_rand': round(float((ce_abl-ce_rand)/max(ben, 1e-6)), 4)}
    h.remove()
    return {'layer': L, 'benefit': round(ben, 4), 'recovery': res}


@torch.no_grad()
def scan_benefit(mdl, blocks, nl, d):
    HK['d'] = d; bens = []
    HK['op'] = None
    for L in range(nl):
        mlp = get_mlp(mdl, L); h = mlp.register_forward_hook(hook)
        HK['op'] = 'ablate'; c = ce(mdl, blocks); HK['op'] = None; h.remove()
        bens.append(c)
    base = ce(mdl, blocks)
    return [round(b - base, 4) for b in bens]


def run_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd
    nl = mdl.config.num_hidden_layers if hasattr(mdl.config, 'num_hidden_layers') else mdl.config.n_layer
    blocks = make_blocks(tok, get_text())
    bens = scan_benefit(mdl, blocks, nl, d)
    # analyze layer 0 + the two highest-benefit layers
    top = sorted(range(nl), key=lambda L: -bens[L])[:2]
    layers = sorted(set([0] + top))
    print(f'  {model_id} benefit-by-layer {bens}', flush=True)
    analyses = {str(L): analyze_layer(mdl, L, blocks, d) for L in layers}
    del mdl; torch.cuda.empty_cache()
    return {'d': d, 'n_layers': nl, 'benefit_by_layer': bens, 'analyzed': analyses}


def main():
    t0 = time.time(); out = {}
    for model_id in MODELS:
        try:
            r = run_model(model_id); out[model_id] = r
            for L, a in r['analyzed'].items():
                print(f'{model_id} L{L} (d{r["d"]}): benefit {a["benefit"]} | ' +
                      ' '.join(f'r{k}:{a["recovery"][k]["keep"]:.2f}(rand {a["recovery"][k]["keep_rand"]:.2f})' for k in map(str, RS)), flush=True)
        except Exception as e:
            out[model_id] = {'error': str(e)[:200]}; print(f'{model_id} FAILED: {e}', flush=True)
    # generalises if, at each model's MAX-benefit analyzed layer, keep-64 >= 0.6 and >> random
    ok = [m for m in out if isinstance(out[m], dict) and 'analyzed' in out[m]]
    def best(m):
        a = out[m]['analyzed']; L = max(a, key=lambda L: a[L]['benefit']); return a[L]['recovery']['64']
    pa = len(ok) > 0 and all(best(m)['keep'] >= 0.6 and best(m)['keep'] > 1.5*best(m)['keep_rand'] for m in ok)
    out['pred_a_generalises'] = bool(pa); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) token-class subspace generalises to real models (keep-64>=0.6 & >>random): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
