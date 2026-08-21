"""ARE LARGE CONSTANT (DC) BIASES A COMMON EARLY-LAYER FEATURE? (follow-up to 805). The
gpt2-medium correction revealed the class+position keep metric drops the per-component
MEAN, and that gpt2-medium mlp0's mean is 91% of its output norm. Question: is a big
constant bias special to gpt2-medium, or a common early-component feature we've been
silently discarding? Re-score with MEAN-PRESERVING keep across five HF models, for the
first attention AND first MLP:
  - mean-norm / output-norm (how much of the output is a constant offset);
  - class+position keep CENTERED (old) vs keep +MEAN (corrected).
This gives the corrected early-layer class+position numbers AND a prevalence map of DC
biases.

REGISTERED PREDICTIONS:
  (a) gpt2-medium mlp0 has the largest mlp DC ratio (~0.9); others are smaller;
  (b) keep +MEAN >= keep CENTERED for every component (mean-preserving never hurts);
  (c) where the DC ratio is large, +MEAN raises keep substantially; where small, the
      two are close -> the correction only matters for high-DC components;
  (d) report which components across the five models carry a large (>0.5) constant bias
      -- is it an early-MLP thing, an attention thing, or model-specific?"""
import json, time, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_dc_results.json'
MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5
HK = {'U': None, 'mean': None, 'op': None, 'd': None}


def hook(mo, i_, o_):
    if HK['op'] is None: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; d = HK['d']; sh = y.shape; v = y.reshape(-1, d).float()
    if HK['op'] == 'ablate': v2 = torch.zeros_like(v)
    elif HK['op'] == 'keep': U = HK['U']; v2 = (v @ U) @ U.T
    else: U = HK['U']; mu = HK['mean']; v2 = mu + ((v - mu) @ U) @ U.T
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


def comps(mdl, model_id):
    if 'pythia' in model_id:
        L0 = mdl.gpt_neox.layers[0]; return [('attn0', L0.attention), ('mlp0', L0.mlp)]
    L0 = mdl.transformer.h[0]; return [('attn0', L0.attn), ('mlp0', L0.mlp)]


def score_component(mdl, mod, blocks, d):
    O, toks, pos = capture(mdl, mod, blocks, d)
    gmean = O.mean(0, keepdim=True)
    dc = float(gmean.norm()) / max(float(O.norm(dim=1).mean()), 1e-6)
    Utok = mean_subspace(O, toks, 64); Upos = mean_subspace(O, pos, 32)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :96].contiguous()
    HK['U'] = Ucp; HK['mean'] = gmean
    h = mod.register_forward_hook(hook)
    HK['op'] = None; ce_full = ce(mdl, blocks)
    HK['op'] = 'ablate'; ce_abl = ce(mdl, blocks); ben = ce_abl - ce_full
    HK['op'] = 'keep'; ce_k = ce(mdl, blocks)
    HK['op'] = 'keepmean'; ce_km = ce(mdl, blocks); HK['op'] = None
    h.remove()
    return {'dc_ratio': round(dc, 3), 'benefit': round(ben, 3),
            'keep_centered': round(float((ce_abl-ce_k)/max(ben, 1e-6)), 4),
            'keep_with_mean': round(float((ce_abl-ce_km)/max(ben, 1e-6)), 4)}


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd; HK['d'] = d
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    r = {'d': d}
    for name, mod in comps(mdl, model_id):
        r[name] = score_component(mdl, mod, blocks, d)
    del mdl; torch.cuda.empty_cache(); return r


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        try:
            r = run(mid); out[mid] = r
            s = ' | '.join(f'{n}: dc {r[n]["dc_ratio"]} keep {r[n]["keep_centered"]}->{r[n]["keep_with_mean"]}' for n in ['attn0', 'mlp0'])
            print(f'{mid} (d{r["d"]}): {s}', flush=True)
        except Exception as e:
            out[mid] = {'error': str(e)[:200]}; print(f'{mid} FAILED: {e}', flush=True)
    # prevalence: which components carry a large constant bias
    big = [f'{mid.split("/")[-1]}/{n}' for mid, r in out.items() if 'd' in r for n in ['attn0', 'mlp0'] if r[n]['dc_ratio'] > 0.5]
    out['large_dc_components'] = big; out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nlarge (>0.5) constant-bias components: {big}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
