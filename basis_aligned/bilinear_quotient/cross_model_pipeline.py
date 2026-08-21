"""IS THE compute->maintain->read PIPELINE UNIVERSAL? (§814 cross-model extension). bilin18's
whole stack is: FRONT computes class+position (collective 6.6 nats, 0.93 share), MIDDLE
maintains it redundantly (1.9 nats, 0.65, super-additive), BACK reads it out (4.6 nats, 0.93).
Test whether GPT-2 and Pythia show the same three-band structure. For each model, split the
layers into thirds (front/middle/back) and measure, per band: (i) collective ablation cost
(zero all the band's components at once) and the per-component sum, to detect redundancy
(super-additive middle); (ii) mean-preserving keep-only class+position on the whole band, to
see if each band is class+position and whether the middle is 'maintenance'.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating a band raises CE; random simultaneous projection is far worse than
      class+position keep;
  (a) UNIVERSAL: front and back bands are high class+position (>=0.7) with high collective
      benefit; the middle band is SUPER-ADDITIVE (collective >> per-component sum) and still
      substantially class+position (>=0.5) -> compute/maintain/read is a shared shape;
  (b) if a model breaks the pattern (e.g. no redundant middle, or middle not class+position),
      report it plainly -- the pipeline may be bilin18-specific or architecture-dependent;
  (c) report per-band collective cost, per-component sum, compounding ratio, class+position
      keep, random keep."""
import json, time, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_pipeline_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RTOK = 64; RPOS = 32
SUBS = {}; MEANS = {}; MODE = {'op': None, 'd': None, 'rand': None, 'active': set()}


def components(mdl):
    out = []
    if hasattr(mdl, 'transformer'):
        for L, blk in enumerate(mdl.transformer.h): out.append((f'attn{L}', blk.attn)); out.append((f'mlp{L}', blk.mlp))
    elif hasattr(mdl, 'gpt_neox'):
        for L, blk in enumerate(mdl.gpt_neox.layers): out.append((f'attn{L}', blk.attention)); out.append((f'mlp{L}', blk.mlp))
    return out


def mk_hook(name):
    def hook(mo, i_, o_):
        if MODE['op'] is None or name not in MODE['active']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; d = MODE['d']; sh = y.shape; v = y.reshape(-1, d).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        elif MODE['op'] == 'keeprand': U = MODE['rand']; v2 = (v @ U) @ U.T
        else: U = SUBS[name]; mu = MEANS[name]; v2 = mu + ((v - mu) @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


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


def mean_subspace(O, labels, r, gmean):
    rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - gmean[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd; MODE['d'] = d
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    comps = components(mdl); nL = len(comps)//2
    MODE['op'] = None; SUBS.clear(); MEANS.clear()
    for name, mod in comps:
        O, toks, pos = capture(mdl, mod, blocks, d)
        MEANS[name] = O.mean(0, keepdim=True)
        Utok = mean_subspace(O, toks, RTOK, MEANS[name]); Upos = mean_subspace(O, pos, RPOS, MEANS[name])
        SUBS[name] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(d, RTOK+RPOS, generator=g, device=DEV))[0]
    hooks = [mod.register_forward_hook(mk_hook(name)) for name, mod in comps]
    b0, b1 = nL//3, 2*nL//3
    bands = {'front': list(range(0, b0)), 'middle': list(range(b0, b1)), 'back': list(range(b1, nL))}
    def names_for(layers): return {f'{w}{L}' for L in layers for w in ('attn', 'mlp')}
    MODE['op'] = None; MODE['active'] = set(); ce_full = ce(mdl, blocks)
    res = {'d': d, 'n_layers': nL, 'bands': {k: [min(v), max(v)] for k, v in bands.items()}, 'ce_full': round(ce_full, 4)}
    for bname, layers in bands.items():
        nm = names_for(layers)
        # per-component benefit sum
        psum = 0.0
        for one in nm:
            MODE['op'] = 'ablate'; MODE['active'] = {one}; psum += ce(mdl, blocks) - ce_full
        MODE['active'] = nm
        MODE['op'] = 'ablate'; ce_abl = ce(mdl, blocks); ben = ce_abl - ce_full
        MODE['op'] = 'keep'; ce_keep = ce(mdl, blocks)
        MODE['op'] = 'keeprand'; ce_rand = ce(mdl, blocks); MODE['op'] = None
        res[bname] = {'collective_benefit': round(ben, 3), 'per_component_sum': round(psum, 3),
                      'compounding': round(ben/max(psum, 1e-6), 2),
                      'keep_classpos': round(float((ce_abl-ce_keep)/max(ben, 1e-6)), 3),
                      'keep_random': round(float((ce_abl-ce_rand)/max(ben, 1e-6)), 3)}
    MODE['active'] = set()
    for h in hooks: h.remove()
    del mdl; torch.cuda.empty_cache(); return res


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        try:
            r = run(mid); out[mid] = r
            for b in ('front', 'middle', 'back'):
                bb = r[b]; print(f'{mid} {b} (L{r["bands"][b][0]}-{r["bands"][b][1]}): collective {bb["collective_benefit"]} sum {bb["per_component_sum"]} (x{bb["compounding"]}) | class+pos {bb["keep_classpos"]} rand {bb["keep_random"]}', flush=True)
        except Exception as e:
            out[mid] = {'error': str(e)[:200]}; print(f'{mid} FAILED: {e}', flush=True)
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
