"""CORRECTED (MEAN-PRESERVING) CROSS-MODEL PER-COMPONENT SCOREBOARD — re-run the §800
whole-model class+position claim with the metric fix from §805/806. The original §800
scoreboard used CENTERED keep (v2 = proj_U(v)), which silently discards each component's
constant MEAN bias; §806 showed large constant biases are common in early layers, so the
§800 nat-weighted headline numbers (bilin18 0.78, gpt2-small 0.77, gpt2-large 0.75,
pythia-160m 0.75, pythia-410m 0.69) are slight UNDERESTIMATES. This re-scores EVERY
component (attention + MLP, all layers) of all five HF models with mean-preserving keep
(v2 = mean + proj_U(v - mean)) and reports the corrected nat-weighted class+position share
alongside the old centered number and the random-subspace null.

REGISTERED PREDICTIONS:
  (0) SANITY: ablate-all raises CE; random-subspace nat-weighted keep stays low;
  (a) CORRECTED >= CENTERED for every model (mean-preserving never lowers the headline);
  (b) the corrected nat-weighted class+position share is >= the §800 value for all five,
      and the ordering (all ~0.7-0.85) is preserved -- the correction raises the floor,
      does not overturn the "common across all six models" conclusion;
  (c) report per-model centered vs mean-preserving vs random."""
import json, time, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_scoreboard_mp_results.json'
MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RTOK = 64; RPOS = 32
SUBS = {}; MEANS = {}; MODE = {'op': None, 'd': None, 'rand': None, 'single': None}


def components(mdl):
    out = []
    if hasattr(mdl, 'transformer'):
        for L, blk in enumerate(mdl.transformer.h):
            out.append((f'attn{L}', blk.attn)); out.append((f'mlp{L}', blk.mlp))
    elif hasattr(mdl, 'gpt_neox'):
        for L, blk in enumerate(mdl.gpt_neox.layers):
            out.append((f'attn{L}', blk.attention)); out.append((f'mlp{L}', blk.mlp))
    return out


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


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd
    MODE['d'] = d
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    comps = components(mdl); SUBS.clear(); MEANS.clear()
    MODE['op'] = None
    for name, mod in comps:
        O, toks, pos = capture(mdl, mod, blocks, d)
        MEANS[name] = O.mean(0, keepdim=True)
        Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
        SUBS[name] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(d, RTOK+RPOS, generator=g, device=DEV))[0]

    def sh(name):  # hook active ONLY for the single named component
        def hh(mo, i_, o_):
            if MODE['op'] is None or MODE['single'] != name: return o_
            y = o_[0] if isinstance(o_, tuple) else o_; sh_ = y.shape; v = y.reshape(-1, d).float()
            if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
            elif MODE['op'] == 'keeprand': U = MODE['rand']; v2 = (v @ U) @ U.T
            elif MODE['op'] == 'keep': U = SUBS[name]; v2 = (v @ U) @ U.T
            else: U = SUBS[name]; mu = MEANS[name]; v2 = mu + ((v - mu) @ U) @ U.T   # keepmean
            yn = v2.reshape(sh_).to(y.dtype); return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
        return hh
    hooks = [mod.register_forward_hook(sh(name)) for name, mod in comps]
    MODE['op'] = None; MODE['single'] = None; ce_full = ce(mdl, blocks)
    per = {}; tb = 0.0; tk = 0.0; tkm = 0.0; tr = 0.0
    for name, mod in comps:
        MODE['single'] = name
        MODE['op'] = 'ablate'; ca = ce(mdl, blocks); ben = ca - ce_full
        MODE['op'] = 'keep'; ck = ce(mdl, blocks)
        MODE['op'] = 'keepmean'; ckm = ce(mdl, blocks)
        MODE['op'] = 'keeprand'; cr = ce(mdl, blocks); MODE['op'] = None
        rec = float((ca-ck)/max(ben, 1e-6)); recm = float((ca-ckm)/max(ben, 1e-6)); recr = float((ca-cr)/max(ben, 1e-6))
        per[name] = {'benefit': round(ben, 3), 'keep_cent': round(rec, 3), 'keep_mean': round(recm, 3), 'rand': round(recr, 3)}
        if ben > 0:
            tb += ben; tk += ben*max(min(rec, 1), 0); tkm += ben*max(min(recm, 1), 0); tr += ben*max(min(recr, 1), 0)
    MODE['single'] = None
    for h in hooks: h.remove()
    del mdl; torch.cuda.empty_cache()
    return {'d': d, 'n_components': len(comps), 'total_benefit': round(tb, 3),
            'nw_centered': round(tk/max(tb, 1e-9), 4), 'nw_meanpreserve': round(tkm/max(tb, 1e-9), 4),
            'nw_random': round(tr/max(tb, 1e-9), 4), 'per_component': per}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        try:
            r = run(mid); out[mid] = r
            print(f'{mid} ({r["n_components"]} comp): NW class+pos CENTERED {r["nw_centered"]} -> MEAN-PRESERVE {r["nw_meanpreserve"]} | random {r["nw_random"]}', flush=True)
        except Exception as e:
            out[mid] = {'error': str(e)[:200]}; print(f'{mid} FAILED: {e}', flush=True)
    ok = [m for m in MODELS if 'nw_meanpreserve' in out.get(m, {})]
    pa = all(out[m]['nw_meanpreserve'] >= out[m]['nw_centered'] - 1e-6 for m in ok)
    out['pred_a_meanpreserve_ge_centered'] = bool(pa); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) mean-preserving >= centered for all models: {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
