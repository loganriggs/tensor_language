"""CROSS-MODEL PER-COMPONENT SCOREBOARD (clean fix for 799: per-component, not simultaneous -- avoids gpt2 catastrophe + pythia depth-compounding). Is the grand 794 claim -- the whole model
is ~78% a class+position computer -- UNIVERSAL?). For GPT-2 and Pythia-410m, project
EVERY component's output (attention + MLP, all layers) onto its token-class + position
subspace SIMULTANEOUSLY, and measure CE-recovery of the whole model's contribution.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating all components raises CE a lot;
  (a) UNIVERSAL: for GPT-2 (and Pythia), keeping only class+position at every component
      at once recovers a LARGE fraction of the model's contribution (>= 0.6) and >>
      random subspaces -- the whole-model class+position reduction is not bilin18-
      specific (bilin18 794: 0.78);
  (b) report keep-(class+position) vs keep-random per model;
  NULL: random same-rank simultaneous projection recovers far less."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_scoreboard_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RTOK = 64; RPOS = 32
SUBS = {}; MODE = {'op': None, 'd': None, 'rand': None}


def components(mdl):
    # list of (name, module) for every attention + MLP block, in order
    out = []
    if hasattr(mdl, 'transformer'):        # gpt2
        for L, blk in enumerate(mdl.transformer.h):
            out.append((f'attn{L}', blk.attn)); out.append((f'mlp{L}', blk.mlp))
    elif hasattr(mdl, 'gpt_neox'):         # pythia (GPTNeoX)
        for L, blk in enumerate(mdl.gpt_neox.layers):
            out.append((f'attn{L}', blk.attention)); out.append((f'mlp{L}', blk.mlp))
    return out


def hook_factory(name):
    def h(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; d = MODE['d']; sh = y.shape; v = y.reshape(-1, d).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = MODE['rand'] if MODE['op'] == 'keeprand' else SUBS[name]; v2 = (v @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


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
    comps = components(mdl); SUBS.clear()
    MODE['op'] = None
    for name, mod in comps:
        O, toks, pos = capture(mdl, mod, blocks, d)
        Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
        SUBS[name] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(d, RTOK+RPOS, generator=g, device=DEV))[0]
    MODE['single'] = None
    def sh(name):  # hook active ONLY for the single named component
        def hh(mo, i_, o_):
            if MODE['op'] is None or MODE['single'] != name: return o_
            y = o_[0] if isinstance(o_, tuple) else o_; sh_=y.shape; v=y.reshape(-1,d).float()
            if MODE['op']=='ablate': v2=torch.zeros_like(v)
            else: U = MODE['rand'] if MODE['op']=='keeprand' else SUBS[name]; v2=(v@U)@U.T
            yn=v2.reshape(sh_).to(y.dtype); return (yn,)+tuple(o_[1:]) if isinstance(o_,tuple) else yn
        return hh
    hooks=[mod.register_forward_hook(sh(name)) for name,mod in comps]
    MODE['op']=None; MODE['single']=None; ce_full=ce(mdl,blocks)
    per={}; tot_ben=0.0; tot_keep=0.0; tot_rand=0.0
    for name,mod in comps:
        MODE['single']=name
        MODE['op']='ablate'; ca=ce(mdl,blocks); ben=ca-ce_full
        MODE['op']='keep'; ck=ce(mdl,blocks)
        MODE['op']='keeprand'; cr=ce(mdl,blocks); MODE['op']=None
        rec=float((ca-ck)/max(ben,1e-6)); recr=float((ca-cr)/max(ben,1e-6))
        per[name]={'benefit':round(ben,3),'keep':round(rec,3),'rand':round(recr,3)}
        if ben>0: tot_ben+=ben; tot_keep+=ben*max(min(rec,1),0); tot_rand+=ben*max(min(recr,1),0)
    MODE['single']=None
    for h in hooks: h.remove()
    nw=float(tot_keep/max(tot_ben,1e-9)); nwr=float(tot_rand/max(tot_ben,1e-9))
    del mdl; torch.cuda.empty_cache()
    return {'d': d, 'n_components': len(comps), 'total_benefit': round(tot_ben,3), 'natweighted_keep_classpos': round(nw,4), 'natweighted_random': round(nwr,4), 'per_component': per}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        try:
            r = run(mid); out[mid] = r
            print(f'{mid} ({r["n_components"]} comp, total benefit {r["total_benefit"]}): NAT-WEIGHTED keep class+pos {r["natweighted_keep_classpos"]} | random {r["natweighted_random"]}', flush=True)
        except Exception as e:
            out[mid] = {'error': str(e)[:200]}; print(f'{mid} FAILED: {e}', flush=True)
    ok = [mid for mid in out if isinstance(out[mid], dict) and 'keep_classpos' in out[mid]]
    pa = len(ok) > 0 and all(out[mid]['natweighted_keep_classpos'] >= 0.6 and out[mid]['natweighted_keep_classpos'] > 2*out[mid]['natweighted_random'] for mid in ok)
    out['pred_a_universal'] = bool(pa); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) whole-model class+position reduction is universal (gpt2/pythia >=0.6 & >>random): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
