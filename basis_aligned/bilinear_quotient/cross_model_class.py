"""CROSS-MODEL CLASS-SHARPENING (does 782's "the first MLP COMPUTES/sharpens
grammatical class" hold for real models? extends 778/781). For GPT-2 and Pythia-410m,
build the first MLP's per-token mean table and compare its grammatical-class geometry
to the raw embedding: effective-rank collapse and Fisher class-separation ratio, on
labelled classes (determiner/number/punct/pronoun/prep) using each model's tokenizer.

REGISTERED PREDICTIONS:
  (0) SANITY: enough labelled tokens per model (>= 30);
  (a) CLASS-COMPUTING GENERAL: in BOTH models the first-MLP mean table has (i) much
      lower effective rank than the embedding (class collapse) and (ii) a HIGHER
      Fisher class-separation ratio than the embedding (>= 1.2x) -- the first MLP
      sharpens grammatical class, cross-model, like bilin18 (782: rank 24 vs 132,
      Fisher 1.8x);
  (b) report eff-rank + Fisher per model;
  NULL: shuffled class labels give Fisher ~ chance."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_class_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 800; MINCOUNT = 20     # more data (user: default to more data)
CLASSES = {
    'determiner': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'no', 'some', 'any', 'each', 'every', 'all', 'both', 'which', 'what', 'whose'},
    'number': {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '20', '30', '50', '100', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'twenty', 'thirty', 'fifty', 'hundred', 'thousand', 'million', 'billion', 'first', 'second', 'third'},
    'punct': {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '"', "'", '--', '-', '/', '*', '&', '%', '#', '@', '...', '।'},
    'pronoun': {'it', 'he', 'she', 'they', 'we', 'you', 'i', 'him', 'them', 'us', 'me', 'his', 'hers', 'theirs', 'himself', 'herself', 'itself', 'themselves', 'who', 'whom'},
    'prep': {'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from', 'into', 'over', 'under', 'about', 'after', 'before', 'through', 'between', 'against', 'during', 'without', 'within', 'across', 'behind', 'beyond', 'toward', 'upon'},
    'aux': {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must'},
    'conj': {'and', 'or', 'but', 'nor', 'yet', 'so', 'because', 'although', 'though', 'while', 'whereas', 'if', 'unless', 'since', 'when', 'where'},
}


def get_mlp(mdl):
    if hasattr(mdl, 'transformer'): return mdl.transformer.h[0].mlp
    if hasattr(mdl, 'gpt_neox'): return mdl.gpt_neox.layers[0].mlp
    raise ValueError('arch')


def get_text(nc=2000, npass=80):
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
    t = []
    for i, ex in enumerate(ds):
        t.append(ex['text'][:nc])
        if i >= npass: break
    return '\n\n'.join(t)


def fisher(V, lab):
    mu = V.mean(0); b = 0.0; w = 0.0
    for c in set(lab):
        ii = [i for i, l in enumerate(lab) if l == c]
        if len(ii) < 2: continue
        s = V[ii]; mc = s.mean(0); b += len(ii)*float(((mc-mu)**2).sum()); w += float(((s-mc)**2).sum())
    return b/max(w, 1e-9)


def eff_rank(X):
    s2 = torch.linalg.svdvals(X - X.mean(0, keepdim=True))**2
    return float((s2.sum()**2)/(s2**2).sum())


@torch.no_grad()
def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]
    nb = min(NBLOCK, ids.shape[0]//SEQ); blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    ssum = {}; scnt = {}; cur = {'b': None}
    mlp = get_mlp(mdl); h = mlp.register_forward_hook(lambda mo, i_, o_: cur.__setitem__('b', (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, d)))
    for i in range(0, nb, 4):
        b = blocks[i:i+4].to(DEV); mdl(b); O = cur['b']; tk = b.reshape(-1)
        for t in torch.unique(tk):
            tid = int(t); mk = tk == t; s = O[mk].sum(0); c = int(mk.sum())
            if tid in ssum: ssum[tid] += s; scnt[tid] += c
            else: ssum[tid] = s.clone(); scnt[tid] = c
    h.remove()
    kept = [t for t in ssum if scnt[t] >= MINCOUNT]
    M = torch.stack([ssum[t]/scnt[t] for t in kept], 0)
    E = mdl.get_input_embeddings().weight.data.float().to(DEV)[torch.tensor(kept, device=DEV)]
    def classify(tid):
        w = tok.decode([tid]).strip().lower()
        for cls, mem in CLASSES.items():
            if w in mem: return cls
        return None
    lab = [classify(t) for t in kept]; ki = [i for i, l in enumerate(lab) if l is not None]
    Mk = M[torch.tensor(ki, device=DEV)]; Ek = E[torch.tensor(ki, device=DEV)]; labk = [lab[i] for i in ki]
    fm = fisher(Mk, labk); fe = fisher(Ek, labk)
    rng = np.random.RandomState(0); fn = fisher(Mk, list(rng.permutation(labk)))
    del mdl; torch.cuda.empty_cache()
    return {'d': d, 'n_tokens': len(kept), 'n_labelled': len(ki), 'eff_rank_mean': round(eff_rank(M), 2),
            'eff_rank_emb': round(eff_rank(E), 2), 'fisher_mean': round(fm, 4), 'fisher_emb': round(fe, 4),
            'fisher_null': round(fn, 4), 'fisher_ratio': round(fm/max(fe, 1e-9), 3)}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        try:
            r = run(mid); out[mid] = r
            print(f'{mid}: {r["n_labelled"]} labelled | eff-rank mean {r["eff_rank_mean"]} emb {r["eff_rank_emb"]} | Fisher mean {r["fisher_mean"]} emb {r["fisher_emb"]} (ratio {r["fisher_ratio"]}, null {r["fisher_null"]})', flush=True)
        except Exception as e:
            out[mid] = {'error': str(e)[:200]}; print(f'{mid} FAILED: {e}', flush=True)
    ok = [mid for mid in out if 'fisher_ratio' in out[mid]]
    pa = len(ok) > 0 and all(out[mid]['fisher_ratio'] >= 1.2 and out[mid]['eff_rank_mean'] < out[mid]['eff_rank_emb'] for mid in ok)
    out['pred_a_class_computing_general'] = bool(pa); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) first MLP sharpens grammatical class cross-model (Fisher>=1.2x & rank-collapse): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
