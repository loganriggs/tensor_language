"""DO GPT-2 AND PYTHIA REDISCOVER THE SAME GRAMMATICAL CLASSES? (§825 cross-model extension).
§825 showed bilin18's first-MLP class directions are nameable parts of speech (determiners,
pronouns vs numbers, punctuation, conjunctions, suffixes, quantifiers). Test universality: run
the same class-naming (SVD of token-conditional-mean first-MLP outputs, top directions by
loading tokens) on GPT-2 and Pythia-410M, decoded with each model's own tokenizer.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-token-label control gives incoherent token lists;
  (a) UNIVERSAL: GPT-2 and Pythia first-MLP class directions are also nameable grammatical
      categories, overlapping bilin18's set (determiners/pronouns/numbers/punctuation/...);
  (b) report the top directions' loading tokens per model for eyeball comparison."""
import json, time, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_class_naming_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 160; MINCOUNT = 8; NDIR = 12; NTOK = 10


def get_text(nc=2000, npass=80):
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
    t = []
    for i, ex in enumerate(ds):
        t.append(ex['text'][:nc])
        if i >= npass: break
    return '\n\n'.join(t)


def first_mlp(mdl):
    if hasattr(mdl, 'transformer'): return mdl.transformer.h[0].mlp
    return mdl.gpt_neox.layers[0].mlp


@torch.no_grad()
def capture(mdl, mod, blocks, d):
    cap = []; toks = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, d))
    hh = mod.register_forward_hook(h)
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i+4].to(DEV); mdl(b); toks.append(b.reshape(-1).cpu().numpy())
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks)


def class_dirs(O, labels, g):
    uniq = []; rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        uniq.append(int(t)); rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    Mdev = torch.stack(rows, 0)
    W = Mdev * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return np.array(uniq), Mdev, Vh, S


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    O, toks = capture(mdl, first_mlp(mdl), blocks, d); g = O.mean(0, keepdim=True)
    uniq, Mdev, Vh, S = class_dirs(O, toks, g)
    proj = Mdev @ Vh.T
    dirs = []
    for k in range(min(NDIR, Vh.shape[0])):
        col = proj[:, k].cpu().numpy()
        pos = uniq[np.argsort(-col)[:NTOK]]; neg = uniq[np.argsort(col)[:NTOK]]
        dirs.append({'k': k, 'top_pos': [repr(tok.decode([int(t)])) for t in pos],
                     'top_neg': [repr(tok.decode([int(t)])) for t in neg]})
    eff = float((S.sum()**2)/(S**2).sum())
    rng = np.random.RandomState(0); sh = toks.copy(); rng.shuffle(sh)
    u2, Md2, Vh2, _ = class_dirs(O, sh, g); col2 = (Md2 @ Vh2.T)[:, 0].cpu().numpy()
    shuf0 = [repr(tok.decode([int(t)])) for t in u2[np.argsort(-col2)[:NTOK]]]
    del mdl; torch.cuda.empty_cache()
    return {'d': d, 'n_tokens': len(uniq), 'eff_num_dirs': round(eff, 1), 'directions': dirs, 'shuffled_dir0': shuf0}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        r = run(mid); out[mid] = r
        print(f'\n=== {mid} (eff {r["eff_num_dirs"]} dirs, {r["n_tokens"]} tokens) ===', flush=True)
        for dd in r['directions'][:8]:
            print(f"  dir {dd['k']}: +[{' '.join(dd['top_pos'][:7])}]  -[{' '.join(dd['top_neg'][:5])}]", flush=True)
        print(f"  shuffled dir0 (control): {' '.join(r['shuffled_dir0'][:7])}", flush=True)
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
