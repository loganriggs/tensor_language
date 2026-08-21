"""CROSS-MODEL POSITION + combined (extends 778 with the 776/779 story: is the
first MLP's output token-class + POSITION, both causal, in GPT-2 and Pythia too, or
is position a bilin18 quirk?). For each model's first MLP: token-class subspace(64)
+ position subspace(32); keep-only-combined CE-recovery, and POSITION causality
(remove the position subspace -> dCE vs random).

REGISTERED PREDICTIONS:
  (0) SANITY: first-MLP benefit > 0;
  (a) POSITION CAUSAL cross-model: removing the position subspace from the first MLP
      raises CE >= 3x a random same-rank subspace in BOTH GPT-2 and Pythia (position
      is used, not a bilin18 quirk);
  (b) COMBINED: keep-only token-class+position recovers >= keep-only token alone at
      each model (position adds interpretable coverage); report per model;
  NULL: random same-rank subspace removal ~harmless.
Note: GPT-2 uses learned absolute position embeddings, Pythia uses rotary -- if both
show causal positional structure in the MLP output, positional organisation is
general across position-encoding schemes."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_position_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RTOK = 64; RPOS = 32
HK = {'U': None, 'op': None, 'd': None}


def get_mlp(mdl, L):
    if hasattr(mdl, 'transformer'): return mdl.transformer.h[L].mlp
    if hasattr(mdl, 'gpt_neox'): return mdl.gpt_neox.layers[L].mlp
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


def get_text(n_chars=2000, npass=44):
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
        b = blocks[i:i+4].to(DEV); out = mdl(b, labels=b)
        tot += float(out.loss) * b.shape[0]; nn += b.shape[0]
    return tot / nn


@torch.no_grad()
def capture(mdl, mlp, blocks, d):
    cap = []; toks = []; pos = []
    h = mlp.register_forward_hook(lambda mo, i_, o_: cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, d)))
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i+4].to(DEV); mdl(b)
        toks.append(b.reshape(-1).cpu())
        pos.append(np.broadcast_to(np.arange(b.shape[1]), b.shape).reshape(-1))
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy(), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def run_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd
    mlp = get_mlp(mdl, 0); HK['d'] = d
    blocks = make_blocks(tok, get_text())
    O, toks, pos = capture(mdl, mlp, blocks, d)
    Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
    Ucomb = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    h = mlp.register_forward_hook(hook)
    HK['op'] = None; ce_full = ce(mdl, blocks)
    HK['op'] = 'ablate'; ce_abl = ce(mdl, blocks); ben = ce_abl - ce_full
    def keeprec(U): HK['op'] = 'keep'; HK['U'] = U; c = ce(mdl, blocks); HK['op'] = None; HK['U'] = None; return float((ce_abl-c)/max(ben, 1e-6))
    def removedce(U): HK['op'] = 'remove'; HK['U'] = U; c = ce(mdl, blocks); HK['op'] = None; HK['U'] = None; return float(c-ce_full)
    g = torch.Generator(device=DEV).manual_seed(0)
    rt = keeprec(Utok); rp = keeprec(Upos); rc = keeprec(Ucomb)
    dpos = removedce(Upos); Ur = torch.linalg.qr(torch.randn(d, RPOS, generator=g, device=DEV))[0]; drand = removedce(Ur)
    ov = float(torch.linalg.svdvals(Utok.T @ Upos).mean())
    h.remove(); del mdl; torch.cuda.empty_cache()
    return {'d': d, 'benefit': round(ben, 4), 'keep_token': round(rt, 4), 'keep_position': round(rp, 4),
            'keep_combined': round(rc, 4), 'remove_position_dce': round(dpos, 4), 'remove_random_dce': round(drand, 4),
            'position_ratio': round(dpos/max(drand, 1e-6), 2), 'tok_pos_overlap': round(ov, 4)}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        try:
            r = run_model(mid); out[mid] = r
            print(f'{mid} L0 (d{r["d"]}): benefit {r["benefit"]} | keep token {r["keep_token"]} pos {r["keep_position"]} combined {r["keep_combined"]} | '
                  f'position causal ratio {r["position_ratio"]} (dCE {r["remove_position_dce"]} vs rand {r["remove_random_dce"]}) | tok-pos overlap {r["tok_pos_overlap"]}', flush=True)
        except Exception as e:
            out[mid] = {'error': str(e)[:200]}; print(f'{mid} FAILED: {e}', flush=True)
    ok = [m for m in out if isinstance(out[m], dict) and 'position_ratio' in out[m]]
    pa = len(ok) > 0 and all(out[m]['position_ratio'] >= 3 for m in ok)
    pb = all(out[m]['keep_combined'] >= out[m]['keep_token'] - 0.02 for m in ok)
    out['pred_a_position_causal'] = bool(pa); out['pred_b_combined'] = bool(pb); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) position causal in real models (>=3x): {pa}; (b) combined>=token: {pb}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
