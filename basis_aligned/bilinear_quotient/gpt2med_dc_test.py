"""IS GPT2-MEDIUM's mlp0 EXCEPTION A DROPPED-MEAN (DC) ARTIFACT? (settle 802/803/804).
gpt2_mlp0_compare showed gpt2-medium mlp0 is LOW-rank (eff-rank 26) yet keep-only its
OWN top-128 principal directions is NEGATIVE -- impossible unless the projection drops
something critical. All the subspaces (token-mean, own-SVD) are built from CENTERED
data, so keep-only-proj DROPS the per-component MEAN (DC) offset. Hypothesis: gpt2-
medium mlp0 has a large, LOSS-CRITICAL constant bias that the centered projection
discards, making keep harmful -- a METRIC artifact, not genuine non-separability.
Test: keep class+position with the MEAN PRESERVED (v_kept = mean + proj_U(v - mean))
vs the centered version (proj_U(v)). If +mean goes POSITIVE, gpt2-medium IS class+
position and 802's "genuine exception" was a dropped-DC artifact. Also test gpt2-small
(should be fine either way -> its DC is not critical).

REGISTERED PREDICTIONS:
  (0) SANITY: centered keep reproduces the negative gpt2-medium value;
  (a) DC ARTIFACT: for gpt2-medium mlp0, keep-(class+position)+MEAN is POSITIVE and
      substantial (>= 0.6), while centered is negative -> the exception was a dropped-
      mean artifact; gpt2-medium IS class+position (+ a large critical bias);
  (b) gpt2-small mlp0 is high both ways (its DC is not critical);
  ALT: if +mean stays low/negative for gpt2-medium, it is a GENUINE exception (802 stands)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gpt2med_dc_test_results.json'
MODELS = ['gpt2', 'gpt2-medium']; SEQ = 128; NBLOCK = 96; MINCOUNT = 5
HK = {'U': None, 'mean': None, 'op': None, 'd': None}


def hook(mo, i_, o_):
    if HK['op'] is None: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; d = HK['d']; sh = y.shape; v = y.reshape(-1, d).float()
    if HK['op'] == 'ablate':
        v2 = torch.zeros_like(v)
    elif HK['op'] == 'keep':                       # centered projection (drops mean)
        U = HK['U']; v2 = (v @ U) @ U.T
    else:                                          # 'keepmean': preserve the mean
        U = HK['U']; mu = HK['mean']; v2 = mu + ((v - mu) @ U) @ U.T
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


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.n_embd; HK['d'] = d; mod = mdl.transformer.h[0].mlp
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    O, toks, pos = capture(mdl, mod, blocks, d)
    gmean = O.mean(0, keepdim=True)                # (1, d) per-component output mean
    mean_norm = float(gmean.norm()); out_norm = float(O.norm(dim=1).mean())
    Utok = mean_subspace(O, toks, 64); Upos = mean_subspace(O, pos, 32)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :96].contiguous()
    HK['U'] = Ucp; HK['mean'] = gmean
    h = mod.register_forward_hook(hook)
    HK['op'] = None; ce_full = ce(mdl, blocks)
    HK['op'] = 'ablate'; ce_abl = ce(mdl, blocks); ben = ce_abl - ce_full
    HK['op'] = 'keep'; ce_keep = ce(mdl, blocks)
    HK['op'] = 'keepmean'; ce_keepm = ce(mdl, blocks); HK['op'] = None
    h.remove(); del mdl; torch.cuda.empty_cache()
    rec = float((ce_abl-ce_keep)/max(ben, 1e-6)); recm = float((ce_abl-ce_keepm)/max(ben, 1e-6))
    return {'d': d, 'benefit': round(ben, 4), 'mean_norm': round(mean_norm, 2), 'out_norm': round(out_norm, 2),
            'mean_over_out': round(mean_norm/max(out_norm, 1e-6), 3),
            'keep_centered': round(rec, 4), 'keep_with_mean': round(recm, 4)}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        r = run(mid); out[mid] = r
        print(f'{mid} (d{r["d"]}): mean/out {r["mean_over_out"]} | keep CENTERED {r["keep_centered"]} | keep +MEAN {r["keep_with_mean"]}', flush=True)
    med = out['gpt2-medium']
    pa = med['keep_with_mean'] >= 0.6 and med['keep_centered'] < 0.3
    out['pred_a_dc_artifact'] = bool(pa); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) gpt2-medium exception is a DROPPED-MEAN artifact (keep+mean>=0.6, centered<0.3): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
