"""Does the GENERATIVE content-priming (§1016) hold ACROSS the family, including the non-bilinear SwiGLU model? §1010/
§1011 showed the two machines are architecture-general STRUCTURALLY; this tests it BEHAVIORALLY: inject a topical
content word early and measure the topic-neighbor log-prob boost at a distant query, per model (bilin18, bilin12,
swiglu18). If the bag-of-words content behavior is architecture-general, topical injection primes the topic (content
>> function) in every model incl swiglu.

REGISTERED PREDICTIONS:
  (0) SANITY: function-word injection primes far less than content-word injection in every model.
  (a) GENERATIVE CONTENT-PRIMING ARCHITECTURE-GENERAL: injecting a topical content word raises its topic-neighbors'
      log-prob (excl W, excl context) at a distant query in EVERY model incl swiglu18, and much more than a function
      word (content/function ratio > 3) -> the bag-of-words content behavior is architecture-general;
  (b) report content-mean, function-mean, and ratio per model."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import m as BILIN, DEV
import census_lib as cl
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_family_results.json'
NEVAL = 200; SEQ = 256; INJ_POS = 3; QUERY = 150; NNEIGH = 20
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden']
FUNCTION_WORDS = [' the', ' of', ' and', ' to']


def forward_logits(mdl, idx, Dm):
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(mdl.lm_head(F.rms_norm(x, (Dm,)))/30.0)


@torch.no_grad()
def neighbors(mdl, wid, k):
    W = mdl.lm_head.weight.float(); wv = W[wid] / (W[wid].norm() + 1e-9)
    sims = (W / (W.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def boost(mdl, blocks, wid, neigh, Dm):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous()
        inj_idx = base_idx.clone(); inj_idx[:, INJ_POS] = wid
        lb = F.log_softmax(forward_logits(mdl, base_idx, Dm).float()[:, -1], -1)
        li = F.log_softmax(forward_logits(mdl, inj_idx, Dm).float()[:, -1], -1)
        has = (base_idx == wid).any(1)
        dlp = (li[:, neigh] - lb[:, neigh]).mean(1)[~has]
        tot += float(dlp.sum()); n += int((~has).sum())
    return tot / max(n, 1)


@torch.no_grad()
def run_model(mdl, blocks, tid, tag):
    Dm = mdl.transformer.wte.weight.shape[1]; V = int(mdl.lm_head.weight.shape[0])
    cvals, fvals = [], []
    for w in CONTENT_WORDS:
        wid = tid(w)
        if wid is None or wid >= V: continue
        cvals.append(boost(mdl, blocks, wid, neighbors(mdl, wid, NNEIGH), Dm))
    for w in FUNCTION_WORDS:
        wid = tid(w)
        if wid is None or wid >= V: continue
        fvals.append(boost(mdl, blocks, wid, neighbors(mdl, wid, NNEIGH), Dm))
    cm = float(np.mean(cvals)); fm = float(np.mean(fvals))
    res = {'content_mean': round(cm, 4), 'function_mean': round(fm, 4), 'ratio': round(cm/max(fm, 1e-6), 2)}
    print(f"{tag}: content {res['content_mean']} function {res['function_mean']} ratio {res['ratio']}", flush=True)
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    out = {'models': {}}
    out['models']['bilin18'] = run_model(BILIN, rows, tid, 'bilin18')
    for short in ['bilin12', 'swiglu18']:
        try:
            mdl, cfg = load_elriggs(short); mdl = mdl.to(DEV).eval()
            out['models'][short] = run_model(mdl, rows, tid, short); del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = [k for k in out['models'] if 'ratio' in out['models'][k]]
    out['pred_a_generative_content_general'] = bool(len(ok) >= 2 and all(out['models'][k]['ratio'] > 3 and out['models'][k]['content_mean'] > 0.1 for k in ok))
    out['swiglu_ratio'] = out['models'].get('swiglu18', {}).get('ratio')
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a generative content-priming architecture-general {out['pred_a_generative_content_general']} | swiglu ratio {out['swiglu_ratio']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
