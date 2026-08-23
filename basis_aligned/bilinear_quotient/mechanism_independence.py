"""Do the mechanisms operate INDEPENDENTLY when co-active? The two-machine+induction account implies content (broad
topic pooling) and induction (long-range copy) are separate mechanisms that should compose without interference. Test
generatively: co-inject a topical CONTENT word AND an induction bigram in the same context, and check that BOTH
effects survive at ~their solo strengths.

Setup (query at 150): content word W at pos 3 (measure W's topic-neighbor boost); induction bigram "A B" at pos 5-6
with A placed at the last fed position 149 (measure ΔlogP(B)). Conditions: content-only, induction-only, and BOTH.

REGISTERED PREDICTIONS:
  (0) SANITY: content-only reproduces the topical boost (~0.5, §1016); induction-only reproduces the copy boost
      (~7-8 nats, §1025).
  (a) INDEPENDENCE: with BOTH injected, the content boost stays ~its solo value AND the induction boost stays ~its
      solo value (each retention in ~[0.7, 1.3]) -> the two mechanisms compose without interfering, as the account
      implies;
  (b) report solo vs joint content-boost and induction-boost + retentions."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mechanism_independence_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; W_POS = 3; PA = 5; PB = 6; NNEIGH = 20; NTRIALS = 16
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden']


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k):
    Wm = m.lm_head.weight.float(); wv = Wm[wid] / (Wm[wid].norm() + 1e-9)
    sims = (Wm / (Wm.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def run(blocks, Wc, neighW, A, B, inject):
    # inject in {'content','induction','both'}; return (content boost = W-neighbor Δlp, induction boost = ΔlogP(B))
    cC = 0.0; cI = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base = bb[:, :QUERY].contiguous()
        inj = base.clone()
        if inject in ('induction', 'both'): inj[:, PA] = A; inj[:, PB] = B; inj[:, QUERY-1] = A
        if inject in ('content', 'both'): inj[:, W_POS] = Wc
        lb = F.log_softmax(forward_logits(base).float()[:, -1], -1)
        li = F.log_softmax(forward_logits(inj).float()[:, -1], -1)
        has = (base == Wc).any(1)
        cC += float((li[:, neighW] - lb[:, neighW]).mean(1)[~has].sum())
        cI += float((li[:, B] - lb[:, B])[~has].sum())
        n += int((~has).sum())
    return cC/max(n, 1), cI/max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    uniq = np.unique(rows.cpu().numpy().reshape(-1)); rng = np.random.RandomState(0)
    pool = [int(t) for t in uniq if 100 < int(t) < 50000]; rng.shuffle(pool)
    cwords = [tid(w) for w in CONTENT_WORDS if tid(w) is not None]
    soloC = []; soloI = []; jointC = []; jointI = []
    it = iter(pool)
    for k in range(NTRIALS):
        Wc = cwords[k % len(cwords)]; neighW = neighbors(Wc, NNEIGH)
        try: A = next(it); B = next(it)
        except StopIteration: break
        cC_c, _ = run(rows, Wc, neighW, A, B, 'content')
        _, cI_i = run(rows, Wc, neighW, A, B, 'induction')
        cC_b, cI_b = run(rows, Wc, neighW, A, B, 'both')
        soloC.append(cC_c); soloI.append(cI_i); jointC.append(cC_b); jointI.append(cI_b)
    soloC = np.array(soloC); soloI = np.array(soloI); jointC = np.array(jointC); jointI = np.array(jointI)
    out = {'content_solo': round(float(soloC.mean()), 4), 'content_joint': round(float(jointC.mean()), 4),
           'induction_solo': round(float(soloI.mean()), 4), 'induction_joint': round(float(jointI.mean()), 4)}
    out['content_retention'] = round(out['content_joint']/max(out['content_solo'], 1e-6), 3)
    out['induction_retention'] = round(out['induction_joint']/max(out['induction_solo'], 1e-6), 3)
    out['pred_a_independent'] = bool(0.7 < out['content_retention'] < 1.3 and 0.7 < out['induction_retention'] < 1.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content: solo {out['content_solo']} joint {out['content_joint']} (ret {out['content_retention']})", flush=True)
    print(f"induction: solo {out['induction_solo']} joint {out['induction_joint']} (ret {out['induction_retention']})", flush=True)
    print(f"pred_a mechanisms-independent {out['pred_a_independent']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
