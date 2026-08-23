"""NEW property test: does the content bag COMPOSE/MIX topics? Inject TWO different-topic words (A at pos 3, B at pos
12) and measure whether BOTH A's and B's topic-neighbors are boosted at a distant query (an additive MIXTURE), or one
topic dominates (winner-take-all). A bag-of-words pools everything, so it should prime BOTH topics simultaneously,
each somewhat diluted vs its solo injection but clearly present. This tests compositionality -- a property structural
(§995-998) and causal (§894) evidence did not directly show.

REGISTERED PREDICTIONS:
  (0) SPECIFICITY: when only A is injected, B's neighbors are NOT boosted (and vice versa) -- confirms the boosts are
      topic-specific (baseline for the mixture comparison).
  (a) ADDITIVE MIXTURE: with BOTH A and B injected, BOTH A-neighbors AND B-neighbors are boosted (each > ~0.15, well
      above 0), i.e. the bag primes both topics at once -- NOT winner-take-all;
  (b) MODEST DILUTION: each topic's boost with both injected is a large fraction (> ~0.5) of its solo boost -> the
      topics SHARE the pool roughly additively rather than suppressing each other. Report solo vs joint boosts and the
      joint/solo retention."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_mixing_results.json'
NEVAL = 200; SEQ = 256; POS_A = 3; POS_B = 12; QUERY = 150; NNEIGH = 20
PAIRS = [(' football', ' ocean'), (' hospital', ' music'), (' science', ' church'), (' army', ' garden'), (' computer', ' medicine')]


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k, exclude):
    W = m.lm_head.weight.float(); wv = W[wid] / (W[wid].norm() + 1e-9)
    sims = (W / (W.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    for e in exclude: sims[e] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def boosts(blocks, widA, widB, neighA, neighB, inject):
    # inject in {'A','B','AB'}; return (mean Δlp of neighA, mean Δlp of neighB) vs no-injection
    tA = 0.0; tB = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous(); inj_idx = base_idx.clone()
        if inject in ('A', 'AB'): inj_idx[:, POS_A] = widA
        if inject in ('B', 'AB'): inj_idx[:, POS_B] = widB
        lb = F.log_softmax(forward_logits(base_idx).float()[:, -1], -1)
        li = F.log_softmax(forward_logits(inj_idx).float()[:, -1], -1)
        has = (base_idx == widA).any(1) | (base_idx == widB).any(1)
        dA = (li[:, neighA] - lb[:, neighA]).mean(1)[~has]; dB = (li[:, neighB] - lb[:, neighB]).mean(1)[~has]
        tA += float(dA.sum()); tB += float(dB.sum()); n += int((~has).sum())
    return tA/max(n, 1), tB/max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    out = {'pairs': {}}
    soloA_r = []; soloB_r = []; jointA_r = []; jointB_r = []; crossA_r = []
    for a, b in PAIRS:
        wa, wb = tid(a), tid(b)
        if wa is None or wb is None: continue
        nA = neighbors(wa, NNEIGH, exclude=[wb]); nB = neighbors(wb, NNEIGH, exclude=[wa])
        aa, ab = boosts(rows, wa, wb, nA, nB, 'A')      # inject A: A-boost (solo), B-boost (cross null)
        ba, bb_ = boosts(rows, wa, wb, nA, nB, 'B')     # inject B: A-boost (cross null), B-boost (solo)
        ja, jb = boosts(rows, wa, wb, nA, nB, 'AB')     # inject both
        out['pairs'][f'{a}+{b}'] = {'soloA': round(aa, 3), 'soloB': round(bb_, 3), 'jointA': round(ja, 3),
                                    'jointB': round(jb, 3), 'crossA_whenB': round(ba, 3), 'crossB_whenA': round(ab, 3)}
        soloA_r.append(aa); soloB_r.append(bb_); jointA_r.append(ja); jointB_r.append(jb); crossA_r.append((ba+ab)/2)
        print(f"{a}+{b}: solo A {aa:.3f} B {bb_:.3f} | joint A {ja:.3f} B {jb:.3f} | cross(null) {(ba+ab)/2:.3f}", flush=True)
    solo = float(np.mean(soloA_r + soloB_r)); joint = float(np.mean(jointA_r + jointB_r)); cross = float(np.mean(crossA_r))
    out['solo_mean'] = round(solo, 3); out['joint_mean'] = round(joint, 3); out['cross_null_mean'] = round(cross, 3)
    out['joint_over_solo_retention'] = round(joint/max(solo, 1e-6), 3)
    out['pred_0_specific'] = bool(cross < 0.5*solo)
    out['pred_a_additive_mixture'] = bool(joint > 0.15 and joint > 2*cross)
    out['pred_b_modest_dilution'] = bool(out['joint_over_solo_retention'] > 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"solo {out['solo_mean']} | joint {out['joint_mean']} (retention {out['joint_over_solo_retention']}) | cross-null {out['cross_null_mean']}", flush=True)
    print(f"pred_0 specific {out['pred_0_specific']} | pred_a additive-mixture {out['pred_a_additive_mixture']} | pred_b modest-dilution {out['pred_b_modest_dilution']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
