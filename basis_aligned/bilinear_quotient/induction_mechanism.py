"""DOES bilin18 do INDUCTION, and is that why 'seen' tokens are cheap? (mechanistic follow-up to §876). §876:
already-seen tokens cost ~1.7 nats vs ~4.2 for first-mentions. The classic mechanism that makes a repeated
token cheap is INDUCTION: [A][B]...[A]->predict [B] (copy what followed this token last time). Test whether
bilin18 has it and locate it.

Method:
  1. SYNTHETIC induction: sequences of RANDOM tokens repeated twice ([P][P], each P length L). At the second
     copy, predicting position L+i+1 = P[i+1] requires copying what followed P[i] in the first copy. Induction
     score = mean loss on the FIRST copy minus mean loss on the SECOND copy (positive = induction). Random
     tokens have no statistical structure, so any 2nd-copy advantage IS induction, not n-gram memorization.
  2. LOCATE: ablate each attention layer (zero its output) and measure the drop in the synthetic induction
     score -> the layer(s) whose ablation collapses induction are the induction layer(s).
  3. NATURAL cross-check: on FineWeb, positions whose (current,next) BIGRAM already occurred earlier in the
     context ('inductable') vs not -> inductable positions should be much cheaper.
Controls: shuffled-target null for the synthetic score (predicting a random token gives ~0 induction);
first-copy loss is the within-experiment baseline.

REGISTERED PREDICTIONS:
  (0) SANITY: first-copy synthetic loss ~ ln(vocab-ish) high; natural inductable positions cheaper than non;
  (a) INDUCTION PRESENT & LOCALIZED: second-copy loss << first-copy (large positive induction score), and
      ablating a SMALL number of attention layers collapses most of it (localized induction heads/layers),
      rather than every layer contributing equally;
  (b) if second-copy ~ first-copy, bilin18 lacks induction and the seen-token cheapness is topic/frequency,
      not copying (report honestly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_mechanism_results.json'
NL = len(m.transformer.h); V = int(m.lm_head.weight.shape[0])
NSYN = 48; L = 64; NEVAL_NAT = 120; SEQ = 256
ABL = {'layer': -1}


def bilin_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ablate_hook(mo, i_, o_):
    y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
    return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z


@torch.no_grad()
def syn_induction_score(seqs):
    """mean(loss over first-copy positions) - mean(loss over second-copy positions)."""
    idx = seqs[:, :-1].contiguous(); tgt = seqs[:, 1:].contiguous()
    lp = F.log_softmax(bilin_logits(idx).float(), -1)
    l = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)          # (B, 2L-1)
    first = l[:, :L-1].mean(); second = l[:, L:2*L-1].mean()   # second copy positions
    return float(first), float(second)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    g = torch.Generator(device=DEV).manual_seed(0)
    # random-token sequences, hi-frequency band to stay in-vocab-sensible (avoid rare byte tokens)
    base = torch.randint(0, 50000, (NSYN, L), generator=g, device=DEV)
    seqs = torch.cat([base, base], 1)                          # (NSYN, 2L)
    first0, second0 = syn_induction_score(seqs)
    ind0 = first0 - second0
    # shuffled-target null: shuffle the second copy so it no longer matches the first
    perm = torch.stack([torch.randperm(L, generator=g, device=DEV) for _ in range(NSYN)])
    seqs_shuf = torch.cat([base, base.gather(1, perm)], 1)
    fs, ss = syn_induction_score(seqs_shuf); ind_null = fs - ss
    # LOCATE: ablate each attention layer, measure induction score drop
    per_layer = {}
    for Li in range(NL):
        hh = m.transformer.h[Li].attn.register_forward_hook(ablate_hook)
        f1, s1 = syn_induction_score(seqs); hh.remove()
        per_layer[Li] = round((f1 - s1), 3)                    # induction score with layer Li ablated
    drops = {Li: round(ind0 - per_layer[Li], 3) for Li in range(NL)}   # how much ablating Li reduces induction
    top = sorted(drops.items(), key=lambda kv: -kv[1])[:5]
    # NATURAL cross-check: inductable bigram positions vs not
    rows = cl.fineweb_rows(NEVAL_NAT); blocks = rows[:, :SEQ].contiguous()
    li_ind = []; li_non = []
    for i in range(0, blocks.shape[0], 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(bilin_logits(idx).float(), -1); l = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        S = bb.cpu().numpy()
        for r in range(bb.shape[0]):
            seen = {}
            for p in range(SEQ-1):
                cur = int(S[r, p]); nxt = int(S[r, p+1]); key = cur
                if key in seen and seen[key] == nxt: li_ind.append(float(l[r, p]))
                else: li_non.append(float(l[r, p]))
                seen[key] = nxt
    out = {'synthetic': {'first_copy_loss': round(first0, 3), 'second_copy_loss': round(second0, 3),
                         'induction_score': round(ind0, 3), 'shuffled_target_null': round(ind_null, 3)},
           'induction_score_with_layer_ablated': per_layer, 'induction_drop_by_layer': drops,
           'top5_induction_layers': top,
           'natural': {'inductable_bigram_loss': round(float(np.mean(li_ind)), 3), 'n_inductable': len(li_ind),
                       'non_inductable_loss': round(float(np.mean(li_non)), 3), 'n_non': len(li_non)},
           'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_induction_localized'] = bool(ind0 > 1.0 and ind0 > 3*max(ind_null, 1e-6) and top[0][1] > 0.3 and
                                             sum(v for _, v in top[:3]) > 0.6*max(sum(drops.values()), 1e-6))
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"SYNTHETIC: first-copy loss {first0:.3f} -> second-copy {second0:.3f} | induction score {ind0:.3f} (shuffled-null {ind_null:.3f})", flush=True)
    print(f"top induction layers (score drop when ablated): {top}", flush=True)
    print(f"NATURAL: inductable-bigram loss {out['natural']['inductable_bigram_loss']} (n={len(li_ind)}) vs non {out['natural']['non_inductable_loss']} (n={len(li_non)})", flush=True)
    print(f"(a) induction present & localized: {out['pred_a_induction_localized']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
