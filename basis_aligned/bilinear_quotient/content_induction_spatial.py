"""EXPLAIN the §1031 content<-induction amplification (induction firing makes a content word's logit contribution ~1.8x
larger). Leading hypothesis: induction fires by attending from the current token (pos 149) back to the earlier
matching token (pos 5) and copying its successor; that attention may ALSO carry the CONTENT of the matched region
forward -- and the injected content word W sat right beside the match (pos 3, next to A@5). If so, the amplification
should be SPATIAL: large when W is NEAR the induction-match position, small when W is FAR from it. If instead the
amplification is uniform across W's position, it is a GLOBAL induction-state effect, not attention carrying nearby
content.

Design: induction bigram A@5,B@6 with A@149 (trigger) vs D@149 (non-trigger, matched); measure the content logit-boost
of W at several positions (3=beside match, 50, 100=far from match). amplification(W_pos) = ce_trigger/ce_nontrigger at
that position (the trigger/non-trigger ratio controls for recency, since both use the same W_pos). All on RAW LOGITS
(per §1031).

REGISTERED PREDICTIONS:
  (0) SANITY: content logit-boost > 0 at every W position in both backgrounds.
  (a) SPATIAL (induction attention carries nearby content): amplification is LARGER for W NEAR the match (pos 3) than
      for W FAR from the match (pos 100) -- amp(3) > amp(100) by a clear margin -> the §1031 interaction is induction
      attention incidentally forwarding nearby content;
  (b) GLOBAL alternative: if amplification is ~uniform across W position, it is a global induction-state effect, not
      spatial. Report amplification vs W-position."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_induction_spatial_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; PA = 5; PB = 6; NNEIGH = 20; NTRIALS = 16
W_POSITIONS = [3, 50, 100]  # 3 = beside the match (pos 5); 100 = far from match, near query
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
def mean_logit(blocks, edits, targets, exclude):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); idx = bb[:, :QUERY].contiguous().clone()
        for p, t in edits.items(): idx[:, p] = t
        lg = forward_logits(idx).float()[:, -1]
        has = torch.zeros(idx.shape[0], dtype=torch.bool, device=DEV)
        for w in exclude: has |= (bb[:, :QUERY] == w).any(1)
        tot += float(lg[:, targets].mean(1)[~has].sum()); n += int((~has).sum())
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    uniq = np.unique(rows.cpu().numpy().reshape(-1)); rng = np.random.RandomState(0)
    pool = [int(t) for t in uniq if 100 < int(t) < 50000]; rng.shuffle(pool)
    cwords = [tid(w) for w in CONTENT_WORDS if tid(w) is not None]
    amp = {str(p): [] for p in W_POSITIONS}
    it = iter(pool)
    for k in range(NTRIALS):
        W = cwords[k % len(cwords)]; nW = neighbors(W, NNEIGH)
        try: A = next(it); B = next(it); Dd = next(it)
        except StopIteration: break
        excl = [W, A, B, Dd]
        trig = {PA: A, PB: B, QUERY-1: A}; nont = {PA: A, PB: B, QUERY-1: Dd}
        for p in W_POSITIONS:
            if p in (PA, PB, QUERY-1): continue
            ce_t = mean_logit(rows, {**trig, p: W}, nW, excl) - mean_logit(rows, trig, nW, excl)
            ce_n = mean_logit(rows, {**nont, p: W}, nW, excl) - mean_logit(rows, nont, nW, excl)
            amp[str(p)].append(ce_t / max(ce_n, 1e-6))
    out = {'amplification_by_Wpos': {p: round(float(np.mean(v)), 3) for p, v in amp.items() if v},
           'match_pos': PA, 'query': QUERY}
    a_near = out['amplification_by_Wpos'].get('3'); a_far = out['amplification_by_Wpos'].get('100')
    out['amp_near_match'] = a_near; out['amp_far_match'] = a_far
    out['pred_a_spatial'] = bool(a_near is not None and a_far is not None and a_near > a_far + 0.3)
    out['pred_b_global'] = bool(a_near is not None and a_far is not None and abs(a_near - a_far) < 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"amplification by W-position (match at {PA}): {out['amplification_by_Wpos']}", flush=True)
    print(f"near-match(3) {a_near} vs far-match(100) {a_far}", flush=True)
    print(f"pred_a spatial {out['pred_a_spatial']} | pred_b global {out['pred_b_global']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
