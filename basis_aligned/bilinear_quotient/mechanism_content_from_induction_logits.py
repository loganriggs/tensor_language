"""DISENTANGLE §1030: is the ~2x content-boost amplification when induction fires a CONTENT-COMPUTATION interaction, or
a SOFTMAX-COUPLING artifact of both mechanisms writing the same logits? Repeat §1030's density-matched control but
measure the content effect on the RAW LOGITS (pre-softmax: 30*tanh(lm_head(rms_norm(x))/30)) instead of log-probs.
Softmax normalization cannot affect raw logits, so if the logit-space content boost is EQUAL in the induction-trigger
and non-trigger backgrounds, the content COMPUTATION (what is written to the logits) is INDEPENDENT of induction, and
§1030's log-prob 2x was purely softmax coupling (induction concentrating mass on B changing the logit->logprob map).

REGISTERED PREDICTIONS:
  (0) SANITY: both logit content-effects > 0.
  (a) CONTENT COMPUTATION INDEPENDENT (softmax coupling explains §1030): the LOGIT-space content boost retention
      (trigger/non-trigger) is ~1 (in [0.7,1.3]) -> induction firing does NOT change what content writes to the
      logits; §1030's 2x was softmax normalization -> the mechanisms are independent in COMPUTATION, coupled only at
      the shared softmax output;
  (b) if the logit retention is also ~2x, then induction genuinely amplifies the content computation (a real
      interaction) -- report which."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mechanism_content_from_induction_logits_results.json'
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
def mean_logit(blocks, edits, targets, exclude):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); idx = bb[:, :QUERY].contiguous().clone()
        for p, t in edits.items(): idx[:, p] = t
        lp = forward_logits(idx).float()[:, -1]  # RAW logits (pre-softmax)
        has = torch.zeros(idx.shape[0], dtype=torch.bool, device=DEV)
        for w in exclude: has |= (bb[:, :QUERY] == w).any(1)
        tot += float(lp[:, targets].mean(1)[~has].sum()); n += int((~has).sum())
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
    ce_trig = []; ce_nontrig = []
    it = iter(pool)
    for k in range(NTRIALS):
        W = cwords[k % len(cwords)]; nW = neighbors(W, NNEIGH)
        try: A = next(it); B = next(it); Dd = next(it)
        except StopIteration: break
        excl = [W, A, B, Dd]
        trig = {PA: A, PB: B, QUERY-1: A}          # last=A -> induction fires
        nont = {PA: A, PB: B, QUERY-1: Dd}         # last=D -> no induction (same A,B density)
        ce_trig.append(mean_logit(rows, {**trig, W_POS: W}, nW, excl) - mean_logit(rows, trig, nW, excl))
        ce_nontrig.append(mean_logit(rows, {**nont, W_POS: W}, nW, excl) - mean_logit(rows, nont, nW, excl))
    ct = float(np.mean(ce_trig)); cn = float(np.mean(ce_nontrig))
    out = {'logit_content_effect_trigger': round(ct, 4), 'logit_content_effect_nontrigger': round(cn, 4),
           'logit_retention_trigger_over_nontrigger': round(ct/max(cn, 1e-6), 3)}
    out['pred_0_sane'] = bool(ct > 0.1 and cn > 0.1)
    out['pred_a_content_computation_independent'] = bool(0.7 < out['logit_retention_trigger_over_nontrigger'] < 1.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"LOGIT content-effect: induction-trigger {ct:.3f} | non-trigger {cn:.3f} | retention {out['logit_retention_trigger_over_nontrigger']}", flush=True)
    print(f"pred_0 sane {out['pred_0_sane']} | pred_a content-computation-independent {out['pred_a_content_computation_independent']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
