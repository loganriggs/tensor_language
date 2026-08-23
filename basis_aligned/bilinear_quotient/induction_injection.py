"""GENERATIVE validation of the INDUCTION mechanism (§877), the THIRD mechanism distinct from the two machines
(grammar/content). Induction = copy the token that FOLLOWED the current token last time it appeared (A B ... A -> B).
Inject a bigram "A B" at an early position, set the LAST fed token to A, and measure whether the model predicts B at
the query. A wrong-source control ("A C" early, then A -> should predict C not B) isolates position-specific
induction from generic priming.

REGISTERED PREDICTIONS:
  (0) NULL: without the earlier bigram (just A at the end), P(B) is at its baseline (no reason to predict B).
  (a) INDUCTION: injecting "A B" early and A at the end RAISES logP(B) at the query far above baseline -> the model
      copies the earlier successor B (induction), long-range (bigram at pos 3, query at 150);
  (b) WRONG-SOURCE CONTROL: injecting "A C" early (A followed by C != B) and A at the end does NOT raise logP(B) --
      induction copies the token that actually followed A, not B; confirms it is position-specific copy, not priming.
  Report ΔlogP(B) for induction vs wrong-source control."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_injection_results.json'
NEVAL = 200; SEQ = 256; PA = 3; PB = 4; QUERY = 150; NTRIALS = 24


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def measure(blocks, A, B, targetB, inject):
    # inject in {'none','AB','AC'}; return mean logP(targetB) at query
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); idx = bb[:, :QUERY].contiguous().clone()
        idx[:, QUERY-1] = A
        if inject == 'AB': idx[:, PA] = A; idx[:, PB] = B
        elif inject == 'AC': idx[:, PA] = A; idx[:, PB] = targetB  # targetB used as the WRONG successor C here
        lp = F.log_softmax(forward_logits(idx).float()[:, -1], -1)
        tot += float(lp[:, B].sum()); n += idx.shape[0]
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    # pick mid-frequency single-token pairs A,B and a distinct C, from tokens present in the corpus
    uniq = np.unique(rows.cpu().numpy().reshape(-1)); rng = np.random.RandomState(0)
    pool = [int(t) for t in uniq if 100 < int(t) < V]  # avoid very common low-id tokens
    rng.shuffle(pool)
    trials = []
    it = iter(pool)
    for _ in range(NTRIALS):
        try:
            A = next(it); B = next(it); C = next(it)
            trials.append((A, B, C))
        except StopIteration:
            break
    ind = []; nul = []; ctl = []
    for A, B, C in trials:
        # induction: earlier A B, then A -> measure logP(B)
        lp_induct = measure(rows, A, B, C, 'AB')       # inject A,B ; measure logP(B)
        lp_null = measure(rows, A, B, C, 'none')       # only A at end ; measure logP(B)
        lp_ctrl = measure(rows, A, B, C, 'AC')         # inject A,C ; measure logP(B) (should stay ~null)
        ind.append(lp_induct); nul.append(lp_null); ctl.append(lp_ctrl)
    ind = np.array(ind); nul = np.array(nul); ctl = np.array(ctl)
    out = {'n_trials': len(trials),
           'dlogpB_induction': round(float(np.mean(ind - nul)), 4),
           'dlogpB_wrong_source_control': round(float(np.mean(ctl - nul)), 4),
           'logpB_null_mean': round(float(np.mean(nul)), 4)}
    out['pred_a_induction'] = bool(out['dlogpB_induction'] > 0.5)
    out['pred_b_control_null'] = bool(abs(out['dlogpB_wrong_source_control']) < 0.3 * out['dlogpB_induction'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ΔlogP(B): induction {out['dlogpB_induction']} | wrong-source control {out['dlogpB_wrong_source_control']} | baseline logP(B) {out['logpB_null_mean']}", flush=True)
    print(f"pred_a induction {out['pred_a_induction']} | pred_b control-null {out['pred_b_control_null']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
