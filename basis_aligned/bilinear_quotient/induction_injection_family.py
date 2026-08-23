"""Is the GENERATIVE induction effect (§1025) ARCHITECTURE-GENERAL? Run the AB...A->B copy test across the family
(bilin18, bilin12, swiglu18): inject a novel bigram "A B" early, present A at the end, measure ΔlogP(B) at a distant
query, with the wrong-source "A C" control. Completes the generative x mechanism x architecture matrix (content was
generalized in §1021; grammar is trivially local everywhere; induction here).

REGISTERED PREDICTIONS:
  (0) CONTROL: the wrong-source ("A C" early) ΔlogP(B) is near-null in every model (position-specific copy).
  (a) INDUCTION ARCHITECTURE-GENERAL: injecting "A B" early + A at the end raises logP(B) large (> ~2 nats) in EVERY
      model incl swiglu18, >> the wrong-source control -> induction (long-range copy) is architecture-general;
  (b) report per-model induction ΔlogP(B) and control ΔlogP(B)."""
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
OUT = PT + 'induction_injection_family_results.json'
NEVAL = 160; SEQ = 256; PA = 3; PB = 4; QUERY = 150; NTRIALS = 20


def forward_logits(mdl, idx, Dm):
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(mdl.lm_head(F.rms_norm(x, (Dm,)))/30.0)


@torch.no_grad()
def measure(mdl, blocks, A, B, targetB, inject, Dm):
    # inject in {'none','AB','AC'}; return mean logP(B) at query
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); idx = bb[:, :QUERY].contiguous().clone()
        idx[:, QUERY-1] = A
        if inject == 'AB': idx[:, PA] = A; idx[:, PB] = B
        elif inject == 'AC': idx[:, PA] = A; idx[:, PB] = targetB
        lp = F.log_softmax(forward_logits(mdl, idx, Dm).float()[:, -1], -1)
        tot += float(lp[:, B].sum()); n += idx.shape[0]
    return tot / max(n, 1)


@torch.no_grad()
def run_model(mdl, blocks, trials, tag):
    Dm = mdl.transformer.wte.weight.shape[1]; V = int(mdl.lm_head.weight.shape[0])
    ind = []; nul = []; ctl = []
    for A, B, C in trials:
        if A >= V or B >= V or C >= V: continue
        ind.append(measure(mdl, blocks, A, B, C, 'AB', Dm))
        nul.append(measure(mdl, blocks, A, B, C, 'none', Dm))
        ctl.append(measure(mdl, blocks, A, B, C, 'AC', Dm))
    ind = np.array(ind); nul = np.array(nul); ctl = np.array(ctl)
    res = {'dlogpB_induction': round(float(np.mean(ind - nul)), 4), 'dlogpB_control': round(float(np.mean(ctl - nul)), 4)}
    print(f"{tag}: induction ΔlogP(B) {res['dlogpB_induction']} | control {res['dlogpB_control']}", flush=True)
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    uniq = np.unique(rows.cpu().numpy().reshape(-1)); rng = np.random.RandomState(0)
    pool = [int(t) for t in uniq if 100 < int(t) < 50000]; rng.shuffle(pool)
    trials = []; it = iter(pool)
    for _ in range(NTRIALS):
        try: trials.append((next(it), next(it), next(it)))
        except StopIteration: break
    out = {'models': {}}
    out['models']['bilin18'] = run_model(BILIN, rows, trials, 'bilin18')
    for short in ['bilin12', 'swiglu18']:
        try:
            mdl, cfg = load_elriggs(short); mdl = mdl.to(DEV).eval()
            out['models'][short] = run_model(mdl, rows, trials, short); del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = [k for k in out['models'] if 'dlogpB_induction' in out['models'][k]]
    out['pred_0_control_null'] = bool(all(out['models'][k]['dlogpB_control'] < 0.3*out['models'][k]['dlogpB_induction'] for k in ok))
    out['pred_a_induction_general'] = bool(len(ok) >= 2 and all(out['models'][k]['dlogpB_induction'] > 2 for k in ok))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_0 control-null {out['pred_0_control_null']} | pred_a induction-architecture-general {out['pred_a_induction_general']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
