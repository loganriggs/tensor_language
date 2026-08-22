"""IS THE CALIBRATION ASYMMETRY UNIVERSAL? (generality check for §889). §889: bilin18 is well-calibrated on
average but OVER-CAUTIOUS on easy tokens (inductable/seen gap H−CE ~+0.85) and OVER-CONFIDENT on first-mentions
(gap −0.59). Is this a bilin18 quirk or a general LM property? Run the same entropy/CE-by-position-type
calibration on GPT-2 and GPT-2-large (same GPT-2 BPE, same FineWeb tokens).

CAVEAT: GPT-2 is WebText-trained (slightly OOD on FineWeb) so absolute CE/entropy are a bit inflated; the
SIGN and shape of the per-bucket calibration gap is the test.

REGISTERED PREDICTIONS:
  (0) SANITY: each model roughly calibrated overall (|H−CE| small);
  (a) UNIVERSAL ASYMMETRY: both GPT-2 models show the same SIGN pattern — positive gap (over-cautious) on
      inductable/seen-other and NEGATIVE gap (over-confident) on first-mention -> the calibration asymmetry is
      a general LM property, not bilin18-specific;
  (b) if the sign pattern differs, the asymmetry is model-specific (report plainly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'calibration_crossmodel_results.json'
NEVAL = 200; SEQ = 256; MODELS = ['gpt2', 'gpt2-large']


@torch.no_grad()
def calib(mdl, blocks, inductable, firstment, other):
    H = []; CE = []
    for i in range(0, blocks.shape[0], 4):
        bb = blocks[i:i+4].to(DEV); lg = mdl(bb[:, :-1]).logits.float(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(lg, -1); p = lp.exp()
        H.append((-(p*lp).sum(-1)).cpu().numpy().reshape(-1))
        CE.append((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).cpu().numpy().reshape(-1))
    H = np.concatenate(H); CE = np.concatenate(CE)
    r = {'overall': {'entropy': round(float(H.mean()), 3), 'ce': round(float(CE.mean()), 3)},
         'entropy_loss_corr': round(float(np.corrcoef(H, CE)[0, 1]), 3)}
    for name, mk in [('inductable', inductable), ('first_mention', firstment), ('seen_other', other)]:
        r[name] = {'entropy': round(float(H[mk].mean()), 3), 'ce': round(float(CE[mk].mean()), 3),
                   'calib_gap_H_minus_CE': round(float(H[mk].mean()-CE[mk].mean()), 3)}
    return r


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    inductable = np.zeros((nb, SEQ-1), dtype=bool); firstment = np.zeros((nb, SEQ-1), dtype=bool)
    for r in range(nb):
        seen_tok = set(); seen_big = {}
        for pp in range(SEQ-1):
            cur = int(S[r, pp]); nxt = int(S[r, pp+1])
            firstment[r, pp] = nxt not in seen_tok
            if cur in seen_big and seen_big[cur] == nxt: inductable[r, pp] = True
            seen_big[cur] = nxt; seen_tok.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable; other = ~inductable & ~firstment
    out = {'bilin18_ref': {'inductable_gap': 0.866, 'seen_other_gap': 0.845, 'first_mention_gap': -0.591}, 'models': {}}
    for mid in MODELS:
        print(f"loading {mid}...", flush=True)
        mdl = AutoModelForCausalLM.from_pretrained(mid).to(DEV).eval()
        r = calib(mdl, blocks, inductable, firstment, other); del mdl; torch.cuda.empty_cache()
        out['models'][mid] = r
        print(f"{mid}: overall H {r['overall']['entropy']} vs CE {r['overall']['ce']} | gaps: inductable {r['inductable']['calib_gap_H_minus_CE']} | seen {r['seen_other']['calib_gap_H_minus_CE']} | first-mention {r['first_mention']['calib_gap_H_minus_CE']}", flush=True)
    def signs_ok(r): return r['inductable']['calib_gap_H_minus_CE'] > 0 and r['seen_other']['calib_gap_H_minus_CE'] > 0 and r['first_mention']['calib_gap_H_minus_CE'] < 0
    out['pred_a_universal_asymmetry'] = bool(all(signs_ok(out['models'][mid]) for mid in MODELS))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) calibration asymmetry universal (over-cautious easy, over-confident first-mention): {out['pred_a_universal_asymmetry']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
