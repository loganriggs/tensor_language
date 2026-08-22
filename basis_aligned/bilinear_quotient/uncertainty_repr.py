"""DOES THE MODEL REPRESENT ITS OWN UNCERTAINTY, and where is it computed? (new layer-relevant question the
loss budget raises). §879: loss splits by position type — inductable (copy, cheap) / seen-other / first-mention
(new word, dear). Does the model KNOW which situation it is in — is its predictive ENTROPY calibrated to the
position type, does entropy track per-token loss ("knows when it doesn't know"), and is the "this is a
first-mention" situation DECODABLE from intermediate layers (i.e. the model computes its own uncertainty)?

Method: per next-token position compute predictive entropy H = -sum p log p, actual loss (CE), top-1 prob.
Bucket by position type (inductable / first-mention / seen-other). Report per-bucket H, CE, top-1, and the
calibration gap (H - CE: <0 = overconfident). Correlate H with per-token loss (does entropy predict
difficulty). Then decode the binary "is-first-mention" situation from each layer's residual (ridge probe) to
locate where the uncertainty-situation is represented. Controls: shuffled-label decode (chance); overall
H vs overall CE (global calibration).

REGISTERED PREDICTIONS:
  (0) SANITY: overall H ~ overall CE (a trained LM is roughly calibrated on average);
  (a) UNCERTAINTY IS REPRESENTED: predictive entropy TRACKS position type (low on inductable, high on
      first-mention, monotone with the bucket's loss) AND correlates with per-token loss (Pearson > 0.4) ->
      the model represents how uncertain it should be; AND "is-first-mention" is DECODABLE well above chance
      from mid/late layers -> the model computes its own uncertainty situation internally;
  (b) if entropy is flat across buckets or uncorrelated with loss, the model does NOT represent its
      uncertainty by situation (report plainly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'uncertainty_repr_results.json'
NEVAL = 200; SEQ = 256; NLAYER = 18


def bilin_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def decode_acc(F_, y, ncls, seed=0):
    n = F_.shape[0]; rng = np.random.RandomState(seed); idx = rng.permutation(n)
    ntr = int(0.7*n); tr, te = idx[:ntr], idx[ntr:]
    Ft = F_[tr]; Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = Ft.T @ Ft + 1e2*torch.eye(Ft.shape[1], device=DEV); Wp = torch.linalg.solve(A, Ft.T @ Y)
    pred = (F_[te] @ Wp).argmax(1).cpu().numpy()
    return float((pred == y[te]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    # per-position entropy, loss, top1; capture all layer residuals for the decode
    caps = {L: [] for L in range(NLAYER)}; hs = []
    for L in range(NLAYER):
        def mk(L):
            def h(mo, i_, o_): caps[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mk(L)))
    H = []; CE = []; TOP1 = []
    for i in range(0, nb, 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = bilin_logits(idx).float(); lp = F.log_softmax(lg, -1); p = lp.exp()
        H.append((-(p*lp).sum(-1)).cpu().numpy().reshape(-1))
        CE.append((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).cpu().numpy().reshape(-1))
        TOP1.append(p.max(-1).values.cpu().numpy().reshape(-1))
    for h in hs: h.remove()
    reps = {L: torch.cat(caps[L], 0) for L in range(NLAYER)}   # (nb*(SEQ-1)? no: nb*(SEQ-1)) -- idx is SEQ-1 long
    H = np.concatenate(H); CE = np.concatenate(CE); TOP1 = np.concatenate(TOP1)
    # buckets
    inductable = np.zeros((nb, SEQ-1), dtype=bool); firstment = np.zeros((nb, SEQ-1), dtype=bool)
    for r in range(nb):
        seen_tok = set(); seen_big = {}
        for pp in range(SEQ-1):
            cur = int(S[r, pp]); nxt = int(S[r, pp+1])
            firstment[r, pp] = nxt not in seen_tok
            if cur in seen_big and seen_big[cur] == nxt: inductable[r, pp] = True
            seen_big[cur] = nxt; seen_tok.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable; other = ~inductable & ~firstment
    buckets = {}
    for name, mk in [('inductable', inductable), ('first_mention', firstment), ('seen_other', other)]:
        buckets[name] = {'entropy': round(float(H[mk].mean()), 3), 'ce': round(float(CE[mk].mean()), 3),
                         'top1_prob': round(float(TOP1[mk].mean()), 3), 'calib_gap_H_minus_CE': round(float(H[mk].mean()-CE[mk].mean()), 3)}
    # entropy-loss correlation (does the model know when it doesn't know)
    corr = float(np.corrcoef(H, CE)[0, 1])
    # decode "is first-mention" from each layer (uncertainty situation)
    y = firstment.astype(np.int64); rng = np.random.RandomState(0); ysh = y.copy(); rng.shuffle(ysh)
    layer_decode = {}
    for L in [0, 2, 5, 8, 11, 14, 17]:
        layer_decode[f'L{L}'] = round(decode_acc(reps[L], y, 2), 3)
    shuf_decode = round(decode_acc(reps[17], ysh, 2), 3)
    fm_base = round(float(1 - y.mean()), 3)  # majority baseline (predict not-first-mention)
    out = {'overall_entropy': round(float(H.mean()), 3), 'overall_ce': round(float(CE.mean()), 3),
           'entropy_loss_corr': round(corr, 3), 'buckets': buckets,
           'first_mention_decode_by_layer': layer_decode, 'decode_majority_baseline': fm_base,
           'shuffled_decode': shuf_decode, 'runtime_s': round(time.time()-t0, 1)}
    ent = [buckets[b]['entropy'] for b in ['inductable', 'seen_other', 'first_mention']]
    out['pred_a_uncertainty_represented'] = bool(ent[0] < ent[1] < ent[2] and corr > 0.4 and
                                                 max(layer_decode.values()) > fm_base + 0.1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"overall: entropy {out['overall_entropy']} vs CE {out['overall_ce']} (calibration) | entropy-loss corr {corr:.3f}", flush=True)
    for b in ['inductable', 'seen_other', 'first_mention']:
        z = buckets[b]; print(f"  {b:>13}: entropy {z['entropy']} | CE {z['ce']} | top1 {z['top1_prob']} | gap {z['calib_gap_H_minus_CE']}", flush=True)
    print(f"is-first-mention decode by layer: {layer_decode} (majority {fm_base}, shuffled {shuf_decode})", flush=True)
    print(f"(a) uncertainty represented + computed internally: {out['pred_a_uncertainty_represented']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
