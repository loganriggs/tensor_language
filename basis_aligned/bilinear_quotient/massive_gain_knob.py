"""MASSIVE GAIN KNOB -- turn the gain-control FINDING (680: the massive
dims' DC offset sets the rms-norm scale for the readout) into a falsifiable
CONTROL claim. If those few dims are genuinely the model's volume knob,
then SCALING their mean offset by a factor g should act like a temperature
/ confidence knob on the output distribution -- monotonically changing the
output entropy -- while scaling RANDOM dims by the same factors should not.
This is a forward-looking 'does the mechanism buy us a control knob' test,
not a re-run of 680's removal.

Method: at the final residual (block 17), find the top-K massive dims by
per-dim RMS. For scale factors g in {0.5,0.8,1.0,1.25,1.6}, multiply those
dims' value by g (scaling their large DC offset, hence the rms-norm scale)
and measure mean output entropy and CE. Compare to scaling K random dims.

REGISTERED PREDICTIONS:
  (0) SANITY: g=1.0 reproduces baseline entropy/CE;
  (a) MONOTONE GAIN KNOB: scaling the massive dims changes output entropy
      MONOTONICALLY with g (larger offset -> larger rms-norm denominator ->
      smaller logits -> higher entropy; so entropy increases with g), a
      clear temperature-like effect (entropy range across g > 0.15 nats);
  (b) report entropy and CE vs g for massive dims and random dims;
  NULL/CONTROL: scaling K RANDOM dims by the same g produces a much smaller
      entropy change (range < 1/3 of the massive-dim range) -- the knob is
      specific to the massive dims."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'massive_gain_knob_results.json'
NFRESH = 24
K = 8
GS = [0.5, 0.8, 1.0, 1.25, 1.6]


@torch.no_grad()
def run(fresh, dims, g):
    """Forward all rows, scaling residual dims `dims` by g just before the
    final rms-norm + head. Return mean entropy and mean CE."""
    ent_sum = 0.0; ce_sum = 0.0; n = 0
    dims_t = torch.tensor(dims, device=DEV)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        x = x.clone()
        x[..., dims_t] = x[..., dims_t] * g
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1)
        p = lp.exp()
        ent = -(p * lp).sum(-1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean')
        ent_sum += float(ent.mean()) * idx.shape[0]; ce_sum += float(ce) * idx.shape[0]
        n += idx.shape[0]
    return ent_sum / n, ce_sum / n


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    # find top-K massive dims by final-residual RMS
    ss = torch.zeros(D, dtype=torch.float64); n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        ss += (x.float() ** 2).reshape(-1, D).sum(0).double().cpu(); n += idx.numel()
    rms = np.sqrt((ss / n).numpy())
    massive = np.argsort(-rms)[:K].tolist()
    rng = np.random.default_rng(0)
    rand = rng.choice(np.argsort(-rms)[K:].tolist(), size=K, replace=False).tolist()
    print(f'massive dims: {sorted(massive)}', flush=True)

    res = {'massive': {}, 'random': {}}
    for label, dims in [('massive', massive), ('random', rand)]:
        for g in GS:
            e, c = run(fresh, dims, g)
            res[label][g] = [round(e, 4), round(c, 4)]
            print(f'{label:7s} g={g:<4}: entropy {e:.4f}  CE {c:.4f}', flush=True)

    ent_m = [res['massive'][g][0] for g in GS]
    ent_r = [res['random'][g][0] for g in GS]
    rng_m = max(ent_m) - min(ent_m); rng_r = max(ent_r) - min(ent_r)
    # monotone in g?
    mono = all(ent_m[i] <= ent_m[i + 1] for i in range(len(ent_m) - 1)) or \
           all(ent_m[i] >= ent_m[i + 1] for i in range(len(ent_m) - 1))
    base_ok = abs(res['massive'][1.0][1] - res['random'][1.0][1]) < 1e-3

    pa = mono and rng_m > 0.15
    null_ok = rng_r < rng_m / 3.0
    print(f'\n(0) g=1 baseline consistent: {base_ok}', flush=True)
    print(f'(a) monotone gain knob (entropy range {rng_m:.3f}>0.15, mono {mono}): {pa}',
          flush=True)
    print(f'NULL random-dim entropy range {rng_r:.3f} < massive/3 '
          f'({rng_m/3:.3f}): {null_ok}', flush=True)

    out = {'massive_dims': sorted(massive), 'entropy_ce_by_g': res,
           'massive_entropy_range': round(rng_m, 4), 'random_entropy_range': round(rng_r, 4),
           'monotone': bool(mono), 'pred_0': bool(base_ok), 'pred_a_gain_knob': bool(pa),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
