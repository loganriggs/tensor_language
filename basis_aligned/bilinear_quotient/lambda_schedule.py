"""LAMBDA SCHEDULE -- examine the last unexamined architectural
component: the per-block residual rescaling. Each block does
x = lambda0 * x + lambda1 * x0 before attention (x0 = the embedding). So
lambda0 rescales the running residual and lambda1 re-injects the
embedding, every block. Is this a systematic learned gain/decay schedule?

Read the lambdas[0], lambdas[1] from every block (they are learned
parameters, not data-dependent), report them by depth, and compute the
cumulative product of lambda0 (how much an early write is scaled down by
the output) and the total embedding re-injection.

REGISTERED PREDICTIONS:
  (0) SANITY: lambdas are readable per block;
  (a) SYSTEMATIC SCHEDULE: lambda0 and lambda1 vary smoothly/systematically
      with depth (not random per layer) -- e.g. lambda0 near or below 1
      (a decay), lambda1 controlling embedding re-injection;
  (b) report lambda0, lambda1 per layer and prod(lambda0) across depth
      (the report noted ~2e-4 over 12 layers -- verify the decay);
  NULL: n/a (reading learned params) -- but confirm the cumulative
      lambda0 product decays (an early writer is heavily attenuated by the
      output, consistent with the residual-rescaling correction)."""
import json, time, torch
import numpy as np
from bilin18_joint_removal import m


@torch.no_grad()
def main():
    t0 = time.time()
    NL = len(m.transformer.h)
    l0 = []; l1 = []
    for li in range(NL):
        lam = m.transformer.h[li].lambdas.detach().float().cpu().numpy()
        l0.append(float(lam[0])); l1.append(float(lam[1]))
    l0 = np.array(l0); l1 = np.array(l1)
    print('lambda0 (residual rescale) by layer:', flush=True)
    print('  ' + ' '.join(f'{v:.3f}' for v in l0), flush=True)
    print('lambda1 (embedding re-inject) by layer:', flush=True)
    print('  ' + ' '.join(f'{v:.3f}' for v in l1), flush=True)

    # cumulative product of lambda0 from layer k to the end (how much a
    # write at block k is scaled by the time it reaches the output)
    cum_to_end = {}
    for k in range(NL):
        p = float(np.prod(l0[k + 1:])) if k + 1 < NL else 1.0
        cum_to_end[k] = round(p, 6)
    print('\nprod(lambda0) from block k+1..end (attenuation of a block-k write):',
          flush=True)
    for k in [0, 4, 8, 12, 16]:
        print(f'  block {k}: x{cum_to_end[k]:.2e}', flush=True)

    p0 = True
    # systematic = low variance of consecutive differences relative to range, or monotone-ish
    smooth = float(np.std(np.diff(l0)))
    pa = smooth < 0.5 * (l0.max() - l0.min() + 1e-9) or (l0.max() - l0.min()) < 0.3
    total_atten = cum_to_end[0]
    print(f'\n(a) systematic schedule (l0 smooth): {pa} (l0 range '
          f'{l0.min():.2f}-{l0.max():.2f}, step-std {smooth:.3f})', flush=True)
    print(f'    early-write attenuation to output: {total_atten:.2e}', flush=True)

    out = {'lambda0': [round(v, 4) for v in l0.tolist()],
           'lambda1': [round(v, 4) for v in l1.tolist()],
           'cum_lambda0_to_end': cum_to_end,
           'lambda0_range': [round(float(l0.min()), 4), round(float(l0.max()), 4)],
           'lambda1_range': [round(float(l1.min()), 4), round(float(l1.max()), 4)],
           'block0_attenuation': total_atten,
           'pred_0': bool(p0), 'pred_a_systematic': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'lambda_schedule_results.json', 'w'), indent=1)
    print(f'\nwrote results ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
