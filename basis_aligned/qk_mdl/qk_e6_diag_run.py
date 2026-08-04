"""E6 TRAINING DIAGNOSTICS (fresh batch-16 single-epoch protocol; motivated by
the fresh-data repricing: E0b costs +0.342 vs E0a, Muon helps vanilla (-0.094)
but HURTS slots+lasso (+0.076), and V11's trained decoders shrink far below
pass-through -- the questions are where in training these costs arise).

Runs 2,000-step INSTRUMENTED replicas of four arms -- E0a (vanilla), E0b
(slots+lasso base), V11 (readout decoders + decoder lasso), V13r1 (rank-1 edge
adapters) -- at the family lr with the schedule computed for the full 8250-step
horizon (so the truncated dynamics match the real runs). Every 100 steps
(train_diag in qk_e_common, a hook layer that is off for every other runner):

 1. global + per-group gradient norms (embedding / attention matrices /
    Left,Right,Down / decoders-adapters / biases), grad-clip hit rate,
    loss-spike count. (The "read-coefficient matrices" of the spec ARE columns
    of the attention/MLP input matrices in this architecture; item 3 measures
    them via the read-matrix set directly.)
 2. update-to-weight ratio per group (lr x ||grad|| / ||w||), cosine of
    successive step gradients, half-batch gradient cosine (batch 16 split into
    two 8s, CE-only gradients -- per-arm gradient-noise proxy).
 3. loss decomposition: CE vs penalty share; on the read matrices, the
    penalty-gradient norm vs the CE-gradient norm (is the lasso drowning the
    data signal?), with a step-0 known-answer check that the subtraction
    decomposition matches a directly computed CE-only gradient.
 4. saturation/scale: fraction of |logits| > 25 (tanh cap), per-layer entry
    norms, per-slot write-norm distribution + collapsed-slot count,
    decoder/adapter norm trajectories (V11: does the lasso shrink the readout
    pass-through EARLY, before the modules learn what to say?).
 5. sparsity dynamics: fraction of the 2016 read groups with norm < 1e-3.

Deliverable: per-arm series in qk_e6.json plus a comparison summary -- for
each of E0b/V11/V13r1, which scalar diagnostics diverge from E0a and from
which step. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E
from qk_e_common import Q, W, torch
import qk_v13_common as X

JP = E.jpath('qk_e6.json')
DIAG_STEPS = 3 if E.SMOKE else 2000

ARMS = (
    ('E0a', E.make_e0a, 0.0),
    ('E0b', E.make_e0b, E.GC),
    ('V11', lambda: W.make_v11('V11', dec_lasso=True, cls=W.V11Route), E.GC),
    ('V13r1', lambda: X.make_v13('V13r1', 1), E.GC),
)

# scalar series compared against the E0a reference (name, divergence rule)
SCALARS = ('grad_norm_preclip', 'clip_rate_window', 'halfbatch_grad_cos',
           'succ_grad_cos', 'logit_sat_frac_gt25', 'upd_ratio_embedding',
           'upd_ratio_attn', 'upd_ratio_mlp', 'ce')
COS_KEYS = {'halfbatch_grad_cos', 'succ_grad_cos'}


def first_divergence(ref, arm, key):
    """First diag step where the arm deviates from E0a: ratio off by >50%
    (absolute difference > 0.2 for cosines / rates near zero)."""
    ref_by = {d['step']: d.get(key) for d in ref}
    for d in arm:
        r = ref_by.get(d['step'])
        v = d.get(key)
        if r is None or v is None:
            continue
        if key in COS_KEYS or abs(r) < 1e-3:
            if abs(v - r) > 0.2:
                return d['step'], v, r
        elif abs(v / r - 1.0) > 0.5:
            return d['step'], v, r
    return None


def summarize(out):
    ref = out['E0a']['series']
    summary = {}
    for key, _, _ in ARMS[1:]:
        arm = out[key]['series']
        div = {}
        for sk in SCALARS:
            hit = first_divergence(ref, arm, sk)
            last_a = next((d.get(sk) for d in reversed(arm)
                           if d.get(sk) is not None), None)
            last_r = next((d.get(sk) for d in reversed(ref)
                           if d.get(sk) is not None), None)
            div[sk] = {'first_divergence_step': (hit[0] if hit else None),
                       'at_divergence': ([hit[1], hit[2]] if hit else None),
                       'final_arm_vs_e0a': [last_a, last_r]}
        # arm-specific trajectories
        picks = [d for d in arm if d['step'] in
                 (0, 100, 500, 1000, arm[-1]['step'])]
        div['penalty_share_trajectory'] = [
            [d['step'], d.get('penalty_share_of_loss')] for d in picks]
        div['read_pen_over_ce_trajectory'] = [
            [d['step'], d.get('read_pen_over_ce')] for d in picks]
        div['read_groups_frac_below_1e-3_trajectory'] = [
            [d['step'], d.get('read_groups_frac_below_1e-3')] for d in picks]
        div['collapsed_slots_trajectory'] = [
            [d['step'], d.get('collapsed_slots_below_1e-3')] for d in picks]
        if any('decoder_fro_norms' in d for d in arm):
            div['decoder_norm_median_trajectory'] = [
                [d['step'], round(float(torch.tensor(
                    d['decoder_fro_norms']).median()), 3)]
                for d in picks if 'decoder_fro_norms' in d]
        if any('adapter_product_norms' in d for d in arm):
            div['adapter_norm_median_trajectory'] = [
                [d['step'], round(float(torch.tensor(
                    d['adapter_product_norms']).median()), 4)]
                for d in picks if 'adapter_product_norms' in d]
        summary[key] = div
    return summary


if __name__ == '__main__':
    E.setup()
    lr = E.get_lr()
    for key, factory, gc in ARMS:
        out = E.loadj(JP)
        if key in out:
            print(f"E6 {key}: already done -- skip", flush=True)
            continue
        print(f"==== E6 instrumented run {key} (lr {lr}, gc {gc}, "
              f"{DIAG_STEPS} steps) ====", flush=True)
        res = E.train_diag(lr, gc, DIAG_STEPS, factory)
        E.merge(JP, key, res)
    out = E.loadj(JP)
    if all(k in out for k, _, _ in ARMS):
        E.merge(JP, 'comparison_summary', summarize(out))
        E.merge(JP, 'protocol', {
            'steps': DIAG_STEPS, 'lr': lr, 'diag_every': 2 if E.SMOKE else 100,
            'schedule': 'warmup 250 + cosine computed for the full 8250-step '
                        'horizon, run truncated',
            'halfbatch_note': 'CE-only gradients (the penalty is deterministic '
                              'and would inflate the cosine)',
            'succ_cos_note': 'combined CE+penalty training gradients (what the '
                             'optimizer actually sees)'})
    print('e6 diag run done', flush=True)
