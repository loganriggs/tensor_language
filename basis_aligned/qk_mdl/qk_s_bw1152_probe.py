"""Wiring readability for the w1152 bandwidth-reinvestment arms.

The branch-point verdict has two halves. qk_s_bw1152_run.py measures the CE
half; this measures the readability half, because the program's retrain rule
(BRAINSTORM_STATE.md) admits an architecture only with "CE better than the
recipe at readability-tie-or-better".

WHAT IS COMPARABLE, AND WHAT IS NOT. The scale session's stored wiring
numbers were all computed by the same light probe on the same rows -- the
documented cooc SUBSTITUTE, fresh34k[5500:6000], bound as Q.HELD/E.OLD_HELD
at import, of which the first ABL_N = 96 sequences carry the causal graph.
This box reconstructs that substitute byte-identically (fresh34k rows
[0:6000]), so the PLAIN Spearman computed here sits on exactly the same
footing as the stored control numbers:

    combo3e5loss (the readable recipe, matched control for bw3e5)
        plain 0.6007 all / 0.5728 effectual / top-10 0.2, 133 of 156 edges
    combo1e4loss (matched control for bw1e4)
        plain 0.7765 all / 0.6841 effectual / top-10 0.2, 121 of 156 edges

The covariance-composed metric (E17/E18), which is the standard REPORTED
metric at w264, cannot be computed for those controls here: composition
needs a forward pass through their weights and no w1152 checkpoint survived
onto this box (checkpoints were never in git). So cov-composed is reported
for the bandwidth arms only, as a within-arm supplement, and the
cross-architecture readability claim rests on PLAIN Spearman. Stated
explicitly rather than quietly compared across metrics.

Per arm this writes: the full per-edge tables (causal consumption and each
weight-support vector, 156 entries -- the standing logging requirement, so
a successor can bootstrap CIs per reviewer-2 R1 without re-running), the
consumption matrix, plain/cov-composed/cov-composed-readout-globalnorm
agreement, and an untrained-init null row (the same probe on an untrained
model of the same architecture, which should score near zero).

Reuses the gate-validated generalized variable-slot-dim machinery from
qk_e18_probe_upgrades verbatim (gen_consumption / wpairs / gen_gram_table /
score / composed_tables) at dims = [65]*24. Idempotent per arm on the JSON
key; run after training so it never contends with the trainer for VRAM.
"""
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import time

import numpy as np

import qk_tokenline_train as Q

Q.gpu_guard = lambda *a, **k: None

import qk_s_gate_run as G
import qk_w1152_train as W2
import qk_e_common as E
import qk_v9_common as C
import qk_deeproute_train_2 as R2
import qk_e17_composed_wiring as E17
import qk_e18_probe_upgrades as E18U
import qk_s_bw1152_run as BW
from qk_e_common import DEPTH, torch

# the E17 import (pulled in by E18U) sets E.DEV='cpu' for its weight-only use
E.DEV = 'cuda'
E18U.DEV = 'cuda'

ARMS = sys.argv[1:] or ['bw1e4', 'bw3e5']

CONTROL_REFERENCE = {
    'bw1e4': {'stem': 'qk_s_w1152_combo1e4loss', 'label': 'combo1e4loss',
              'plain_all': 0.7765, 'plain_effectual': 0.6841,
              'top10': 0.2, 'n_effectual': 121},
    'bw3e5': {'stem': 'qk_s_w1152_combo3e5loss', 'label': 'combo3e5loss',
              'plain_all': 0.6007, 'plain_effectual': 0.5728,
              'top10': 0.2, 'n_effectual': 133}}

CAVEAT = ('probe eval data = the documented cooc substitute '
          'fresh34k[5500:6000] (first 96 seqs causal), byte-identical to the '
          'rows every stored w1152 light_probe used; covariance pass on '
          'fresh34k[33000:33300]')


def load_arm(stem, slot):
    """Trained model of the bandwidth architecture at the arm's slot dim."""
    ck = torch.load(os.path.join(G.OUT_DIR, f'{stem}.pt'), map_location='cpu',
                    weights_only=False)
    m = BW.make_bw(s=slot, std=BW.write_std())
    missing, unexpected = m.load_state_dict(ck['state_dict'], strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m.to(E.DEV).eval().float(), ck


def per_edge_tables(wp, cau, vecs):
    return [{'consumer': li, 'source': R2.stream_name(si),
             'source_index': si, 'causal_dce': round(cau[k], 6),
             'plain': round(vecs['plain'][k], 6),
             'cov_composed': round(vecs['cov'][k], 6),
             'cov_composed_readout_globalnorm': round(vecs['cov_ro'][k], 6)}
            for k, (li, si) in enumerate(wp)]


def null_row(slot, wp, cau):
    """Untrained-init model of the same architecture, same probe: the floor."""
    m = BW.make_bw(s=slot, std=BW.write_std()).to(E.DEV).eval().float()
    Gt = E18U.gen_gram_table(m, [slot] * 24)
    plain = E18U.score(Gt, wp)
    eff_idx = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
    rec = E17.agreement(plain, cau, eff_idx)
    del m, Gt
    torch.cuda.empty_cache()
    return rec


def probe(arm):
    stem = f'qk_s_w1152_{arm}'
    jp = os.path.join(G.OUT_DIR, f'{stem}.json')
    out = G.loadj(jp)
    if 'light_probe' in out and 'composed_wiring' in out:
        print(f"{arm}: probes already present -- skip", flush=True)
        return
    ckpt = os.path.join(G.OUT_DIR, f'{stem}.pt')
    if not os.path.exists(ckpt):
        print(f"{arm}: no checkpoint at {ckpt} -- skip", flush=True)
        return
    slot = out.get('controls', {}).get('slot')
    assert slot, f'{arm}: controls block missing the solved slot dim'
    dims = [slot] * 24
    print(f"==== probing {arm} (slot {slot}, stream {24 * slot}) ====",
          flush=True)

    m, ck = load_arm(stem, slot)
    Ws = m.wte.weight.shape[1]
    wp = E18U.wpairs(m, dims)

    t0 = time.time()
    base, dce = E18U.gen_consumption(m, Ws)
    cau = [dce[li][si] for li, si in wp]
    print(f"  consumption done in {time.time() - t0:.0f}s (base CE {base:.5f})",
          flush=True)

    tables, meta, vecs = E18U.composed_tables(m, dims, cau, wp, E.DEV,
                                              remnant=False)
    eff_idx = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]

    ref = CONTROL_REFERENCE[arm]
    out['light_probe'] = {
        'base_ce_fp32_abl_oldheld': round(base, 5),
        'wiring_n_pairs': len(wp),
        'wiring_spearman_all': tables['plain']['spearman_all'],
        'wiring_n_effectual': len(eff_idx),
        'wiring_spearman_effectual': tables['plain']['spearman_effectual'],
        'wiring_top10_precision': tables['plain']['top10_precision'],
        'consumption_matrix': {str(li): {str(si): round(v, 6)
                                         for si, v in row.items()}
                               for li, row in dce.items()},
        'caveat': CAVEAT}
    out['composed_wiring'] = {
        'tables': tables, 'meta': meta,
        'slot_dims': dims,
        'note': 'cov-composed is reported for this arm only; the w1152 '
                'recipe controls have no surviving checkpoint on this box, '
                'so the cross-architecture readability comparison uses PLAIN '
                'Spearman, which IS on the same rows and the same probe'}
    out['per_edge_table'] = per_edge_tables(wp, cau, vecs)
    out['readability_vs_matched_control'] = {
        'control': ref['label'],
        'control_plain_all': ref['plain_all'],
        'control_plain_effectual': ref['plain_effectual'],
        'control_top10': ref['top10'],
        'arm_plain_all': tables['plain']['spearman_all'],
        'arm_plain_effectual': tables['plain']['spearman_effectual'],
        'arm_top10': tables['plain']['top10_precision'],
        'delta_plain_all': round(tables['plain']['spearman_all']
                                 - ref['plain_all'], 4),
        'tie_note': 'reviewer-2 R1: with n=156 the Spearman SE is about 0.08, '
                    'so gaps below ~0.1 are TIES until bootstrapped'}
    G.savej(jp, out)

    del m
    torch.cuda.empty_cache()
    out['wiring_null_untrained_init'] = null_row(slot, wp, cau)
    G.savej(jp, out)

    p = tables['plain']
    c = tables['cov_composed']
    print(f"== {arm} PLAIN Spearman {p['spearman_all']} all / "
          f"{p['spearman_effectual']} effectual / top10 "
          f"{p['top10_precision']} ({len(eff_idx)} of {len(wp)} effectual)",
          flush=True)
    print(f"== {arm} COV-COMPOSED {c['spearman_all']} all / "
          f"{c['spearman_effectual']} effectual / top10 "
          f"{c['top10_precision']}", flush=True)
    print(f"== {arm} vs matched control {ref['label']} (plain, same rows): "
          f"{tables['plain']['spearman_all']:.4f} vs {ref['plain_all']} "
          f"-> {tables['plain']['spearman_all'] - ref['plain_all']:+.4f}",
          flush=True)
    print(f"== {arm} untrained-init null: "
          f"{out['wiring_null_untrained_init']}", flush=True)


def main():
    W2.patch_width(G.WIDTH)
    for arm in ARMS:
        BW.ARM = arm
        BW.CFG = {'bw1e4': dict(coeff=1e-4), 'bw3e5': dict(coeff=3e-5)}[arm]
        probe(arm)


if __name__ == '__main__':
    main()
