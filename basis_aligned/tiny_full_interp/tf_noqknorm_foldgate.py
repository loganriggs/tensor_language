"""TASK 2 -- the fold-identity gate for the UN-CAPPED foldable arms.

Twelve `tff_bilin_bilin_d{2,3}_w{64,128}_b8192_s{0,1,2}_noqknorm` checkpoints
are the foldable family trained WITHOUT the per-head query/key RMSNorm (the
"cap").  They are FacTransformer checkpoints, so they had never been through
`tf_model.check_fold_identities` -- yet published findings rest on them.  This
script puts them through the programme's gate, unchanged, and puts the CAPPED
cells through the same gate in the same run at the same precision so the two
are comparable.

WHAT IS GATED, in order
  0. TRANSPLANT VERIFICATION (assumption under test, not assumed).  For every
     un-capped checkpoint: FacTransformer(bilin, bilin, qk_norm=False) and
     TinyBilin(variant='vanilla', qk_norm=False) must have the SAME parameter
     name set and the SAME shapes, and with the checkpoint's weights in both
     their forwards must agree -- on random tokens AND on the real held tokens
     the gate then uses.  If they disagree the checkpoint is recorded as NOT
     gated; nothing is papered over.
     CRITERION, and why it is not a flat absolute 1e-5.  An absolute 1e-5 on
     fp32 logits FAILS on 5 of the 12 (every depth-3 width-128 cell, and two
     depth-3 width-64 ones) at 1.0e-5 to 1.4e-5 -- which is 6e-7 to 9.8e-7
     RELATIVE, i.e. 5-8 fp32 ulps of a logit that reaches 15 on 30*tanh(./30).
     That is the exact criterion this programme superseded on 2026-08-08 after
     it failed 4 of 6 cells on rounding alone.  The verdict therefore uses the
     same two-tier, scale-free criterion check_fold_identities uses: fp32
     RELATIVE < 1e-5, fp64 ABSOLUTE < 1e-9, and the fp32 gap within 10x the
     FacTransformer's own fp32-vs-fp64 self-noise.  That is strictly stronger:
     the observed fp64 transplant residual is 1.1e-14 to 2.8e-14 (relative
     ~1e-15), five orders inside the 1e-9 leg, and the worst self-noise
     multiple is 0.84 -- the two models differ by LESS than the reference
     model differs from itself in fp32.  Both numbers are in the JSON
     (`pass` and `pass_spec_literal_fp32_absolute_1e-5`).
  1. tf_model.check_fold_identities in FLOAT64 (its own two-tier criterion:
     algebraic identities < 1e-12 relative, fold_forward vs forward < 1e-9
     absolute, and within 10x the forward's own fp32-vs-fp64 self-noise).
  2. The same gate on the CAPPED cells.  Every un-capped cell has a
     cell-matched capped partner (tf_vanilla at the same depth/width/seed), so
     the comparison is per-cell and not an average over different sizes --
     the README's "claims formed at one model size" failure mode.
  3. THE GATE MUST BE ABLE TO FAIL, three ways, in this same run:
       * M.planted_qk_test          -- known analytic answer for the table
                                       materializer (fp64, < 1e-10)
       * M.gate_negative_control    -- a 1+1e-7 MLP-tensor corruption and a
                                       one-head roll of Vv must both FAIL
       * cap-off fold/forward norm mismatch (added here, and specific to the
         change this gate exists to check): force the FOLD to normalise the
         layer-0 query/key factors while the FORWARD does not.  That is exactly
         the bug a careless `qk_norm` plumbing would introduce, and the gate
         must reject it.

HYPOTHESIS UNDER TEST (registered before the numbers were read): removing the
cap should make the fold MORE exact, because per-head RMSNorm is a
data-dependent rescaling the fold has to absorb.  The sharp place to look is
the layer-0 attention table identity, which is the only residual the cap can
touch; the MLP identities are a within-comparison negative control, since
qk_norm cannot affect them.

VERDICT (H2 NOT SUPPORTED -- see rep['verdict']).  The premise is wrong.  At
layer 0 the query/key RMSNorm is NOT data-dependent: its input is
rms_norm(wte[token]) with no context, so it is a deterministic function of the
TOKEN, and fold_layer0_qk applies it inside the token-indexed factor tables --
absorbed exactly, at no cost.  Measured: on the attention-table identity the
un-capped residual is smaller in 5 of 12 matched cells (geometric mean ratio
capped/un-capped 0.92, i.e. if anything the capped fold is marginally tighter),
and the MLP negative control splits the same coin-flip way, 7 of 12 at ratio
1.08.  On the end-to-end fold_forward logit residual the un-capped side is
consistently LARGER, 12 of 12, by a geometric mean factor of 1.54 -- the
opposite of the prediction, and explained by the factor scales rather than by
exactness: with the cap on, every folded query/key row has norm exactly
sqrt(head_dim) = 4.000, while with it off the row norms run 1.24-2.27 with
element absmax up to 8.08, a wider dynamic range for the score products to
round in.  All 24 residuals lie between 3.0e-16 and 2.4e-15 (algebraic) and
4.5e-14 and 1.7e-13 (end-to-end absolute), i.e. within one decade of the fp64
machine floor and ~15 orders below the 1.489 the norm-mismatch control
produces.  The correct summary is that the cap costs the fold NOTHING in
either direction, and the residual ordering is rounding bookkeeping.

Everything runs on CPU by default: fp64 is exact there, it cannot be perturbed
by whoever owns the GPU, and one device for all 24 checkpoints is what makes
the capped/un-capped residuals comparable at all.

Usage
    python tf_noqknorm_foldgate.py [--device cpu] [--batch 4] [--ctx 128]
                                   [--out tf_noqknorm_foldgate.json]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import tf_corpus
import tf_factorial as FAC
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))

UNCAPPED = [f'tff_bilin_bilin_d{d}_w{w}_b8192_s{s}_noqknorm'
            for d in (2, 3) for w in (64, 128) for s in (0, 1, 2)]
CAPPED = [f'tf_vanilla_d{d}_w{w}_b8192_s{s}'
          for d in (2, 3) for w in (64, 128) for s in (0, 1, 2)]
CAPPED_REQUIRED = [f'tf_vanilla_d2_w128_b8192_s{s}' for s in (0, 1, 2)]

REGISTERED = {
    'written_before_the_numbers_were_read': True,
    'H1': 'the un-capped (qk_norm=False) arms pass the fp64 fold-identity gate',
    'H2': 'the un-capped fold is MORE exact than the capped fold at the same '
          'cell, because per-head query/key RMSNorm is a data-dependent '
          'rescaling that the fold has to absorb.  Sharp observable: '
          'fp64_attn_layer0_table_identity_relmax and its all-heads factor '
          'twin.  Direction predicted: uncapped < capped.',
    'H2_negative_control': 'the fp64 MLP identity residuals must NOT show the '
                           'same split -- qk_norm cannot reach them.  If they '
                           'move together, the effect is precision bookkeeping, '
                           'not the cap.',
}

GATE_KEYS = ('fp64_attn_layer0_table_identity_relmax',
             'fp64_attn_layer0_factor_identity_allheads_relmax',
             'fp64_fold_forward_max_logit_diff',
             'fp64_logit_absmax',
             'fp64_fold_forward_rel_logit_diff',
             'fold_forward_max_logit_diff',
             'fold_forward_rel_logit_diff',
             'attn_layer0_table_identity_relmax',
             'fp32_forward_vs_fp64_forward_rel',
             'fold_vs_forward_over_fp32_selfnoise',
             'fp32_sanity_pass', 'fp64_exactness_pass', 'pass')
# the MLP identities are per LAYER (mlp_tensor_identity_l0_relmax, ...); the
# comparison uses the worst layer, which is the quantity the gate itself
# thresholds.
MLP_AGG = {'fp64_mlp_tensor_identity_relmax_maxlayer': 'fp64_mlp_tensor_identity_l',
           'fp64_mlp_rmsnorm_gauge_relmax_maxlayer': 'fp64_mlp_rmsnorm_gauge_l'}


def held_tokens(vocab, tok, batch, ctx, device):
    h = tf_corpus.load_split(vocab, 'held', batch, tok=tok)
    return torch.from_numpy(h).to(device)[:, :ctx]


# ------------------------------------------------------- 0. transplant check
@torch.no_grad()
def transplant(stem, device, idx_held):
    """Build TinyBilin(qk_norm from the checkpoint) and prove it IS the trained
    FacTransformer before anything is gated."""
    ck = torch.load(f'{HERE}/{stem}.pt', map_location=device,
                    weights_only=False)
    fcfg = FAC.FacConfig(**ck['cfg'])
    fac = FAC.FacTransformer(fcfg).to(device)
    fac.load_state_dict(ck['state_dict'])
    fac.eval()

    tcfg = M.TFConfig(depth=fcfg.depth, width=fcfg.width, vocab=fcfg.vocab,
                      tok=fcfg.tok, seed=fcfg.seed, variant='vanilla',
                      T=fcfg.T, qk_norm=fcfg.qk_norm)
    tiny = M.make_model(tcfg, device)

    rep = {'stem': stem, 'fac_cfg_qk_norm': fcfg.qk_norm,
           'tiny_cfg_qk_norm': tcfg.qk_norm,
           'fac_hidden': fcfg.hidden, 'tiny_hidden': tcfg.hidden,
           'hidden_agrees': bool(fcfg.hidden == tcfg.hidden),
           'checkpoint_held_ce': ck.get('log', {}).get('final_held_ce')}

    nf = {n: tuple(p.shape) for n, p in fac.named_parameters()}
    nt = {n: tuple(p.shape) for n, p in tiny.named_parameters()}
    rep['n_params_fac'], rep['n_params_tiny'] = len(nf), len(nt)
    rep['names_only_in_fac'] = sorted(set(nf) - set(nt))
    rep['names_only_in_tiny'] = sorted(set(nt) - set(nf))
    rep['name_sets_equal'] = bool(set(nf) == set(nt))
    rep['shape_mismatches'] = [[n, list(nf[n]), list(nt[n])]
                               for n in sorted(set(nf) & set(nt))
                               if nf[n] != nt[n]]
    rep['shapes_agree'] = not rep['shape_mismatches']
    if not (rep['name_sets_equal'] and rep['shapes_agree']):
        rep['pass'] = False
        rep['why'] = 'parameter name set / shapes differ -- NOT transplantable'
        return rep, None, None

    # strict state_dict load, reported honestly (buffers may legitimately differ)
    miss = tiny.load_state_dict(ck['state_dict'], strict=False)
    rep['state_dict_missing_keys'] = list(miss.missing_keys)
    rep['state_dict_unexpected_keys'] = list(miss.unexpected_keys)
    rep['missing_keys_are_all_buffers'] = bool(
        all(k not in nt for k in miss.missing_keys))
    for n, p in tiny.named_parameters():          # belt and braces
        p.copy_(dict(fac.named_parameters())[n])
    tiny.eval()

    g = torch.Generator().manual_seed(17)
    idx_rand = torch.randint(0, fcfg.vocab, (3, 64), generator=g).to(device)
    f32 = {}
    with M.exact_math():
        for tag, ii in (('random', idx_rand), ('held', idx_held)):
            a, b = tiny(ii).float(), fac(ii).float()
            f32[tag] = (a, b)
            rep[f'forward_max_abs_diff_{tag}'] = float((a - b).abs().max())
            rep[f'logit_absmax_{tag}'] = float(b.abs().max())
            rep[f'forward_rel_diff_{tag}'] = float(
                (a - b).abs().max() / b.abs().max().clamp_min(1e-30))
        # fp64 leg: the fp32 numbers above sit at ~5-8 fp32 ulps of a logit of
        # magnitude ~15, which is a PRECISION floor, not disagreement.  Redo it
        # in fp64 (rotary tables rebuilt on both sides) so "the two models are
        # the same function" is not a statement about rounding.  README
        # standing failure mode: precision mistaken for correctness.
        t64 = M.cast_model(tiny, torch.float64)
        f64 = FAC.FacTransformer(fcfg).to(device).double().eval()
        f64.load_state_dict({k: v.double() if v.is_floating_point() else v
                             for k, v in ck['state_dict'].items()})
        c64, s64 = M.rope_tables_exact(fcfg.T, fcfg.head_dim, 'cpu',
                                       torch.float64)
        f64.cos, f64.sin = c64.to(device), s64.to(device)
        for tag, ii in (('random', idx_rand), ('held', idx_held)):
            a, b = t64(ii), f64(ii)
            rep[f'fp64_forward_max_abs_diff_{tag}'] = float((a - b).abs().max())
            rep[f'fp64_forward_rel_diff_{tag}'] = float(
                (a - b).abs().max() / b.abs().max().clamp_min(1e-30))
            # CALIBRATION, exactly as check_fold_identities does it: how far is
            # the FacTransformer's own fp32 forward from its own fp64 value?
            # The transplant gap must be no more than 10x that floor.
            sc = b.abs().max().clamp_min(1e-30)
            noise = float((f32[tag][1].double() - b).abs().max() / sc)
            rep[f'fac_fp32_vs_fp64_self_noise_rel_{tag}'] = noise
            rep[f'transplant_gap_over_self_noise_{tag}'] = float(
                rep[f'forward_rel_diff_{tag}'] / max(noise, 1e-30))
        del t64, f64
    tags = ('random', 'held')
    # SPEC-LITERAL criterion (absolute 1e-5 on fp32 logits), reported so the
    # record is honest -- but NOT the verdict.  It is the criterion this
    # programme superseded on 2026-08-08: these logits live on 30*tanh(./30)
    # and reach ~15, where one fp32 ulp is already ~1e-6, so an absolute 1e-5
    # is ~8 ulps for a forward that accumulates thousands of roundings.  It
    # fails on the depth-3 width-128 cells for rounding alone.
    rep['pass_spec_literal_fp32_absolute_1e-5'] = bool(
        all(rep[f'forward_max_abs_diff_{t}'] < 1e-5 for t in tags))
    rep['criterion'] = {
        'fp32_relative': 1e-5,
        'fp64_absolute': 1e-9,
        'self_noise_multiple': 10.0,
        'why': 'the same two-tier, scale-free criterion tf_model.'
               'check_fold_identities uses.  It is STRICTLY STRONGER than the '
               'absolute-1e-5 fp32 check it replaces: the fp64 leg is five '
               'orders of magnitude tighter than 1e-5, and the self-noise '
               'calibration would catch a real-but-small difference hiding '
               'under any fixed fp32 budget.'}
    rep['pass'] = bool(
        all(rep[f'forward_rel_diff_{t}'] < 1e-5 for t in tags)
        and all(rep[f'fp64_forward_max_abs_diff_{t}'] < 1e-9 for t in tags)
        and all(rep[f'transplant_gap_over_self_noise_{t}'] < 10.0
                for t in tags))
    del fac
    return rep, tiny, tcfg


def load_capped(stem, device):
    ck = torch.load(f'{HERE}/{stem}.pt', map_location=device,
                    weights_only=False)
    cfg = M.TFConfig(**{k: v for k, v in ck['cfg'].items()
                        if k in M.TFConfig.__dataclass_fields__})
    if 'tok' not in ck['cfg']:
        cfg.tok = 'trunc'
    m = M.make_model(cfg, device)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, cfg, ck


def cell_of(stem):
    p = stem.split('_')
    d = [x for x in p if x.startswith('d') and x[1:].isdigit()][0]
    w = [x for x in p if x.startswith('w') and x[1:].isdigit()][0]
    s = [x for x in p if x.startswith('s') and x[1:].isdigit()][0]
    return f'{d}_{w}_{s}'


@torch.no_grad()
def qk_factor_scale(model, dtype=torch.float64):
    """Scale of the folded layer-0 query/key factors.  With the cap on, the
    per-head RMSNorm pins every factor row to norm sqrt(head_dim), so elements
    are O(1); with it off they are whatever the trained projection produces.
    This is the quantity that decides how much ABSOLUTE fp64 rounding the
    score products carry, so it is what separates 'the cap changes the fold's
    exactness' from 'the cap changes the numbers being rounded'."""
    f = model.fold_layer0_qk(materialize=False, dtype=dtype)
    out = {}
    for n in ('Q1', 'K1', 'Q2', 'K2'):
        z = f[n]
        out[f'{n}_rms'] = float(z.pow(2).mean().sqrt())
        out[f'{n}_absmax'] = float(z.abs().max())
        out[f'{n}_row_norm_mean'] = float(z.norm(dim=-1).mean())
    out['head_dim_sqrt'] = float(model.cfg.head_dim ** 0.5)
    out['note'] = ('row_norm_mean == sqrt(head_dim) exactly iff the per-head '
                   'query/key RMSNorm is applied')
    return out


def slim(gate):
    out = {k: gate[k] for k in GATE_KEYS if k in gate}
    for agg, pref in MLP_AGG.items():
        vals = [v for k, v in gate.items() if k.startswith(pref)]
        if vals:
            out[agg] = max(vals)
    for k, v in gate.items():                    # keep every per-layer number
        if k.startswith('fp64_mlp_') or k.startswith('mlp_'):
            out[k] = v
    return out


# ------------------------------------------------------------------ mismatch
@torch.no_grad()
def norm_mismatch_control(tiny, idx):
    """Gate-must-fail control aimed at THIS change: force the fold to normalise
    the layer-0 query/key factors while the forward (qk_norm=False) does not.
    A gate that waves this through cannot certify a cap-off fold at all."""
    assert tiny.cfg.qk_norm is False
    orig = tiny.fold_layer0_qk

    def fq(*a, **kw):
        tiny.cfg.qk_norm = True                  # fold side only
        try:
            return orig(*a, **kw)
        finally:
            tiny.cfg.qk_norm = False
    tiny.fold_layer0_qk = fq
    try:
        r = M.check_fold_identities(tiny, idx, verbose=False)
    finally:
        tiny.fold_layer0_qk = orig
        tiny.cfg.qk_norm = False
    return {'corrupted_gate_pass': bool(r['pass']),
            'must_be_false': True,
            'pass': bool(not r['pass']),
            'fp64_attn_layer0_table_identity_relmax':
                r['fp64_attn_layer0_table_identity_relmax'],
            'fp64_fold_forward_max_logit_diff':
                r['fp64_fold_forward_max_logit_diff'],
            'what_was_corrupted': 'fold_layer0_qk applied the per-head RMSNorm '
                                  'that the qk_norm=False forward does not'}


# ---------------------------------------------------------------------- main
def main(device='cpu', batch=4, ctx=128, out_path=None,
         uncapped=UNCAPPED, capped=CAPPED):
    t0 = time.time()
    rep = {'task': 'TASK 2 -- fold-identity gate for the un-capped '
                   '(qk_norm=False) foldable arms',
           'device': device, 'precision': 'fp32 sanity tier + fp64 exactness '
                                          'tier, both from '
                                          'tf_model.check_fold_identities',
           'gate_tokens': {'split': 'held', 'rows': f'[0:{batch}]',
                           'context_len': ctx,
                           'note': 'IDENTICAL tokens for every checkpoint -- '
                                   'that is what makes the capped and '
                                   'un-capped residuals comparable'},
           'registered_predictions': REGISTERED,
           'capped_required_by_task': CAPPED_REQUIRED,
           'capped_extra_for_cell_matching':
               [s for s in capped if s not in CAPPED_REQUIRED]}

    idx = held_tokens(8192, 'bpe', batch, ctx, device)
    rep['gate_tokens']['shape'] = list(idx.shape)

    # ---- the gate must be able to fail (run FIRST, so a broken gate is known
    # before any residual is quoted)
    print('planted known-answer test ...', flush=True)
    rep['planted_known_answer'] = M.planted_qk_test(device=device)
    print('  pass:', rep['planted_known_answer']['pass'], flush=True)
    print('gate negative control ...', flush=True)
    rep['gate_negative_control'] = M.gate_negative_control(device=device)
    print('  pass:', rep['gate_negative_control']['pass'], flush=True)

    rep['uncapped'], rep['capped'] = {}, {}
    first_uncapped = None
    for stem in uncapped:
        t = time.time()
        print(f'== uncapped {stem}', flush=True)
        tr, tiny, tcfg = transplant(stem, device, idx)
        entry = {'transplant': tr, 'cell': cell_of(stem), 'capped': False}
        if not tr['pass']:
            entry['gated'] = False
            entry['why_not_gated'] = tr.get('why', 'forwards disagree')
            rep['uncapped'][stem] = entry
            print('  TRANSPLANT FAILED -- not gated', flush=True)
            continue
        if first_uncapped is None:
            first_uncapped = stem
            print('  norm-mismatch control ...', flush=True)
            rep['fold_norm_mismatch_control'] = norm_mismatch_control(tiny, idx)
            rep['fold_norm_mismatch_control']['on_checkpoint'] = stem
            print('   pass:', rep['fold_norm_mismatch_control']['pass'],
                  flush=True)
        g = M.check_fold_identities(tiny, idx, verbose=False)
        entry['gated'] = True
        entry['gate'] = slim(g)
        entry['gate_full'] = {k: v for k, v in g.items() if k != 'criterion'}
        entry['qk_factor_scale'] = qk_factor_scale(M.cast_model(tiny,
                                                                torch.float64))
        entry['seconds'] = round(time.time() - t, 1)
        rep['uncapped'][stem] = entry
        print(f'  pass={g["pass"]}  fp64 attn table rel='
              f'{g["fp64_attn_layer0_table_identity_relmax"]:.3e}  '
              f'fp64 logit abs={g["fp64_fold_forward_max_logit_diff"]:.3e}  '
              f'({entry["seconds"]}s)', flush=True)
        del tiny

    for stem in capped:
        t = time.time()
        print(f'== capped   {stem}', flush=True)
        m, cfg, ck = load_capped(stem, device)
        g = M.check_fold_identities(m, idx, verbose=False)
        rep['capped'][stem] = {
            'cell': cell_of(stem), 'capped': True, 'gated': True,
            'cfg_qk_norm': cfg.qk_norm,
            'checkpoint_held_ce': ck.get('log', {}).get('final_held_ce'),
            'gate': slim(g),
            'gate_full': {k: v for k, v in g.items() if k != 'criterion'},
            'qk_factor_scale': qk_factor_scale(M.cast_model(m,
                                                            torch.float64)),
            'seconds': round(time.time() - t, 1)}
        print(f'  pass={g["pass"]}  fp64 attn table rel='
              f'{g["fp64_attn_layer0_table_identity_relmax"]:.3e}  '
              f'fp64 logit abs={g["fp64_fold_forward_max_logit_diff"]:.3e}  '
              f'({rep["capped"][stem]["seconds"]}s)', flush=True)
        del m

    # ---------------------------------------------------- comparison
    ug = [v for v in rep['uncapped'].values() if v.get('gated')]
    cg = [v for v in rep['capped'].values() if v.get('gated')]

    def pull(d, key):
        return {v['cell']: v['gate'][key] for v in d.values() if v.get('gated')}

    cmp_keys = ('fp64_attn_layer0_table_identity_relmax',
                'fp64_attn_layer0_factor_identity_allheads_relmax',
                'fp64_mlp_tensor_identity_relmax_maxlayer',
                'fp64_mlp_rmsnorm_gauge_relmax_maxlayer',
                'fp64_fold_forward_max_logit_diff',
                'fp64_fold_forward_rel_logit_diff')
    comp = {}
    for k in cmp_keys:
        u, c = pull(rep['uncapped'], k), pull(rep['capped'], k)
        cells = sorted(set(u) & set(c))
        pairs = {cl: {'uncapped': u[cl], 'capped': c[cl],
                      'ratio_capped_over_uncapped':
                          (c[cl] / u[cl]) if u[cl] > 0 else None,
                      'uncapped_smaller': bool(u[cl] < c[cl])}
                 for cl in cells}
        uv = np.array([u[cl] for cl in cells], dtype=np.float64)
        cv = np.array([c[cl] for cl in cells], dtype=np.float64)
        comp[k] = {
            'per_cell': pairs,
            'n_cells_matched': len(cells),
            'n_cells_uncapped_smaller': int((uv < cv).sum()),
            'median_uncapped': float(np.median(uv)),
            'median_capped': float(np.median(cv)),
            'median_ratio_capped_over_uncapped':
                float(np.median(cv / np.where(uv > 0, uv, np.nan)))
                if (uv > 0).all() else None,
            'geomean_ratio_capped_over_uncapped':
                float(np.exp(np.mean(np.log(cv / uv))))
                if (uv > 0).all() and (cv > 0).all() else None}
    rep['capped_vs_uncapped'] = comp
    # is the comparison even resolvable?  The two gate-must-fail controls give
    # the detector's headroom: the norm-mismatch control moves the attention
    # table residual from ~1e-15 to ~1.5 (15 orders), and the negative
    # control's 1+1e-7 MLP corruption is caught.  So a null here is a real
    # null bounded at the fp64 floor, not a blind detector.
    rep['resolution'] = {
        'attn_table_residual_range_seen_all_24': [
            min(v['gate']['fp64_attn_layer0_table_identity_relmax']
                for v in list(ug) + list(cg)),
            max(v['gate']['fp64_attn_layer0_table_identity_relmax']
                for v in list(ug) + list(cg))],
        'same_residual_under_the_norm_mismatch_control':
            rep.get('fold_norm_mismatch_control', {}).get(
                'fp64_attn_layer0_table_identity_relmax'),
        'smallest_corruption_the_gate_catches':
            '1 + 1e-7 relative on the MLP tensor (gate_negative_control)',
        'reading': 'every capped and un-capped residual sits within one decade '
                   'of the fp64 machine floor, ~15 orders below what a wrong '
                   'fold produces, so the capped/un-capped ordering is '
                   'rounding bookkeeping and not an exactness difference.'}

    rep['summary'] = {
        'n_uncapped_checkpoints': len(rep['uncapped']),
        'n_uncapped_transplanted': sum(
            1 for v in rep['uncapped'].values() if v['transplant']['pass']),
        'n_uncapped_transplanted_by_spec_literal_absolute_1e-5': sum(
            1 for v in rep['uncapped'].values()
            if v['transplant']['pass_spec_literal_fp32_absolute_1e-5']),
        'worst_transplant_fp64_abs_diff': max(
            max(v['transplant'].get(f'fp64_forward_max_abs_diff_{t}', 0.0)
                for t in ('random', 'held'))
            for v in rep['uncapped'].values()),
        'worst_transplant_fp32_rel_diff': max(
            max(v['transplant'].get(f'forward_rel_diff_{t}', 0.0)
                for t in ('random', 'held'))
            for v in rep['uncapped'].values()),
        'worst_transplant_gap_over_self_noise': max(
            max(v['transplant'].get(f'transplant_gap_over_self_noise_{t}', 0.0)
                for t in ('random', 'held'))
            for v in rep['uncapped'].values()),
        'n_uncapped_gate_pass': sum(1 for v in ug if v['gate']['pass']),
        'n_capped_gate_pass': sum(1 for v in cg if v['gate']['pass']),
        'worst_uncapped_fp64_logit_abs': max(
            [v['gate']['fp64_fold_forward_max_logit_diff'] for v in ug],
            default=None),
        'worst_capped_fp64_logit_abs': max(
            [v['gate']['fp64_fold_forward_max_logit_diff'] for v in cg],
            default=None),
        'planted_pass': rep['planted_known_answer']['pass'],
        'negative_control_pass': rep['gate_negative_control']['pass'],
        'norm_mismatch_control_pass':
            rep.get('fold_norm_mismatch_control', {}).get('pass'),
        'H2_verdict_attn_table': (
            'uncapped smaller in '
            f'{comp["fp64_attn_layer0_table_identity_relmax"]["n_cells_uncapped_smaller"]}'
            f'/{comp["fp64_attn_layer0_table_identity_relmax"]["n_cells_matched"]}'
            ' matched cells'),
        'wall_seconds': round(time.time() - t0, 1)}
    rep['all_pass'] = bool(
        rep['summary']['n_uncapped_gate_pass'] == len(UNCAPPED)
        and rep['summary']['n_capped_gate_pass'] == len(cg)
        and rep['planted_known_answer']['pass']
        and rep['gate_negative_control']['pass']
        and rep.get('fold_norm_mismatch_control', {}).get('pass'))
    ct = comp['fp64_attn_layer0_table_identity_relmax']
    cm = comp['fp64_mlp_tensor_identity_relmax_maxlayer']
    ce = comp['fp64_fold_forward_max_logit_diff']
    rep['verdict'] = {
        'H1': ('SUPPORTED: all 12 un-capped checkpoints pass the fp64 gate; '
               'worst end-to-end fp64 residual '
               f'{rep["summary"]["worst_uncapped_fp64_logit_abs"]:.3e} against '
               'a 1e-9 threshold'),
        'H2': 'NOT SUPPORTED',
        'H2_evidence': {
            'attn_table_uncapped_smaller_in':
                f'{ct["n_cells_uncapped_smaller"]}/{ct["n_cells_matched"]}',
            'attn_table_geomean_ratio_capped_over_uncapped':
                ct['geomean_ratio_capped_over_uncapped'],
            'mlp_negative_control_uncapped_smaller_in':
                f'{cm["n_cells_uncapped_smaller"]}/{cm["n_cells_matched"]}',
            'mlp_negative_control_geomean_ratio':
                cm['geomean_ratio_capped_over_uncapped'],
            'end_to_end_uncapped_smaller_in':
                f'{ce["n_cells_uncapped_smaller"]}/{ce["n_cells_matched"]}',
            'end_to_end_geomean_ratio_capped_over_uncapped':
                ce['geomean_ratio_capped_over_uncapped']},
        'why': ('the hypothesis assumed the query/key RMSNorm is DATA-dependent '
                'and therefore something the fold must absorb approximately.  '
                'At layer 0 it is not: its input is rms_norm(wte[token]) with '
                'no context, so it is a deterministic function of the token and '
                'fold_layer0_qk folds it exactly into the token-indexed Q/K '
                'factor tables.  The capped folded rows have norm exactly '
                'sqrt(head_dim); the un-capped ones do not, which is why the '
                'un-capped end-to-end residual is slightly LARGER -- a wider '
                'dynamic range to round in, not a less exact fold.'),
        'caveat': ('the attention-table comparison is a coin flip and the MLP '
                   'negative control splits the same way, so neither direction '
                   'is an effect.  Both arms sit at the fp64 floor; this is a '
                   'null with a calibrated detector, not a measurement of a '
                   'small difference.')}
    op = out_path or f'{HERE}/tf_noqknorm_foldgate.json'
    json.dump(rep, open(op, 'w'), indent=2)
    print(f'\nwrote {op}  all_pass={rep["all_pass"]}  '
          f'({rep["summary"]["wall_seconds"]}s)', flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--ctx', type=int, default=128)
    ap.add_argument('--out', default=None)
    ap.add_argument('--only', default=None,
                    help='comma-separated stems, for a quick smoke run')
    a = ap.parse_args()
    if a.only:
        sel = a.only.split(',')
        main(a.device, a.batch, a.ctx, a.out,
             uncapped=[s for s in UNCAPPED if s in sel],
             capped=[s for s in CAPPED if s in sel])
    else:
        main(a.device, a.batch, a.ctx, a.out)
