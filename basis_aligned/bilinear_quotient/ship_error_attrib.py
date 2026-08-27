# ship_error_attrib: WHICH PLANK GROUP CAUSES THE FUNCTION-TOKEN BACKGROUND?
# (S1561: the most-frequent 100 tokens carry a diffuse background of ship damage.)
# Metric: CE on positions whose target is one of the 100 most frequent tokens.
# Arms: clean / attention-only / attn+mlp0,1,2 / full ship — the increments
# attribute the background to plank groups.
# Original: TOTAL GLASS REBUILT WITH THE UNIT CLASS (S1534-35: top-1024-unit
# sub-MLPs are the price-efficient deep planks, ~.55-.89 recovery vs the v1
# stream-linalls' ~.45). Config: attention three-tier (all 18) + mlp0/1/2 planks +
# mlp2 correction (loaded) + mlp3 output-projection + mlps 4-17 as top-1024-unit
# sub-MLPs computed INLINE from each block's own input (no captures, no ridges).
# Unit rankings from 96 clean rows.
# Original header: ATTENTION THREE-TIER COMPOSITE + THE mlp0/mlp1 PLANKS
# (S1477: the attention composite costs only .162 CE; the board's top two are mlp1
# and mlp0, which have verified per-module planks. This measures the actual glass
# ship so far.) MLP stand-ins are CONTEXT-FIT (lesson 3): tier tables + ridges fit on
# captures taken UNDER the attention composite; at eval they are computed INLINE
# single-pass (stand0 = TIER0[tok] + ridge0(a0); stand1 = TIER1[tok] +
# ridge1([a1, m0_in_stream]) where m0_in_stream is stand0 when mlp0 is also
# replaced). Arms: clean / attn / mlp1 / mlp0 / attn+mlp1 / attn+mlp1+mlp0.
# Original header: THE THREE-TIER CLASS AS A COMPOSITE (S1474: per-layer whitened
# r32 QK >= .84 everywhere; composite lessons 1-3 say per-layer fids do NOT predict
# the composite, so measure it). ALL 18 layers simultaneously on their best class:
# whitened r32 for generic heads (plain SVD at 8/16/17 per S1469 best-of), FULL QK
# for the named roster heads (SPEC below). MLPs/values live. Also measures each
# layer's SINGLE-replacement delta in-script so the compounding factor is
# self-contained. NR=960.
# Original header: INPUT-WHITENED per-head truncation (S1467: plain SVD went 0-for-3
# — attn5 r32 = -1.61, WORSE than the kernel, and attn8 r8 = -2.49. The registered
# assumption 'whitening omitted' is the prime suspect: plain SVD ranks directions by
# weight norm, but the stream covariance is far from isotropic, so the kept directions
# can miss the high-variance inputs the sink head reads. Fix: per-layer stream
# covariance Sigma of xin (96 rows), whiten W' = W_head @ Sigma^{1/2}, truncate, map
# back. Same price, same layers, same bars.)
# Original header: A NEW MID-PRICE STAND-IN CLASS FOR KERNEL-RESISTANT LAYERS
# (licensed by S1464: attention-pattern edges are extremely low-rank — the composed
# mlp0->pattern edge was 98% rank-8). Claim to test: each head's QK maps are
# themselves low-rank readable — replace ALL FOUR pattern maps (c_q/c_k/c_q2/c_k2)
# with PER-HEAD rank-r truncations (plain SVD of each head's [128,1152] slice;
# input-covariance whitening deliberately omitted — registered assumption) at the 7
# kernel-resistant content layers {5,8,10,13,14,16,17}, one layer at a time,
# everything else live. Values live. fid_opt vs frozen anchors.
# Price: rank-r/head = 4 maps x 9 heads x r x (128+1152) x 16b — r=32: 23.6 Mbit/layer
# (vs 85 Mbit full QK, .037 Mbit kernel). The rung between kernel and full head.
#
# Registered predictions:
#   pred_a attention contributes <= .35 of the frequent-class damage.
#   pred_b the mlp0/1/2 plank group contributes >= .35.
#   pred_c increments sum to the full-ship damage within 15% (additivity holds on
#          this class too).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ship_error_attrib_results.json'
NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
LAYERS = list(range(18))
PLAIN = {8, 16, 17}
SPEC = {5: {7}, 8: {1, 2, 3, 7}, 10: {2, 3, 4, 5, 6}, 13: {0, 5, 8},
        14: {4, 6, 7}, 16: {0, 3, 4, 5}, 17: {0, 1, 2}}


def trunc_perhead(W, r, Wh, Whi):
    """Whitened per-head rank-r: SVD of W_head @ Wh, then map back with Whi."""
    Wf = W.float().to(DEV).view(9, 128, D)
    out = torch.zeros_like(Wf)
    for h in range(9):
        U, S, Vt = torch.linalg.svd(Wf[h] @ Wh, full_matrices=False)
        out[h] = ((U[:, :r] * S[:r]) @ Vt[:r]) @ Whi
    return out.view(9 * 128, D)


SHIP = {'t0': None, 'r0': None, 't1': None, 'r1': None, 'r2': None, 'r17': None,
        'p3': None, 'mean3': None}
CORR = {'on': False, 'b': None, 'U': None, 'V': None}
CONTENT_CORR = {'on': False, 'site': None, 'weight': None, 'bias': None,
                'basis': None}
ORACLE_CORR = {'on': False, 'site': None, 'basis': None, 'scale': 1.0,
               'capture': None}


def add_content_correction(site, z, mo):
    """Apply a fixed-output-basis correction from the live normalized MLP input."""
    if not CONTENT_CORR['on'] or CONTENT_CORR['site'] != site:
        return mo
    coeff = z.float().reshape(-1, D) @ CONTENT_CORR['weight'] + CONTENT_CORR['bias']
    delta = (coeff @ CONTENT_CORR['basis'].T).view_as(mo)
    return mo + delta.to(mo.dtype)


def add_oracle_correction(site, block, z, mo):
    """Inject the live original-minus-plank residual, optionally projected."""
    should_capture = ORACLE_CORR['capture'] is not None and site in ORACLE_CORR['capture']
    should_inject = ORACLE_CORR['on'] and ORACLE_CORR['site'] == site
    if not should_capture and not should_inject:
        return mo
    residual = block.mlp(z).float() - mo.float()
    if should_capture:
        ORACLE_CORR['capture'][site].append(residual[:, 64::3].detach().cpu())
    if not should_inject:
        return mo
    basis = ORACLE_CORR['basis']
    if basis is None:
        delta = residual
    else:
        flat = residual.reshape(-1, D)
        delta = ((flat @ basis) @ basis.T).view_as(residual)
    return mo + (ORACLE_CORR['scale'] * delta).to(mo.dtype)


def fwd_arm(idx, layers, TWALL, mlps=frozenset(), cap=None):
    """layers: attn layers replaced. mlps: subset of {0,1,2,17} replaced inline.
    cap: optional dict collecting fit-pass tensors."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    toks = idx.reshape(-1)
    a0_out = None; a1_out = None; m0_stream = None
    mstream = []
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if L in layers:
            TW = TWALL[L]
            qp = (xin.float() @ TW['q'].T).view(B, T, 9, 128)
            kp = (xin.float() @ TW['k'].T).view(B, T, 9, 128)
            q2p = (xin.float() @ TW['q2'].T).view(B, T, 9, 128)
            k2p = (xin.float() @ TW['k2'].T).view(B, T, 9, 128)
            if L in SPEC:
                qf = at.c_q(xin).view(B, T, 9, 128).float()
                kf = at.c_k(xin).view(B, T, 9, 128).float()
                q2f = at.c_q2(xin).view(B, T, 9, 128).float()
                k2f = at.c_k2(xin).view(B, T, 9, 128).float()
                for hh in SPEC[L]:
                    qp[:, :, hh] = qf[:, :, hh]; kp[:, :, hh] = kf[:, :, hh]
                    q2p[:, :, hh] = q2f[:, :, hh]; k2p[:, :, hh] = k2f[:, :, hh]
        else:
            qp = at.c_q(xin).view(B, T, 9, 128).float()
            kp = at.c_k(xin).view(B, T, 9, 128).float()
            q2p = at.c_q2(xin).view(B, T, 9, 128).float()
            k2p = at.c_k2(xin).view(B, T, 9, 128).float()
        cos, sin = at.rotary(qp)
        q = are(F.rms_norm(qp, (128,)), cos, sin)
        k = are(F.rms_norm(kp, (128,)), cos, sin)
        q2 = are(F.rms_norm(q2p, (128,)), cos, sin)
        k2 = are(F.rms_norm(k2p, (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        ao = at.c_proj(y.reshape(B, T, D))
        x = xm + ao
        z = F.rms_norm(x, (D,))
        if L == 0:
            a0_out = ao.float()
            if 0 in mlps:
                W0, xm0_, ym0_ = SHIP['r0']
                st = SHIP['t0'][toks].view(B, T, D) \
                    + (ym0_ + (a0_out.reshape(-1, D) - xm0_) @ W0).view(B, T, D)
                mo = st.to(x.dtype)
            else:
                mo = blk.mlp(z)
            mo = add_oracle_correction(0, blk, z, mo)
            mo = add_content_correction(0, z, mo)
            m0_stream = mo.float()
            mstream.append(m0_stream)
            x = x + mo
        elif L == 1:
            a1_out = ao.float()
            if 1 in mlps:
                W1, xm1_, ym1_ = SHIP['r1']
                X2 = torch.cat([a1_out, m0_stream], -1).reshape(-1, 2 * D)
                st = SHIP['t1'][toks].view(B, T, D) \
                    + (ym1_ + (X2 - xm1_) @ W1).view(B, T, D)
                mo = st.to(x.dtype)
            else:
                mo = blk.mlp(z)
            mo = add_oracle_correction(1, blk, z, mo)
            mo = add_content_correction(1, z, mo)
            mstream.append(mo.float())
            x = x + mo
        elif L == 2:
            if cap is not None:
                cap.setdefault('a2', []).append(ao.float().cpu())
                cap.setdefault('m2', []).append(blk.mlp(z).float().cpu())
            if 2 in mlps:
                W2, xm2_, ym2_ = SHIP['r2']
                X2 = torch.cat([ao.float(), mstream[1]], -1).reshape(-1, 2 * D)
                mo_ = ym2_ + (X2 - xm2_) @ W2
                if CORR['on']:
                    mo_ = mo_ + CORR['b'] + ((X2 - xm2_) @ CORR['V']) @ CORR['U'].T
                mo = mo_.view(B, T, D).to(x.dtype)
            else:
                mo = blk.mlp(z)
            mo = add_oracle_correction(2, blk, z, mo)
            mo = add_content_correction(2, z, mo)
            mstream.append(mo.float())
            x = x + mo
        elif L == 3:
            if 3 in mlps:
                mo_full = blk.mlp(z).float()
                cen = mo_full - SHIP['mean3']
                mo = (SHIP['mean3']
                      + (cen.reshape(-1, D) @ SHIP['p3'].T) @ SHIP['p3']
                      ).view(B, T, D).to(x.dtype)
            else:
                mo = blk.mlp(z)
            mstream.append(mo.float())
            x = x + mo
        elif L >= 4:
            if cap is not None and L == 17:
                cap.setdefault('mstream', []).append(torch.cat(mstream, -1).cpu())
            if L in mlps:
                U = SHIP[f'u{L}']
                h = (z.float() @ U['l'].T) * (z.float() @ U['r'].T)
                mo = (h @ U['d'].T + U['b']).to(x.dtype)
            else:
                mo = blk.mlp(z)
            if L != 17:
                mstream.append(mo.float())
            x = x + mo
        else:
            mo = blk.mlp(z)
            mstream.append(mo.float())
            x = x + mo
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def _token_masks(rows):
    """Match the registered factorial's held-token cell definitions."""
    counts = torch.zeros(50257)
    for start in range(0, len(rows), 8):
        targets = rows[start:start + 8, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        valid[:, :64] = False
        counts.index_add_(0, targets.reshape(-1), valid.reshape(-1).float())
    threshold = counts.sort(descending=True).values[500]
    return (counts < threshold).to(DEV)


@torch.no_grad()
def _score_content_rows(rows, TWALL, all_attention, all_mlps, rare_vocab=None,
                        retain_row_ce=False):
    rare_vocab = _token_masks(rows) if rare_vocab is None else rare_vocab
    sums = {cell: 0.0 for cell in ('global', 'copy', 'novel_freq', 'novel_rare')}
    counts = {cell: 0 for cell in sums}
    row_values = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8].to(DEV)
        idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        logits = fwd_arm(idx, all_attention, TWALL, all_mlps).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), reduction='none').view_as(targets)
        valid = torch.ones_like(targets, dtype=torch.bool)
        valid[:, :64] = False
        copy = torch.zeros_like(valid)
        for lag in range(64):
            past = torch.roll(idx, lag, dims=1)
            if lag:
                past[:, :lag] = -1
            copy |= past == targets
        copy &= valid
        rare = rare_vocab[targets] & valid
        masks = {
            'global': valid,
            'copy': copy,
            'novel_freq': valid & ~copy & ~rare,
            'novel_rare': valid & ~copy & rare,
        }
        for cell, select in masks.items():
            sums[cell] += float(ce[select].sum())
            counts[cell] += int(select.sum())
        if retain_row_ce:
            row_values.extend(ce[:, 64:].mean(1).detach().cpu().tolist())
    result = {'ce': {cell: sums[cell] / max(counts[cell], 1) for cell in sums},
              'counts': counts}
    if retain_row_ce:
        result['row_global_ce'] = row_values
    return result


def _set_content_arm(site, basis, state):
    CONTENT_CORR.update({'on': True, 'site': site, 'basis': basis,
                         'weight': state['weight'], 'bias': state['bias']})


def _train_content_arm(site, basis, train_rows, validation_rows, TWALL,
                       all_attention, all_mlps, seed, steps=120):
    """Optimize the restricted correction on end-to-end full-ship CE."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.zeros(D, basis.shape[1], device=DEV, requires_grad=True)
    bias = torch.zeros(basis.shape[1], device=DEV, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=2e-3, weight_decay=1e-5)
    _set_content_arm(site, basis, {'weight': weight, 'bias': bias})
    best = None
    curve = []
    for step in range(1, steps + 1):
        chosen = torch.randint(len(train_rows), (2,), generator=generator)
        batch = train_rows[chosen].to(DEV)
        idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        logits = fwd_arm(idx, all_attention, TWALL, all_mlps).float()
        per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    targets.reshape(-1), reduction='none').view_as(targets)
        loss = per_token[:, 64:].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weight, bias], 5.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == steps:
            validation = _score_content_rows(validation_rows, TWALL, all_attention,
                                             all_mlps)['ce']['global']
            curve.append({'step': step, 'train_ce': float(loss.detach()),
                          'validation_ce': validation})
            if best is None or validation < best['validation_ce']:
                best = {'validation_ce': validation,
                        'weight': weight.detach().clone(),
                        'bias': bias.detach().clone(), 'step': step}
            print(f"content correction site={site} step={step} "
                  f"train={float(loss):.4f} validation={validation:.4f}", flush=True)
    _set_content_arm(site, basis, best)
    return best, curve


def run_content_correction(TWALL, all_attention, start_time):
    """Run the preregistered price-matched whole-ship content-interface test."""
    from pathlib import Path

    rank = 48
    seed = 271828
    steps = 120
    all_mlps = frozenset(range(18))
    train_rows = cl.fineweb_rows(192, skip=2000)[:, :T + 1].contiguous()
    validation_rows = cl.fineweb_rows(64, skip=5000)[:, :T + 1].contiguous()
    discovery_rows = cl.fineweb_rows(480, skip=7000)[:, :T + 1].contiguous()
    heldout_rows = cl.fineweb_rows(480, skip=11000)[:, :T + 1].contiguous()
    factor_path = (Path(__file__).resolve().parent.parent / 'polynomial_causal' /
                   'content_product_frontier_factors.pt')
    factors = torch.load(factor_path, map_location='cpu')
    content_basis = factors['sites']['0']['content_basis'][:, :rank].to(DEV).float()
    assert torch.allclose(content_basis.T @ content_basis,
                          torch.eye(rank, device=DEV), atol=2e-4, rtol=2e-4)

    CONTENT_CORR['on'] = False
    validation_baseline = _score_content_rows(validation_rows, TWALL, all_attention,
                                              all_mlps)
    trained = {}
    curves = {}
    for site in (0, 1, 2):
        state, curve = _train_content_arm(site, content_basis, train_rows,
                                          validation_rows, TWALL, all_attention,
                                          all_mlps, seed + site, steps)
        trained[f'content_mlp{site}'] = {'site': site, 'basis': content_basis,
                                        'weight': state['weight'],
                                        'bias': state['bias'],
                                        'validation_ce': state['validation_ce'],
                                        'best_step': state['step']}
        curves[f'content_mlp{site}'] = curve
    winner = min((0, 1, 2),
                 key=lambda site: trained[f'content_mlp{site}']['validation_ce'])

    random_generator = torch.Generator(device=DEV).manual_seed(seed + 1000)
    random_matrix = torch.randn(D, rank, device=DEV, generator=random_generator)
    random_basis = torch.linalg.qr(random_matrix, mode='reduced').Q
    random_state, random_curve = _train_content_arm(
        winner, random_basis, train_rows, validation_rows, TWALL, all_attention,
        all_mlps, seed + 2000, steps)
    trained['random_winner'] = {'site': winner, 'basis': random_basis,
                                'weight': random_state['weight'],
                                'bias': random_state['bias'],
                                'validation_ce': random_state['validation_ce'],
                                'best_step': random_state['step']}
    curves['random_winner'] = random_curve

    evaluations = {}
    for split, rows in (('discovery', discovery_rows), ('heldout', heldout_rows)):
        CONTENT_CORR['on'] = False
        evaluations[split] = {
            'ship_baseline': _score_content_rows(rows, TWALL, all_attention, all_mlps)
        }
        for arm, state in trained.items():
            _set_content_arm(state['site'], state['basis'], state)
            evaluations[split][arm] = _score_content_rows(
                rows, TWALL, all_attention, all_mlps)

    heldout = evaluations['heldout']
    winner_name = f'content_mlp{winner}'
    baseline = heldout['ship_baseline']['ce']
    selected = heldout[winner_name]['ce']
    random_ce = heldout['random_winner']['ce']
    gains = {cell: baseline[cell] - selected[cell] for cell in baseline}
    split_winners = {
        split: min((0, 1, 2),
                   key=lambda site: evaluations[split][f'content_mlp{site}']['ce']['global'])
        for split in evaluations
    }
    decisions = {
        'A_heldout_global_gain_ge_0p05': gains['global'] >= 0.05,
        'B_novel_rare_gain_ge_10pct_ship_excess': gains['novel_rare'] >= 0.11755031378547045,
        'C_no_copy_or_novel_freq_regression_gt_0p01': all(
            selected[cell] - baseline[cell] <= 0.01 for cell in ('copy', 'novel_freq')),
        'D_content_beats_matched_random_by_0p02_global': (
            random_ce['global'] - selected['global'] >= 0.02),
        'E_site_winner_stable_validation_discovery_heldout': all(
            site == winner for site in split_winners.values()),
    }
    standalone_parameters = 2 * D * rank + rank
    existing_glue_parameters = D + D * 32 + (2 * D) * 32
    output = {
        'config': {
            'model': 'bilin18', 'sites': [0, 1, 2], 'basis_rank': rank,
            'train_rows': 192, 'validation_rows': 64,
            'discovery_rows': 480, 'heldout_rows': 480,
            'train_skip': 2000, 'validation_skip': 5000,
            'discovery_skip': 7000, 'heldout_skip': 11000,
            'steps_per_arm': steps, 'batch_sequences': 2, 'seed': seed,
            'objective': 'end-to-end CE with complete current ship live',
            'baseline': 'current ship including existing generic rank-32 MLP2 glue',
        },
        'pricing': {
            'content_or_random_standalone_parameters': standalone_parameters,
            'existing_generic_mlp2_glue_parameters': existing_glue_parameters,
            'ratio_to_existing_glue': standalone_parameters / existing_glue_parameters,
            'rule': 'both input map and frozen output basis count at standalone price',
        },
        'validation_ship_baseline': validation_baseline,
        'validation_ce': {arm: state['validation_ce'] for arm, state in trained.items()},
        'validation_winner_site': winner,
        'split_winner_sites': split_winners,
        'evaluations': evaluations,
        'heldout_winner_gains': gains,
        'heldout_content_minus_random_global_ce': selected['global'] - random_ce['global'],
        'decisions': decisions,
        'training_curves': curves,
        'runtime_s': round(time.time() - start_time, 1),
    }
    out_path = Path(__file__).resolve().parent / 'ship_content_correction_results.json'
    params_path = Path(__file__).resolve().parent / 'ship_content_correction_params.pt'
    out_path.write_text(json.dumps(output, indent=2) + '\n')
    torch.save({
        'config': output['config'], 'pricing': output['pricing'],
        'validation_winner_site': winner,
        'arms': {arm: {key: (value.detach().cpu() if torch.is_tensor(value) else value)
                       for key, value in state.items()}
                 for arm, state in trained.items()},
    }, params_path)
    CONTENT_CORR['on'] = False
    print(json.dumps({'winner': winner, 'gains': gains,
                      'decisions': decisions}, indent=2), flush=True)
    print(f"wrote {out_path} and {params_path} ({output['runtime_s']}s)", flush=True)


def _paired_bootstrap_gain(baseline_rows, arm_rows, seed, draws=2000):
    baseline = torch.tensor(baseline_rows, dtype=torch.float64)
    arm = torch.tensor(arm_rows, dtype=torch.float64)
    difference = baseline - arm
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(difference), (draws, len(difference)),
                            generator=generator)
    boot = difference[indices].mean(1)
    return {
        'mean': float(difference.mean()),
        'ci95': [float(torch.quantile(boot, 0.025)),
                 float(torch.quantile(boot, 0.975))],
    }


def run_oracle_content_screen(TWALL, all_attention, start_time):
    """Optimizer-free screen for content alignment of the missing early MLP map."""
    from pathlib import Path

    rank = 64
    support_rank = 256
    nulls = 20
    seed = 161803
    all_mlps = frozenset(range(18))
    basis_rows = cl.fineweb_rows(96, skip=1200)[:, :T + 1].contiguous()
    discovery_rows = cl.fineweb_rows(192, skip=7000)[:, :T + 1].contiguous()
    heldout_rows = cl.fineweb_rows(192, skip=11000)[:, :T + 1].contiguous()
    factor_path = (Path(__file__).resolve().parent.parent / 'polynomial_causal' /
                   'content_product_frontier_factors.pt')
    factors = torch.load(factor_path, map_location='cpu')
    content_basis = factors['sites']['0']['content_basis'][:, :rank].to(DEV).float()
    assert torch.allclose(content_basis.T @ content_basis,
                          torch.eye(rank, device=DEV), atol=2e-4, rtol=2e-4)

    # Capture the exact original-minus-deployed-plank residual on an independent
    # corpus with all other ship components and the incumbent MLP2 glue live.
    CONTENT_CORR['on'] = False
    ORACLE_CORR.update({'on': False, 'capture': {0: [], 1: [], 2: []}})
    with torch.no_grad():
        for start in range(0, len(basis_rows), 8):
            idx = basis_rows[start:start + 8, :-1].to(DEV).contiguous()
            fwd_arm(idx, all_attention, TWALL, all_mlps)
    captured = {site: torch.cat(parts).reshape(-1, D)
                for site, parts in ORACLE_CORR['capture'].items()}
    ORACLE_CORR['capture'] = None

    bases = {}
    fit_metrics = {}
    for site, cpu_residual in captured.items():
        residual = cpu_residual.to(DEV)
        _, _, vectors = torch.pca_lowrank(residual, q=support_rank,
                                          center=False, niter=4)
        local_basis = vectors[:, :rank].contiguous()
        support = vectors[:, :support_rank].contiguous()

        def correction_rms(basis):
            coefficients = residual @ basis
            return math.sqrt(float(coefficients.double().square().sum()) /
                             (len(residual) * D))

        content_rms = correction_rms(content_basis)
        full_rms = math.sqrt(float(residual.double().square().mean()))
        site_bases = {
            'full': {'basis': None, 'scale': 1.0, 'fit_correction_rms': full_rms},
            'content': {'basis': content_basis, 'scale': 1.0,
                        'fit_correction_rms': content_rms},
            'local_pca': {'basis': local_basis, 'scale': 1.0,
                          'fit_correction_rms': correction_rms(local_basis)},
        }
        generator = torch.Generator(device=DEV).manual_seed(seed + site)
        for null_index in range(nulls):
            coordinates = torch.randn(support_rank, rank, device=DEV,
                                      generator=generator)
            haar = torch.linalg.qr(coordinates, mode='reduced').Q
            basis = (support @ haar).contiguous()
            raw_rms = correction_rms(basis)
            scale = content_rms / max(raw_rms, 1e-12)
            site_bases[f'null_{null_index:02d}'] = {
                'basis': basis, 'scale': scale,
                'fit_correction_rms': raw_rms * scale,
                'raw_fit_correction_rms': raw_rms,
            }
        bases[site] = site_bases
        fit_metrics[site] = {
            name: {key: value for key, value in row.items() if key != 'basis'}
            for name, row in site_bases.items()
        }
        del residual, vectors, support
        torch.cuda.empty_cache()

    # Freeze discovery-derived token-frequency strata for both evaluation splits.
    rare_vocab = _token_masks(discovery_rows)
    evaluations = {}
    for split, rows in (('discovery', discovery_rows), ('heldout', heldout_rows)):
        ORACLE_CORR['on'] = False
        baseline = _score_content_rows(rows, TWALL, all_attention, all_mlps,
                                       rare_vocab=rare_vocab, retain_row_ce=True)
        evaluations[split] = {'ship_baseline': baseline, 'sites': {}}
        for site in (0, 1, 2):
            evaluations[split]['sites'][str(site)] = {}
            for arm, row in bases[site].items():
                ORACLE_CORR.update({'on': True, 'site': site,
                                    'basis': row['basis'], 'scale': row['scale']})
                evaluations[split]['sites'][str(site)][arm] = _score_content_rows(
                    rows, TWALL, all_attention, all_mlps, rare_vocab=rare_vocab,
                    retain_row_ce=True)
                print(f"oracle screen {split} site={site} arm={arm} done", flush=True)

    gains = {}
    decisions = {}
    for site in (0, 1, 2):
        key = str(site)
        gains[key] = {}
        for split in ('discovery', 'heldout'):
            base = evaluations[split]['ship_baseline']
            gains[key][split] = {}
            for arm, scored in evaluations[split]['sites'][key].items():
                gains[key][split][arm] = {
                    'global': _paired_bootstrap_gain(
                        base['row_global_ce'], scored['row_global_ce'],
                        seed + 10000 * site + (0 if split == 'discovery' else 5000)
                        + sum(map(ord, arm))),
                    'cell_ce_gain': {
                        cell: base['ce'][cell] - scored['ce'][cell]
                        for cell in ('copy', 'novel_freq', 'novel_rare')
                    },
                }
        heldout_nulls = [gains[key]['heldout'][f'null_{j:02d}']['global']['mean']
                         for j in range(nulls)]
        null95 = float(torch.quantile(torch.tensor(heldout_nulls), 0.95))
        content_discovery = gains[key]['discovery']['content']['global']
        content_heldout = gains[key]['heldout']['content']['global']
        full_heldout = gains[key]['heldout']['full']['global']
        decisions[key] = {
            'full_oracle_ci95_lower_gt_zero': full_heldout['ci95'][0] > 0.0,
            'content_positive_both_splits': (
                content_discovery['mean'] > 0.0 and content_heldout['mean'] > 0.0),
            'content_beats_matched_null95_heldout': content_heldout['mean'] > null95,
            'content_heldout_gain': content_heldout['mean'],
            'matched_null95_heldout_gain': null95,
            'content_fraction_of_full_oracle_gain': (
                content_heldout['mean'] / full_heldout['mean']
                if abs(full_heldout['mean']) > 1e-12 else None),
        }

    output = {
        'config': {
            'model': 'bilin18', 'ship': 'current K=3072 ship with MLP2 glue live',
            'sites': [0, 1, 2], 'projection_rank': rank,
            'matched_null_support_rank': support_rank,
            'matched_nulls_per_site': nulls, 'seed': seed,
            'basis_rows': 96, 'basis_skip': 1200,
            'discovery_rows': 192, 'discovery_skip': 7000,
            'heldout_rows': 192, 'heldout_skip': 11000,
            'copy_definition': 'target recurs at distance 1 through 64 in context',
            'frequency_vocab': 'frozen from discovery rows and reused on heldout',
            'status': 'optimizer-free singleton oracle screen; not a learned correction',
        },
        'arms': {
            'full': 'exact live original MLP output minus deployed plank output',
            'content': 'full residual projected through frozen deep content basis',
            'local_pca': 'full residual projected through fit-corpus top residual PCs',
            'matched_null': 'Haar subspace inside local top-256 residual support, scaled to content correction RMS',
        },
        'fit_correction_metrics': fit_metrics,
        'evaluations': evaluations,
        'paired_gains': gains,
        'site_decisions': decisions,
        'training_license_sites': [int(site) for site, row in decisions.items()
                                   if row['full_oracle_ci95_lower_gt_zero']
                                   and row['content_positive_both_splits']
                                   and row['content_beats_matched_null95_heldout']],
        'interpretation_guardrail': (
            'A passing projection licenses prediction of the missing original '
            'computation. It does not by itself establish semantic content, causal '
            'transport, or a simpler learned program.'),
        'runtime_s': round(time.time() - start_time, 1),
    }
    out_path = Path(__file__).resolve().parent / 'ship_content_oracle_screen_results.json'
    out_path.write_text(json.dumps(output, indent=2) + '\n')
    ORACLE_CORR['on'] = False
    print(json.dumps({'decisions': decisions,
                      'training_license_sites': output['training_license_sites']},
                     indent=2), flush=True)
    print(f"wrote {out_path} ({output['runtime_s']}s)", flush=True)


@torch.no_grad()
def run_factorial(discovery_rows, TWALL, all_attention, start_time):
    """Score the complete A x M012 x deep cube on two disjoint row sets."""
    poly = str((__import__('pathlib').Path(__file__).resolve().parent.parent /
                'polynomial_causal'))
    if poly not in sys.path:
        sys.path.insert(0, poly)
    from factorial_causal_attribution import analyze_cells, powerset

    groups = ('attention', 'mlp012', 'deep')
    heldout_rows = cl.fineweb_rows(len(discovery_rows), skip=11000)[:, :T + 1].contiguous()
    row_splits = {'discovery': discovery_rows, 'heldout': heldout_rows}
    out_path = PT + 'ship_error_factorial_results.json'

    def masks_for_rows(rows):
        counts = torch.zeros(50257)
        for start in range(0, len(rows), 8):
            targets = rows[start:start + 8, 1:]
            valid = torch.ones_like(targets, dtype=torch.bool)
            valid[:, :64] = False
            counts.index_add_(0, targets.reshape(-1), valid.reshape(-1).float())
        threshold = counts.sort(descending=True).values[500]
        rare = counts < threshold
        top100 = torch.zeros(50257, dtype=torch.bool)
        top100[counts.argsort(descending=True)[:100]] = True
        return rare.to(DEV), top100.to(DEV)

    split_results = {}
    raw_values = {}
    for split, rows in row_splits.items():
        rare_vocab, top100_vocab = masks_for_rows(rows)
        cell_sums = {
            'primary': {cell: {} for cell in ('copy', 'novel_freq', 'novel_rare')},
            'frequency': {cell: {} for cell in ('top100', 'non_top100')},
        }
        cell_counts = {'primary': None, 'frequency': None}

        for arm in powerset(groups):
            layers = all_attention if 'attention' in arm else frozenset()
            mlps = set()
            if 'mlp012' in arm:
                mlps.update((0, 1, 2))
            if 'deep' in arm:
                mlps.update(range(3, 18))
            mlps = frozenset(mlps)
            sums = {cell: 0.0 for cell in ('copy', 'novel_freq', 'novel_rare',
                                                   'top100', 'non_top100')}
            counts = {cell: 0 for cell in sums}
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8].to(DEV)
                idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
                logits = fwd_arm(idx, layers, TWALL, mlps).float()
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     targets.reshape(-1), reduction='none').view_as(targets)
                valid = torch.ones_like(targets, dtype=torch.bool)
                valid[:, :64] = False
                copy = torch.zeros_like(valid)
                for lag in range(1, 65):
                    past = torch.roll(idx, lag, dims=1)
                    past[:, :lag] = -1
                    copy |= past == targets
                copy &= valid
                rare = rare_vocab[targets] & valid
                top = top100_vocab[targets] & valid
                masks = {
                    'copy': copy,
                    'novel_freq': valid & ~copy & ~rare,
                    'novel_rare': valid & ~copy & rare,
                    'top100': top,
                    'non_top100': valid & ~top,
                }
                for cell, select in masks.items():
                    sums[cell] += float(ce[select].sum())
                    counts[cell] += int(select.sum())

            for cell in cell_sums['primary']:
                cell_sums['primary'][cell][arm] = sums[cell] / max(counts[cell], 1)
            for cell in cell_sums['frequency']:
                cell_sums['frequency'][cell][arm] = sums[cell] / max(counts[cell], 1)
            primary_counts = {cell: counts[cell] for cell in cell_sums['primary']}
            frequency_counts = {cell: counts[cell] for cell in cell_sums['frequency']}
            if cell_counts['primary'] is None:
                cell_counts = {'primary': primary_counts, 'frequency': frequency_counts}
            else:
                assert cell_counts['primary'] == primary_counts
                assert cell_counts['frequency'] == frequency_counts
            print(f"{split} arm={'+'.join(arm) or 'clean'} done", flush=True)

        split_results[split] = {
            'primary': analyze_cells(groups, cell_counts['primary'], cell_sums['primary']),
            'frequency': analyze_cells(groups, cell_counts['frequency'], cell_sums['frequency']),
        }
        raw_values[split] = {
            partition: {
                cell: {'+'.join(arm) if arm else 'clean': value for arm, value in values.items()}
                for cell, values in cells.items()
            }
            for partition, cells in cell_sums.items()
        }

    def group_share(split, cell, group):
        row = split_results[split]['primary']['cells'][cell]
        return row['shapley'][group] / max(abs(row['total_effect']), 1e-30)

    early_license = all(
        split_results[split]['primary']['cells']['novel_rare']['shapley']['mlp012'] >= 0.05
        and group_share(split, 'novel_rare', 'mlp012') >= 0.20
        for split in row_splits
    )
    interaction_material = {
        split: {
            cell: split_results[split]['primary']['cells'][cell]
                  ['interaction_l1_fraction_of_total'] >= 0.20
            for cell in ('copy', 'novel_freq', 'novel_rare')
        }
        for split in row_splits
    }
    dominant = {
        split: {
            cell: max(groups, key=lambda group: abs(split_results[split]['primary']
                                                     ['cells'][cell]['shapley'][group]))
            for cell in ('copy', 'novel_freq', 'novel_rare')
        }
        for split in row_splits
    }
    stable = all(dominant['discovery'][cell] == dominant['heldout'][cell]
                 for cell in dominant['discovery'])
    closure = all(
        abs(split_results[split][partition]['weighted_shapley_closure_error']) <= 1e-8
        for split in row_splits for partition in ('primary', 'frequency')
    )
    result = {
        'config': {
            'model': 'bilin18', 'groups': list(groups),
            'rows_per_split': len(discovery_rows),
            'discovery_skip': 7000, 'heldout_skip': 11000,
            'status': 'token_cell_factorial_stage; output/intervention cross-tab pending',
        },
        'splits': split_results,
        'raw_cell_ce': raw_values,
        'dominant_group': dominant,
        'interaction_material_20pct': interaction_material,
        'decisions': {
            'early_mlp012_novel_rare_license': early_license,
            'dominant_group_stable': stable,
            'shapley_closure': closure,
        },
        'runtime_s': round(time.time() - start_time, 1),
    }
    with open(out_path, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result['decisions'], indent=2), flush=True)
    print(f"wrote {out_path} ({result['runtime_s']}s)", flush=True)


def main(factorial=False, content_correction=False, oracle_content_screen=False):
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    for p in m.parameters():
        p.requires_grad_(False)
    eval_rows = 480 if factorial else NR
    EVR = cl.fineweb_rows(eval_rows, skip=7000)[:, :T + 1].contiguous()

    def ce_run(layers, TWALL):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, layers, TWALL).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    # per-layer xin covariance whiteners (96 rows)
    CR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    XACC = {L: torch.zeros(D, D, device=DEV) for L in LAYERS}
    ncov = 0
    for i in range(0, 96, 8):
        idx = CR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1_ = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            if L in XACC:
                Xf = xin.float().reshape(-1, D)
                XACC[L] += Xf.T @ Xf
            x, v1_ = blk(x, v1_, x0)
        ncov += idx.shape[0] * T
    WHITEN = {}
    for L in LAYERS:
        Sg = XACC[L] / ncov
        ev, V = torch.linalg.eigh(Sg)
        ev = ev.clamp_min(1e-6)
        WHITEN[L] = (V @ torch.diag(ev.sqrt()) @ V.T,
                     V @ torch.diag(ev.rsqrt()) @ V.T)
    print("whiteners built", flush=True)

    TWALL = {}
    for LT in LAYERS:
        at = H[LT].attn
        if LT in PLAIN:
            eye = torch.eye(D, device=DEV)
            Wh, Whi = eye, eye
        else:
            Wh, Whi = WHITEN[LT]
        TWALL[LT] = {'q': trunc_perhead(at.c_q.weight, 32, Wh, Whi),
                     'k': trunc_perhead(at.c_k.weight, 32, Wh, Whi),
                     'q2': trunc_perhead(at.c_q2.weight, 32, Wh, Whi),
                     'k2': trunc_perhead(at.c_k2.weight, 32, Wh, Whi)}
    print("attn maps built", flush=True)

    # ---- context-fit the mlp0/mlp1 planks UNDER the attention composite ----
    FITR = cl.fineweb_rows(480, skip=80)[:, :T + 1].contiguous()
    ALLL = frozenset(LAYERS)
    caps = {n: [] for n in ('a0', 'm0', 'a1', 'm1')}
    hks = [H[0].attn.c_proj.register_forward_hook(
               lambda mo, a, o: caps['a0'].append(o.detach().float().cpu())),
           H[0].mlp.register_forward_hook(
               lambda mo, a, o: caps['m0'].append(o.detach().float().cpu())),
           H[1].attn.c_proj.register_forward_hook(
               lambda mo, a, o: caps['a1'].append(o.detach().float().cpu())),
           H[1].mlp.register_forward_hook(
               lambda mo, a, o: caps['m1'].append(o.detach().float().cpu()))]
    for i in range(0, 480, 8):
        fwd_arm(FITR[i:i + 8, :-1].to(DEV).contiguous(), ALLL, TWALL)
    for h in hks:
        h.remove()
    FT = {n: torch.cat(v) for n, v in caps.items()}
    toksF = FITR[:, :-1].reshape(-1)
    print("context capture done", flush=True)

    def tier_table(Y, toks):
        tsum = torch.zeros(50257, D); tcnt = torch.zeros(50257)
        tsum.index_add_(0, toks, Y); tcnt.index_add_(0, toks,
                                                     torch.ones(toks.shape[0]))
        gm = Y.mean(0)
        TAB = torch.where(tcnt.unsqueeze(1) > 0,
                          tsum / tcnt.clamp_min(1).unsqueeze(1), gm.unsqueeze(0))
        Tc = (TAB - gm).to(DEV)
        U_, S_, Vt_ = torch.svd_lowrank(Tc, q=96, niter=4)
        base = gm.to(DEV) + (U_[:, :64] * S_[:64]) @ Vt_[:, :64].T
        keep = tcnt.argsort(descending=True)[:2000]
        TIER = base.clone(); TIER[keep] = TAB[keep].to(DEV)
        return TIER

    def ridge(X, Y):
        Xg = X.to(DEV); Yg = Y.to(DEV)
        xm_ = Xg.mean(0); ym_ = Yg.mean(0)
        Xc = Xg - xm_; Yc = Yg - ym_
        XtX = Xc.T @ Xc
        lam = 0.01 * float(torch.diagonal(XtX).mean())
        W = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=DEV),
                               Xc.T @ Yc)
        return W, xm_, ym_

    T0 = tier_table(FT['m0'].reshape(-1, D), toksF)
    R0Y = FT['m0'].reshape(-1, D) - T0[toksF].cpu()
    SHIP['t0'] = T0; SHIP['r0'] = ridge(FT['a0'].reshape(-1, D), R0Y)
    T1 = tier_table(FT['m1'].reshape(-1, D), toksF)
    R1Y = FT['m1'].reshape(-1, D) - T1[toksF].cpu()
    X1 = torch.cat([FT['a1'], FT['m0']], -1).reshape(-1, 2 * D)
    SHIP['t1'] = T1; SHIP['r1'] = ridge(X1, R1Y)
    print("planks 0/1 context-fit", flush=True)

    # mlp2 ridge (all3-context) + mlp3 projection stats via one more capture pass
    cap2 = {}
    for i in range(0, 480, 8):
        fwd_arm(FITR[i:i + 8, :-1].to(DEV).contiguous(), ALLL, TWALL,
                frozenset({0, 1}), cap=cap2)
    C2 = {k: torch.cat(v) for k, v in cap2.items()}
    X2f = torch.cat([C2['a2'],
                     C2['mstream'].reshape(-1, T, 17 * D)[:, :, D:2 * D]
                     .reshape(C2['a2'].shape)], -1).reshape(-1, 2 * D)
    SHIP['r2'] = ridge(X2f, C2['m2'].reshape(-1, D))
    M3f = C2['mstream'].reshape(-1, T, 17 * D)[:, :, 3 * D:4 * D].reshape(-1, D)
    SHIP['mean3'] = M3f.mean(0).to(DEV)
    U3, S3, V3 = torch.svd_lowrank((M3f - M3f.mean(0)).to(DEV), q=280, niter=4)
    SHIP['p3'] = V3[:, :256].T
    print("planks 2/3 fit", flush=True)

    # unit sub-MLPs for 4-17: rankings from 96 clean rows
    HR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    store = {}
    def mk_pre(L):
        def hk(mod, args):
            store.setdefault(L, []).append(args[0].detach())
            return None
        return hk
    pre_hooks = [H[L].mlp.register_forward_pre_hook(mk_pre(L))
                 for L in range(4, 18)]
    acc1 = {L: 0 for L in range(4, 18)}; acc2 = {L: 0 for L in range(4, 18)}
    n0 = 0
    for i in range(0, 96, 8):
        store.clear()
        idxh = HR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idxh), (D,)); x0 = x; v1_ = None
        for L, blk in enumerate(H):
            x, v1_ = blk(x, v1_, x0)
        for L in range(4, 18):
            zz = store[L][0]
            hh_ = (H[L].mlp.Left(zz).float() * H[L].mlp.Right(zz).float()) \
                .reshape(-1, H[L].mlp.Left.weight.shape[0])
            acc1[L] = acc1[L] + hh_.sum(0); acc2[L] = acc2[L] + (hh_ * hh_).sum(0)
        n0 += 8 * T
    for h in pre_hooks:
        h.remove()
    KU = 3072
    for L in range(4, 18):
        mu = acc1[L] / n0
        hsd = (acc2[L] / n0 - mu * mu).clamp_min(0).sqrt()
        score = hsd * H[L].mlp.Down.weight.float().norm(dim=0)
        topu = score.argsort(descending=True)[:KU]
        SHIP[f'u{L}'] = {'l': H[L].mlp.Left.weight.float()[topu].clone(),
                         'r': H[L].mlp.Right.weight.float()[topu].clone(),
                         'd': H[L].mlp.Down.weight.float()[:, topu].clone(),
                         'b': H[L].mlp.Down_bias.detach().float().clone()}
    print("unit planks built", flush=True)

    gp = torch.load(PT + 'mlp2_glue_params.pt', map_location=DEV)
    CORR['b'] = gp['b'].to(DEV); CORR['U'] = gp['U'].to(DEV)
    CORR['V'] = gp['V'].to(DEV)
    CORR['on'] = True

    if factorial:
        run_factorial(EVR, TWALL, ALLL, t0)
        return
    if oracle_content_screen:
        run_oracle_content_screen(TWALL, ALLL, t0)
        return
    if content_correction:
        run_content_correction(TWALL, ALLL, t0)
        return

    def ce2(layers, mlps, rows=None):
        rows = EVR if rows is None else rows
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = rows[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, layers, TWALL, mlps).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    ALLM = frozenset(range(18))
    # find the 100 most frequent target tokens on eval rows
    ecnt = torch.zeros(50257)
    for i in range(0, NR, 8):
        ecnt.index_add_(0, EVR[i:i + 8, 1:].reshape(-1), torch.ones(8 * T))
    top100 = ecnt.argsort(descending=True)[:100]
    FREQM = torch.zeros(50257, dtype=torch.bool); FREQM[top100] = True

    def freq_ce(layers, mlps):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, layers, TWALL, mlps).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = FREQM.to(DEV)[tg] & mk
            s_ += float(ce[cm].sum()); n_ += int(cm.sum())
        return s_ / max(n_, 1)

    c0 = freq_ce(frozenset(), frozenset())
    c_attn = freq_ce(ALLL, frozenset())
    c_m012 = freq_ce(ALLL, frozenset({0, 1, 2}))
    c_full = freq_ce(ALLL, ALLM)
    d_attn = c_attn - c0
    d_m012 = c_m012 - c_attn
    d_deep = c_full - c_m012
    total = c_full - c0
    pa = d_attn <= 0.35 * total
    pb = d_m012 >= 0.35 * total
    pc = abs((d_attn + d_m012 + d_deep) - total) <= 0.15 * total
    out = {'freq_ce': {'clean': round(c0, 4), 'attn': round(c_attn, 4),
                       'attn_m012': round(c_m012, 4), 'full': round(c_full, 4)},
           'increments': {'attention': round(d_attn, 4),
                          'mlp012_planks': round(d_m012, 4),
                          'deep_unit_planks': round(d_deep, 4)},
           'shares': {'attention': round(d_attn / max(total, 1e-6), 3),
                      'mlp012': round(d_m012 / max(total, 1e-6), 3),
                      'deep': round(d_deep / max(total, 1e-6), 3)},
           'pred_a_attn_le_35': bool(pa), 'pred_b_m012_ge_35': bool(pb),
           'pred_c_additive_15': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out['shares'])
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
