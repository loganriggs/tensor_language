#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_integrity pred_b_typed_beats_untyped pred_c_data_relieves_overfitting pred_d_large_outputs_kept pred_e_cross_start_stability
"""Typed vs untyped correction with seven times the fitting data (BIDEGREE_DATA_ABLATION_PLAN_V1.md).
Same arms, capacity, seeds and loop as run_bidegree_correction_v1 (whose functions are imported), but fitted on 38,912 states from 608 FineWeb
documents outside both existing panels (held-out 64 documents for the snapshot), with the exact teacher F4(x) as target and an unweighted
relative squared error; the sensitivity-weighted calibration error is a held-out secondary metric; the fresh panel and the matched pairs are
scored exactly as before and never enter fitting.
pred_a_integrity: S+C replays stored rows (calibration, fresh) and x (new panel) <= 1e-4; teacher replays stored calibration targets <= 1e-4;
                  exports reload to <= 1e-8; the previous rung's planted controls pass.
pred_b_typed_beats_untyped: both seeds, typed fresh small-output value RMS and response RMS <= .85 x untyped.
pred_c_data_relieves_overfitting: both seeds and both arms, fresh small-output value RMS <= .45.
pred_d_large_outputs_kept: both seeds, typed fresh value error on outputs 0-3 <= parent's; output 3 <= .9 x parent's.
pred_e_cross_start_stability: typed cross-start fresh cosine over outputs 4-15: mean >= .9 and min >= .8.
Price: 84 + 12 + 32 forwards (bar 140); 0 model backwards; 4 fits x 400 Adam updates of 442,464 parameters on 38,912 states.
"""
import os, sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; P = ROOT / 'basis_aligned/polynomial_causal/direct_tensor_match'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bidegree_correction_v1 import ATOMS, SEEDS, STEPS, LR, EVAL_EVERY, allocate, build_sources, atoms, profile, objective, errors, controls, split_states
FORWARDS_MAX = 140; VAL_DOCS = 64
BARS = dict(replay=1e-4, export=1e-8, typed_vs_untyped=.85, value=.45, output3=.9, cos_mean=.9, cos_min=.8)
INPUTS = ['BIDEGREE_DATA_ABLATION_PLAN_V1.md', 'BIDEGREE_CORRECTION_PLAN_V1.md', 'BIDEGREE_CORRECTION_V1.json', 'SENSITIVE_ROOT_CALIBRATION_V2.pt', 'NATIVE_QUARTIC_COVARIANCE_V1.pt',
          'QUARTIC_ADDITIONAL_STATES_V1.pt', 'MIXED_CP_FEATURES_SEED1001_V1.pt', 'EXPANDED_ROOT_EMPIRICAL_V1.pt', 'RESIDUAL_FRESH_STATES_V1.pt',
          'RESIDUAL_FRESH_TOKENS_V1.pt', 'ALL_FEATURE_MATCHED_RESPONSES_V1.json']
TOKENS_CAL = ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n96_skip1200.pt'
TOKENS_NEW = [ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n480_skip80.pt', ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt']


def main():
    import torch
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    plan = dict(candidate_id='bidegree_data_ablation.v1', forwards_max=FORWARDS_MAX, model_backwards=0, model_updates=0, fit_parameters=4 * (96 * 4 * 1152 + 96),
                fits=4, steps=STEPS, fit_states=608 * 64, gpu_accessed=False, model_loaded=False, execution_policy='managed_queue_only', bars=BARS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        plan['controls'] = controls(); print(json.dumps(plan, indent=2, sort_keys=True)); return
    sys.path.insert(0, str(P))
    from circuit_fast_screen_producer import Bilin18TorchBackend
    from run_source_geometry_census_v1 import quartic_pieces, parent_pieces
    torch.backends.cuda.matmul.allow_tf32 = False; start = time.monotonic(); dev = 'cuda'
    out = P / 'BIDEGREE_DATA_ABLATION_V1.json'; assert not out.exists()
    manifest = {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in INPUTS}
    manifest['tokens_calibration'] = hashlib.sha256(TOKENS_CAL.read_bytes()).hexdigest(); manifest['tokens_new'] = [hashlib.sha256(t.read_bytes()).hexdigest() for t in TOKENS_NEW]
    checks = controls()
    model = Bilin18TorchBackend.load('cuda').model.float(); forwards = 0
    # ---- stored panels ------------------------------------------------------------------------------------------------------------
    tokens = torch.load(TOKENS_CAL, weights_only=True)[:96, :64].contiguous()
    labels = torch.load(P / 'SENSITIVE_ROOT_CALIBRATION_V2.pt', weights_only=True); cov = torch.load(P / 'NATIVE_QUARTIC_COVARIANCE_V1.pt', weights_only=True)
    extra = torch.load(P / 'QUARTIC_ADDITIONAL_STATES_V1.pt', weights_only=True)
    digest = lambda z: hashlib.sha256(z.contiguous().numpy().tobytes()).hexdigest(); assert digest(tokens) == labels['token_sha256']['calibration']
    calx = torch.cat([cov['panels'][0]['rows'], extra['rows']]).to(dev).double(); caly = labels['panels'][0]['target'].to(dev).double()
    calw = labels['panels'][0]['weight'].to(dev).double(); calw = calw / calw.mean(0, keepdim=True)
    fresh = torch.load(P / 'RESIDUAL_FRESH_STATES_V1.pt', weights_only=True); fx = fresh['rows'].to(dev).double(); fy = fresh['target'].to(dev).double()
    ftok = torch.load(P / 'RESIDUAL_FRESH_TOKENS_V1.pt', weights_only=True)[:, :64].contiguous(); assert ftok.shape[0] == 256
    pairs = torch.tensor(json.loads((P / 'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs'], device=dev).T; don, rec = pairs[0], pairs[1]
    scale = float(caly[:, :4].abs().mean()); caly = caly / scale; fy = fy / scale
    p = torch.load(P / 'MIXED_CP_FEATURES_SEED1001_V1.pt', weights_only=True); pf = [f.to(dev).double() for f in p['factors']]; pC = p['coefficients'].to(dev).double() / scale
    parent = lambda x: torch.stack([x @ f.T for f in pf]).prod(0) @ pC.T
    cal_parent = parent(calx); fresh_parent = parent(fx); cal_r = caly - cal_parent
    # ---- teacher and readers ------------------------------------------------------------------------------------------------------
    state = model.state_dict(); w = lambda l, n: state[f'transformer.h.{l}.mlp.{n}.weight'].double()
    w16 = (w(16, 'Left'), w(16, 'Right'), w(16, 'Down')); w17 = (w(17, 'Left'), w(17, 'Right'), w(17, 'Down')); lam17 = float(model.transformer.h[17].lambdas[0])
    writer = torch.load(P / 'EXPANDED_ROOT_EMPIRICAL_V1.pt', weights_only=True)['writer'].to(dev).double(); vocab = state['lm_head.weight'].double()
    uw = vocab @ writer; readers = vocab.T @ uw / uw.square().sum(0); del vocab, uw
    teacher = lambda x: quartic_pieces(x, torch.zeros_like(x), w16, w17, lam17, readers)[0] / scale
    # ---- splits on all three panels -----------------------------------------------------------------------------------------------
    S, C, xn, fw = split_states(model, tokens, dev); forwards += fw; replay_cal = float(((S + C) - calx).norm() / calx.norm())
    fS, fC, fxn, fw = split_states(model, ftok, dev); forwards += fw; replay_fresh = float(((fS + fC) - fx).norm() / fx.norm())
    newtok = torch.cat([torch.load(t, weights_only=True)[:, :64] for t in TOKENS_NEW]).contiguous(); assert newtok.shape == (672, 64)
    nS, nC, nxn, fw = split_states(model, newtok, dev); forwards += fw; replay_new = float(((nS + nC) - nxn).norm() / nxn.norm())
    teacher_replay = float((teacher(calx) - caly).norm() / caly.norm())
    ny = teacher(nxn); new_parent = parent(nxn); new_r = ny - new_parent
    print(f'[split] calibration {replay_cal:.2e} | fresh {replay_fresh:.2e} | new {replay_new:.2e} | teacher vs stored targets {teacher_replay:.2e} | forwards {forwards}', flush=True)
    # ---- allocation (same rule as the previous rung, on the calibration census) ---------------------------------------------------
    tp = quartic_pieces(S, C, w16, w17, lam17, readers) / scale; pp = parent_pieces(S, C, pf, pC); rp = tp - pp
    share = torch.diagonal(torch.einsum('ing,jng->gij', rp, rp), dim1=1, dim2=2); share = (share / share.sum(1, keepdim=True)).tolist(); alloc = [allocate(s) for s in share]; del tp, pp, rp
    prev = json.loads((P / 'BIDEGREE_CORRECTION_V1.json').read_text()); assert alloc == prev['allocation'], 'allocation differs from the previous rung'
    # ---- fits on the new panel -----------------------------------------------------------------------------------------------------
    doc = torch.arange(672, device=dev).repeat_interleave(64); fit_m = doc < 672 - VAL_DOCS; val_m = ~fit_m; ones = torch.ones_like(new_r)
    panels = dict(fit=(nS[fit_m], nC[fit_m], nxn[fit_m], new_r[fit_m], ones[fit_m]), val=(nS[val_m], nC[val_m], nxn[val_m], new_r[val_m], ones[val_m]))
    baseline = errors(fresh_parent, fy, don, rec); rows = []; exports = {}
    cal_weighted = lambda pred: ((calw * (pred - caly).square()).sum(0) / (calw * caly.square()).sum(0)).sqrt().tolist()
    for arm in ('typed', 'untyped'):
        src = build_sources(alloc, typed=(arm == 'typed')).to(dev)
        for seed in SEEDS:
            torch.manual_seed(seed); reads = torch.randn(16, ATOMS, 4, 1152, dtype=torch.float64, device=dev, requires_grad=True)
            opt = torch.optim.Adam([reads], lr=LR); best = None; history = []; t0 = time.monotonic()
            for step in range(STEPS + 1):
                if step % EVAL_EVERY == 0:
                    with torch.no_grad():
                        vloss, _ = objective(reads, src, *panels['val']); floss, _ = objective(reads, src, *panels['fit'])
                    history.append(dict(step=step, fit=float(floss), val=float(vloss)))
                    if best is None or float(vloss) < best[0]:
                        best = (float(vloss), step, reads.detach().clone())
                if step == STEPS:
                    break
                with torch.enable_grad():
                    loss, _ = objective(reads, src, *panels['fit'])
                    opt.zero_grad(); loss.backward(); opt.step()
            reads_b = best[2]
            with torch.no_grad():
                A = atoms(reads_b, src, *panels['fit'][:3]); c = profile(A, panels['fit'][3], panels['fit'][4])
                fpred = fresh_parent + (atoms(reads_b, src, fS, fC, fxn) * c[None]).sum(-1); cpred = cal_parent + (atoms(reads_b, src, S, C, xn) * c[None]).sum(-1)
            m = errors(fpred, fy, don, rec); mc = dict(value=((cpred - caly).square().sum(0) / caly.square().sum(0)).sqrt().tolist(), sensitivity_weighted=cal_weighted(cpred))
            key = f'{arm}_seed{seed}'; exports[key] = dict(reads=(reads_b / reads_b.norm(dim=-1, keepdim=True)).cpu(), src=src.cpu(), coefficients=c.cpu(), scale=scale, allocation=alloc, arm=arm, seed=seed)
            rows.append(dict(arm=arm, seed=seed, selected_step=best[1], selected_val_objective=best[0], history=history, fit_seconds=time.monotonic() - t0, fresh=m, calibration=mc,
                             fresh_correction=(fpred - fresh_parent).cpu()))
            print(f'[{key}] step {best[1]} val {best[0]:.4f} fit-at-end {history[-1]["fit"]:.4f} | fresh small value {m["small_value_rms"]:.4f} response {m["small_response_rms"]:.4f} | calibration weighted small {(sum(v * v for v in mc["sensitivity_weighted"][4:]) / 12) ** .5:.4f} | outputs 0-3 {[round(v, 4) for v in m["value"][:4]]} (parent {[round(v, 4) for v in baseline["value"][:4]]}) | {time.monotonic() - t0:.0f}s', flush=True)
    def cosines(arm):
        a, b = [r['fresh_correction'] for r in rows if r['arm'] == arm]
        return ((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-300)).tolist()
    cos = {arm: cosines(arm) for arm in ('typed', 'untyped')}
    torch.save(exports, P / 'BIDEGREE_DATA_ABLATION_EXPORTS_V1.pt')
    reloaded = torch.load(P / 'BIDEGREE_DATA_ABLATION_EXPORTS_V1.pt', weights_only=False); export_err = 0.0
    for r in rows:
        e = reloaded[f'{r["arm"]}_seed{r["seed"]}']
        pred = fresh_parent + (atoms(e['reads'].to(dev), e['src'].to(dev), fS, fC, fxn) * e['coefficients'].to(dev)[None]).sum(-1)
        m = errors(pred, fy, don, rec); export_err = max(export_err, max(abs(x - y) for x, y in zip(m['value'] + m['response'], r['fresh']['value'] + r['fresh']['response'])))
        del r['fresh_correction']
    by = {(r['arm'], r['seed']): r['fresh'] for r in rows}
    ratio = {s: dict(value=by['typed', s]['small_value_rms'] / by['untyped', s]['small_value_rms'], response=by['typed', s]['small_response_rms'] / by['untyped', s]['small_response_rms']) for s in SEEDS}
    tcos = cos['typed'][4:]
    preds = dict(pred_a_integrity=max(replay_cal, replay_fresh, replay_new, teacher_replay) <= BARS['replay'] and export_err <= BARS['export'],
                 pred_b_typed_beats_untyped=all(ratio[s]['value'] <= BARS['typed_vs_untyped'] and ratio[s]['response'] <= BARS['typed_vs_untyped'] for s in SEEDS),
                 pred_c_data_relieves_overfitting=all(by[a, s]['small_value_rms'] <= BARS['value'] for a in ('typed', 'untyped') for s in SEEDS),
                 pred_d_large_outputs_kept=all(all(by['typed', s]['value'][o] <= baseline['value'][o] for o in range(4)) and by['typed', s]['value'][3] <= BARS['output3'] * baseline['value'][3] for s in SEEDS),
                 pred_e_cross_start_stability=(sum(tcos) / len(tcos) >= BARS['cos_mean'] and min(tcos) >= BARS['cos_min']))
    assert forwards <= FORWARDS_MAX, forwards
    cost = dict(atoms=96, reads_per_atom=4, read_multiply_adds=96 * 4 * 1152, products=96 * 3, coefficient_multiplies=96, adds=96, added_coefficients=96 * 4 * 1152 + 96)
    receipt = dict(predictions=preds, controls=checks, manifest=manifest, replay=dict(calibration=replay_cal, fresh=replay_fresh, new=replay_new, teacher=teacher_replay, export=export_err),
                   allocation=alloc, fit_states=int(fit_m.sum()), val_states=int(val_m.sum()), baseline=baseline, rows=rows, typed_over_untyped=ratio, cross_start_cosines=cos, scale=scale,
                   literal_cost=cost, forwards=forwards, backwards=0, seconds=time.monotonic() - start, peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
                   scope='Data-size ablation of the typed vs untyped correction; no feature names, no intervention, no circuit claim.')
    out.write_text(json.dumps(receipt, indent=1)); print(json.dumps(dict(predictions=preds, replay=receipt['replay'], typed_over_untyped=ratio, forwards=forwards), indent=1), flush=True)


if __name__ == '__main__':
    main()
