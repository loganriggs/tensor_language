#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_integrity pred_b_shared_not_worse pred_c_shared_better pred_d_cross_start_stability pred_e_bank_identified
"""A shared 96-atom bank versus output-local atoms at 3,040 documents (SHARED_BANK_CORRECTION_PLAN_V1.md).
SHARED: 96 quartic atoms (four unit reads of x each) used by all 16 outputs with per-output profiled coefficients (443,904 coefficients);
LOCAL: the previous receipt's output-local fits at 3,040 documents (BIDEGREE_DATA_SCALING_V1.json, not re-run). Same data, teacher, loop,
seeds and scoring as the previous rung (functions imported).
pred_a_integrity: replays <= 1e-4; teacher <= 1e-4; exports <= 1e-8; controls pass.
pred_b_shared_not_worse: both seeds, shared fresh small-output value RMS <= 1.05 x local and response RMS <= 1.05 x local (same seed).
pred_c_shared_better: both seeds, shared value RMS <= .95 x local and response RMS <= .95 x local.
pred_d_cross_start_stability: shared cross-start fresh cosine over outputs 4-15: mean >= .9 and min >= .8.
pred_e_bank_identified: >= 48 of the 96 canonical correlations between the seeds' shared-bank spans on the fresh panel are >= .9.
Price: 432 forwards (bar 460); 0 model backwards; 2 fits x 400 Adam updates of 443,904 parameters on 194,560 states.
"""
import os, sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; P = ROOT / 'basis_aligned/polynomial_causal/direct_tensor_match'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bidegree_correction_v1 import SEEDS, STEPS, LR, EVAL_EVERY, atoms, profile, errors, controls, split_states
from run_bidegree_data_scaling_v1 import span_cca, cca_controls, TOKENS_CAL, TOKENS_PREV
FORWARDS_MAX = 460; VAL_DOCS = 64; SIZE = 3040; BANK = 96; FIT_EVAL_EVERY = 50
BARS = dict(replay=1e-4, export=1e-8, not_worse=1.05, better=.95, cos_mean=.9, cos_min=.8, cca=.9, cca_count=48)
INPUTS = ['SHARED_BANK_CORRECTION_PLAN_V1.md', 'BIDEGREE_DATA_SCALING_V1.json', 'FIT_CORPUS_TOKENS_V1.pt', 'SENSITIVE_ROOT_CALIBRATION_V2.pt', 'NATIVE_QUARTIC_COVARIANCE_V1.pt',
          'QUARTIC_ADDITIONAL_STATES_V1.pt', 'MIXED_CP_FEATURES_SEED1001_V1.pt', 'EXPANDED_ROOT_EMPIRICAL_V1.pt', 'RESIDUAL_FRESH_STATES_V1.pt', 'RESIDUAL_FRESH_TOKENS_V1.pt',
          'ALL_FEATURE_MATCHED_RESPONSES_V1.json']


def shared_objective(reads, src, X, r, w):
    """reads [1, BANK, 4, D]; atoms shared by all outputs; coefficients profiled per output."""
    import torch
    z = torch.zeros_like(X); A = atoms(reads, src, z, z, X)[:, 0]                          # [N, BANK]
    Ae = A[:, None, :].expand(A.shape[0], r.shape[1], A.shape[1]); c = profile(Ae, r, w)     # [G, BANK]
    pred = A @ c.T; num = (w * (pred - r).square()).sum(0); den = (w * r.square()).sum(0)
    return (num / den).mean(), c, A


def shared_controls():
    import torch
    torch.manual_seed(11); dt = torch.float64; N, D, K, G = 50, 12, 5, 3
    X = torch.randn(N, D, dtype=dt); reads = torch.randn(1, K, 4, D, dtype=dt); src = torch.full((1, K, 4), 2, dtype=torch.long)
    z = torch.zeros_like(X); A = atoms(reads, src, z, z, X)[:, 0]
    Ax = torch.stack([X @ (reads[0, :, i] / reads[0, :, i].norm(dim=-1, keepdim=True)).T for i in range(4)]).prod(0)
    e1 = float((A - Ax).abs().max())
    cstar = torch.randn(G, K, dtype=dt); r = A @ cstar.T; w = torch.ones_like(r); loss, c, _ = shared_objective(reads, src, X, r, w)
    e2 = float((c - cstar).abs().max()) + float(loss)
    assert e1 < 1e-12 and e2 < 1e-6, (e1, e2)   # the 1e-8 relative ridge in profile() perturbs the planted recovery at ~1e-8
    return dict(shared_atoms_equal_products=e1, shared_profile_recovers_planted=e2)


def main():
    import torch
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    plan = dict(candidate_id='shared_bank_correction.v1', forwards_max=FORWARDS_MAX, model_backwards=0, model_updates=0, fit_parameters=2 * (BANK * 4 * 1152 + 16 * BANK),
                fits=2, steps=STEPS, gpu_accessed=False, model_loaded=False, execution_policy='managed_queue_only', bars=BARS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        plan['controls'] = dict(**controls(), **cca_controls(), **shared_controls()); print(json.dumps(plan, indent=2, sort_keys=True)); return
    sys.path.insert(0, str(P))
    from circuit_fast_screen_producer import Bilin18TorchBackend
    from run_source_geometry_census_v1 import quartic_pieces
    torch.backends.cuda.matmul.allow_tf32 = False; start = time.monotonic(); dev = 'cuda'
    out = P / 'SHARED_BANK_CORRECTION_V1.json'; assert not out.exists()
    manifest = {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in INPUTS}
    manifest['tokens_calibration'] = hashlib.sha256(TOKENS_CAL.read_bytes()).hexdigest(); manifest['tokens_prev'] = [hashlib.sha256(t.read_bytes()).hexdigest() for t in TOKENS_PREV]
    checks = dict(**controls(), **cca_controls(), **shared_controls())
    model = Bilin18TorchBackend.load('cuda').model.float(); forwards = 0
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
    cal_parent = parent(calx); fresh_parent = parent(fx)
    state = model.state_dict(); w = lambda l, n: state[f'transformer.h.{l}.mlp.{n}.weight'].double()
    w16 = (w(16, 'Left'), w(16, 'Right'), w(16, 'Down')); w17 = (w(17, 'Left'), w(17, 'Right'), w(17, 'Down')); lam17 = float(model.transformer.h[17].lambdas[0])
    writer = torch.load(P / 'EXPANDED_ROOT_EMPIRICAL_V1.pt', weights_only=True)['writer'].to(dev).double(); vocab = state['lm_head.weight'].double()
    uw = vocab @ writer; readers = vocab.T @ uw / uw.square().sum(0); del vocab, uw
    teacher = lambda x: quartic_pieces(x, torch.zeros_like(x), w16, w17, lam17, readers)[0] / scale
    S, C, xn, fw = split_states(model, tokens, dev); forwards += fw; replay_cal = float(((S + C) - calx).norm() / calx.norm()); del S, C
    fS, fC, fxn, fw = split_states(model, ftok, dev); forwards += fw; replay_fresh = float(((fS + fC) - fx).norm() / fx.norm()); del fS, fC
    prevtok = torch.cat([torch.load(t, weights_only=True)[:, :64] for t in TOKENS_PREV]).contiguous(); assert prevtok.shape == (672, 64)
    pS, pC_, pxn, fw = split_states(model, prevtok, dev); forwards += fw; replay_prev = float(((pS + pC_) - pxn).norm() / pxn.norm()); del pS, pC_
    corpus = torch.load(P / 'FIT_CORPUS_TOKENS_V1.pt', weights_only=True)[:, :64].contiguous(); assert corpus.shape == (2432, 64)
    cS, cC, cxn, fw = split_states(model, corpus, dev); forwards += fw; replay_corpus = float(((cS + cC) - cxn).norm() / cxn.norm()); del cS, cC
    teacher_replay = float((teacher(calx) - caly).norm() / caly.norm())
    print(f'[split] calibration {replay_cal:.2e} | fresh {replay_fresh:.2e} | prev {replay_prev:.2e} | corpus {replay_corpus:.2e} | teacher {teacher_replay:.2e} | forwards {forwards}', flush=True)
    doc = torch.arange(672, device=dev).repeat_interleave(64); val_m = doc >= 672 - VAL_DOCS; fit_prev = pxn[~val_m]; val_x = pxn[val_m]
    X = torch.cat([fit_prev, cxn[:(SIZE - 608) * 64]]); R = teacher(X) - parent(X); ones = torch.ones_like(R); r_val = teacher(val_x) - parent(val_x); ones_v = torch.ones_like(r_val)
    assert X.shape[0] == SIZE * 64, X.shape   # SIZE counts fitting documents; the 64 held-out documents are outside it
    prev = json.loads((P / 'BIDEGREE_DATA_SCALING_V1.json').read_text()); local = {int(r['seed']): r['fresh'] for r in prev['rows'] if r['size'] == SIZE}
    src = torch.full((1, BANK, 4), 2, dtype=torch.long, device=dev); zeros = lambda x: torch.zeros_like(x)
    baseline = errors(fresh_parent, fy, don, rec); rows = []; exports = {}; corrections = {}; spans = {}
    cal_weighted = lambda pred: ((calw * (pred - caly).square()).sum(0) / (calw * caly.square()).sum(0)).sqrt().tolist()
    for seed in SEEDS:
        torch.manual_seed(seed); reads = torch.randn(1, BANK, 4, 1152, dtype=torch.float64, device=dev, requires_grad=True)
        opt = torch.optim.Adam([reads], lr=LR); best = None; history = []; t0 = time.monotonic()
        for step in range(STEPS + 1):
            if step % EVAL_EVERY == 0:
                with torch.no_grad():
                    vloss = float(shared_objective(reads, src, val_x, r_val, ones_v)[0]); floss = float(shared_objective(reads, src, X, R, ones)[0]) if step % FIT_EVAL_EVERY == 0 else None
                history.append(dict(step=step, fit=floss, val=vloss))
                if best is None or vloss < best[0]:
                    best = (vloss, step, reads.detach().clone())
            if step == STEPS:
                break
            with torch.enable_grad():
                loss, _, _ = shared_objective(reads, src, X, R, ones)
                opt.zero_grad(); loss.backward(); opt.step()
        reads_b = best[2]
        with torch.no_grad():
            _, c, _ = shared_objective(reads_b, src, X, R, ones)
            Af = atoms(reads_b, src, zeros(fxn), zeros(fxn), fxn)[:, 0]; fpred = fresh_parent + Af @ c.T
            cpred = cal_parent + atoms(reads_b, src, zeros(xn), zeros(xn), xn)[:, 0] @ c.T
        m = errors(fpred, fy, don, rec); mc = dict(value=((cpred - caly).square().sum(0) / caly.square().sum(0)).sqrt().tolist(), sensitivity_weighted=cal_weighted(cpred))
        key = f'shared_seed{seed}'; exports[key] = dict(reads=(reads_b / reads_b.norm(dim=-1, keepdim=True)).cpu(), src=src.cpu(), coefficients=c.cpu(), scale=scale, seed=seed)
        corrections[seed] = (fpred - fresh_parent).cpu(); spans[seed] = Af.cpu()
        rows.append(dict(arm='shared', seed=seed, fit_states=int(X.shape[0]), selected_step=best[1], selected_val_objective=best[0], history=history, fit_seconds=time.monotonic() - t0, fresh=m, calibration=mc,
                         local_reference=local[seed]))
        print(f'[{key}] step {best[1]} val {best[0]:.4f} | fresh small value {m["small_value_rms"]:.4f} (local {local[seed]["small_value_rms"]:.4f}) response {m["small_response_rms"]:.4f} (local {local[seed]["small_response_rms"]:.4f}) | calibration weighted small {(sum(v * v for v in mc["sensitivity_weighted"][4:]) / 12) ** .5:.4f} | outputs 0-3 {[round(v, 4) for v in m["value"][:4]]} | {time.monotonic() - t0:.0f}s', flush=True)
    a, b = corrections[SEEDS[0]], corrections[SEEDS[1]]; cos = ((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-300)).tolist()
    cca = span_cca(spans[SEEDS[0]], spans[SEEDS[1]]); n_ok = sum(1 for x in cca if x >= BARS['cca'])
    print(f'[audit] cosines 4-15 mean {sum(cos[4:]) / 12:.3f} min {min(cos[4:]):.3f} | bank CCs >= .9: {n_ok} of {BANK}; CCs {[round(x, 2) for x in cca[::8]]} ...', flush=True)
    torch.save(exports, P / 'SHARED_BANK_CORRECTION_EXPORTS_V1.pt')
    reloaded = torch.load(P / 'SHARED_BANK_CORRECTION_EXPORTS_V1.pt', weights_only=False); export_err = 0.0
    for r in rows:
        e = reloaded[f'shared_seed{r["seed"]}']
        pred = fresh_parent + atoms(e['reads'].to(dev), e['src'].to(dev), zeros(fxn), zeros(fxn), fxn)[:, 0] @ e['coefficients'].to(dev).T
        m = errors(pred, fy, don, rec); export_err = max(export_err, max(abs(x - y) for x, y in zip(m['value'] + m['response'], r['fresh']['value'] + r['fresh']['response'])))
    by = {r['seed']: r['fresh'] for r in rows}; tcos = cos[4:]
    ratio = {s: dict(value=by[s]['small_value_rms'] / local[s]['small_value_rms'], response=by[s]['small_response_rms'] / local[s]['small_response_rms']) for s in SEEDS}
    preds = dict(pred_a_integrity=max(replay_cal, replay_fresh, replay_prev, replay_corpus, teacher_replay) <= BARS['replay'] and export_err <= BARS['export'],
                 pred_b_shared_not_worse=all(ratio[s]['value'] <= BARS['not_worse'] and ratio[s]['response'] <= BARS['not_worse'] for s in SEEDS),
                 pred_c_shared_better=all(ratio[s]['value'] <= BARS['better'] and ratio[s]['response'] <= BARS['better'] for s in SEEDS),
                 pred_d_cross_start_stability=(sum(tcos) / len(tcos) >= BARS['cos_mean'] and min(tcos) >= BARS['cos_min']),
                 pred_e_bank_identified=n_ok >= BARS['cca_count'])
    assert forwards <= FORWARDS_MAX, forwards
    cost = dict(atoms=BANK, reads_per_atom=4, read_multiply_adds=BANK * 4 * 1152, products=BANK * 3, coefficient_multiplies=16 * BANK, adds=16 * BANK, added_coefficients=BANK * 4 * 1152 + 16 * BANK)
    receipt = dict(predictions=preds, controls=checks, manifest=manifest, replay=dict(calibration=replay_cal, fresh=replay_fresh, prev=replay_prev, corpus=replay_corpus, teacher=teacher_replay, export=export_err),
                   baseline=baseline, rows=rows, shared_over_local=ratio, audit=dict(cross_start_cosine=cos, bank_canonical_correlations=cca, bank_ccs_over_bar=n_ok), scale=scale, literal_cost=cost,
                   forwards=forwards, backwards=0, seconds=time.monotonic() - start, peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
                   scope='Shared bank vs output-local correction at matched coefficients; no feature names, no intervention, no circuit claim.')
    out.write_text(json.dumps(receipt, indent=1)); print(json.dumps(dict(predictions=preds, replay=receipt['replay'], shared_over_local=ratio, forwards=forwards), indent=1), flush=True)


if __name__ == '__main__':
    main()
