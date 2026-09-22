#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_integrity pred_b_more_data_helps pred_c_monotone pred_d_cross_start_stability pred_e_spans_identified
"""Data scaling of the untyped 96-atom correction with a cross-start span audit (BIDEGREE_DATA_SCALING_PLAN_V1.md).
Untyped arm only (typed closed by BIDEGREE_DATA_ABLATION_V1), same loop/seeds/capacity as run_bidegree_correction_v1 (functions imported).
Ladder: 608 documents (previous receipt, not re-run), 1,216 and 3,040 documents (the 608 + the frozen FIT_CORPUS_TOKENS_V1 corpus); the same 64
held-out documents select the snapshot; exact teacher F4; unweighted objective; fresh panel and matched pairs scored as before, never fitted.
pred_a_integrity: replays (S+C vs stored rows on calibration/fresh, vs x on both fitting panels) <= 1e-4; teacher <= 1e-4; exports <= 1e-8; controls.
pred_b_more_data_helps: at 3040 docs both seeds fresh small-output value RMS <= .36 and response RMS <= .39.
pred_c_monotone: both seeds, value RMS at 3040 <= at 1216 <= at 608 (previous receipt).
pred_d_cross_start_stability: at 3040 docs cross-start fresh cosine over outputs 4-15: mean >= .9 and min >= .8.
pred_e_spans_identified: at 3040 docs >= 6 of the 12 small outputs have all six canonical correlations >= .9 between the seeds' atom spans.
Price: 304 + 84 + 12 + 32 forwards (bar 460); 0 model backwards; 4 fits x 400 Adam updates, the largest on 194,560 states.
"""
import os, sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; P = ROOT / 'basis_aligned/polynomial_causal/direct_tensor_match'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bidegree_correction_v1 import ATOMS, SEEDS, STEPS, LR, EVAL_EVERY, allocate, build_sources, atoms, profile, objective, errors, controls, split_states
FORWARDS_MAX = 460; VAL_DOCS = 64; SIZES = (1216, 3040); FIT_EVAL_EVERY = 50
BARS = dict(replay=1e-4, export=1e-8, value=.36, response=.39, cos_mean=.9, cos_min=.8, cca=.9, cca_outputs=6)
INPUTS = ['BIDEGREE_DATA_SCALING_PLAN_V1.md', 'BIDEGREE_DATA_ABLATION_V1.json', 'FIT_CORPUS_TOKENS_V1.pt', 'FIT_CORPUS_ROWS_V1.json', 'SENSITIVE_ROOT_CALIBRATION_V2.pt',
          'NATIVE_QUARTIC_COVARIANCE_V1.pt', 'QUARTIC_ADDITIONAL_STATES_V1.pt', 'MIXED_CP_FEATURES_SEED1001_V1.pt', 'EXPANDED_ROOT_EMPIRICAL_V1.pt', 'RESIDUAL_FRESH_STATES_V1.pt',
          'RESIDUAL_FRESH_TOKENS_V1.pt', 'ALL_FEATURE_MATCHED_RESPONSES_V1.json']
TOKENS_CAL = ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n96_skip1200.pt'
TOKENS_PREV = [ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n480_skip80.pt', ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt']


def span_cca(A1, A2):
    """Canonical correlations between the column spans of A1, A2 [N, K] under the plain inner product."""
    import torch
    Q1, _ = torch.linalg.qr(A1 / A1.norm(dim=0, keepdim=True)); Q2, _ = torch.linalg.qr(A2 / A2.norm(dim=0, keepdim=True))
    return torch.linalg.svdvals(Q1.T @ Q2).clamp(max=1.0).tolist()


def cca_controls():
    import torch
    torch.manual_seed(5); A = torch.randn(200, 6, dtype=torch.float64); M = torch.randn(6, 6, dtype=torch.float64)
    same = span_cca(A, A @ M); other = span_cca(A, torch.randn(200, 6, dtype=torch.float64))
    half = span_cca(A, torch.cat([A[:, :3] @ torch.randn(3, 3, dtype=torch.float64), torch.randn(200, 3, dtype=torch.float64)], 1))
    e1 = max(abs(1 - c) for c in same); e2 = max(abs(1 - c) for c in half[:3]); ok = all(c < .5 for c in other)
    assert e1 < 1e-10 and e2 < 1e-10 and ok, (same, half, other)
    return dict(same_span=e1, shared_half_span=e2, unrelated_spans_max=max(other))


def main():
    import torch
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    plan = dict(candidate_id='bidegree_data_scaling.v1', forwards_max=FORWARDS_MAX, model_backwards=0, model_updates=0, fit_parameters=4 * (96 * 4 * 1152 + 96),
                fits=4, steps=STEPS, sizes=SIZES, gpu_accessed=False, model_loaded=False, execution_policy='managed_queue_only', bars=BARS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        plan['controls'] = dict(**controls(), **cca_controls()); print(json.dumps(plan, indent=2, sort_keys=True)); return
    sys.path.insert(0, str(P))
    from circuit_fast_screen_producer import Bilin18TorchBackend
    from run_source_geometry_census_v1 import quartic_pieces, parent_pieces
    torch.backends.cuda.matmul.allow_tf32 = False; start = time.monotonic(); dev = 'cuda'
    out = P / 'BIDEGREE_DATA_SCALING_V1.json'; assert not out.exists()
    manifest = {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in INPUTS}
    manifest['tokens_calibration'] = hashlib.sha256(TOKENS_CAL.read_bytes()).hexdigest(); manifest['tokens_prev'] = [hashlib.sha256(t.read_bytes()).hexdigest() for t in TOKENS_PREV]
    checks = dict(**controls(), **cca_controls())
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
    cal_parent = parent(calx); fresh_parent = parent(fx)
    state = model.state_dict(); w = lambda l, n: state[f'transformer.h.{l}.mlp.{n}.weight'].double()
    w16 = (w(16, 'Left'), w(16, 'Right'), w(16, 'Down')); w17 = (w(17, 'Left'), w(17, 'Right'), w(17, 'Down')); lam17 = float(model.transformer.h[17].lambdas[0])
    writer = torch.load(P / 'EXPANDED_ROOT_EMPIRICAL_V1.pt', weights_only=True)['writer'].to(dev).double(); vocab = state['lm_head.weight'].double()
    uw = vocab @ writer; readers = vocab.T @ uw / uw.square().sum(0); del vocab, uw
    teacher = lambda x: quartic_pieces(x, torch.zeros_like(x), w16, w17, lam17, readers)[0] / scale
    # ---- splits ------------------------------------------------------------------------------------------------------------------
    S, C, xn, fw = split_states(model, tokens, dev); forwards += fw; replay_cal = float(((S + C) - calx).norm() / calx.norm())
    fS, fC, fxn, fw = split_states(model, ftok, dev); forwards += fw; replay_fresh = float(((fS + fC) - fx).norm() / fx.norm())
    prevtok = torch.cat([torch.load(t, weights_only=True)[:, :64] for t in TOKENS_PREV]).contiguous(); assert prevtok.shape == (672, 64)
    pS, pC_, pxn, fw = split_states(model, prevtok, dev); forwards += fw; replay_prev = float(((pS + pC_) - pxn).norm() / pxn.norm()); del pS, pC_
    corpus = torch.load(P / 'FIT_CORPUS_TOKENS_V1.pt', weights_only=True)[:, :64].contiguous(); assert corpus.shape == (2432, 64)
    cS, cC, cxn, fw = split_states(model, corpus, dev); forwards += fw; replay_corpus = float(((cS + cC) - cxn).norm() / cxn.norm()); del cS, cC
    teacher_replay = float((teacher(calx) - caly).norm() / caly.norm())
    print(f'[split] calibration {replay_cal:.2e} | fresh {replay_fresh:.2e} | prev {replay_prev:.2e} | corpus {replay_corpus:.2e} | teacher {teacher_replay:.2e} | forwards {forwards}', flush=True)
    # residuals on the fitting panels (x only: the untyped arm reads x)
    doc = torch.arange(672, device=dev).repeat_interleave(64); val_m = doc >= 672 - VAL_DOCS; fit_prev = pxn[~val_m]; val_x = pxn[val_m]
    r_prev = teacher(fit_prev) - parent(fit_prev); r_val = teacher(val_x) - parent(val_x); r_corpus = teacher(cxn) - parent(cxn)
    zeros = lambda x: torch.zeros_like(x)
    prev = json.loads((P / 'BIDEGREE_DATA_ABLATION_V1.json').read_text()); alloc = prev['allocation']
    prev608 = {int(r['seed']): r['fresh'] for r in prev['rows'] if r['arm'] == 'untyped'}
    src = build_sources(alloc, typed=False).to(dev)
    baseline = errors(fresh_parent, fy, don, rec); rows = []; exports = {}; corrections = {}; spans = {}
    cal_weighted = lambda pred: ((calw * (pred - caly).square()).sum(0) / (calw * caly.square()).sum(0)).sqrt().tolist()
    for size in SIZES:
        extra_docs = size - 608; X = torch.cat([fit_prev, cxn[:extra_docs * 64]]); R = torch.cat([r_prev, r_corpus[:extra_docs * 64]]); ones = torch.ones_like(R)
        fitp = (zeros(X), zeros(X), X, R, ones); valp = (zeros(val_x), zeros(val_x), val_x, r_val, torch.ones_like(r_val))
        for seed in SEEDS:
            torch.manual_seed(seed); reads = torch.randn(16, ATOMS, 4, 1152, dtype=torch.float64, device=dev, requires_grad=True)
            opt = torch.optim.Adam([reads], lr=LR); best = None; history = []; t0 = time.monotonic()
            for step in range(STEPS + 1):
                if step % EVAL_EVERY == 0:
                    with torch.no_grad():
                        vloss, _ = objective(reads, src, *valp); floss = float(objective(reads, src, *fitp)[0]) if step % FIT_EVAL_EVERY == 0 else None
                    history.append(dict(step=step, fit=floss, val=float(vloss)))
                    if best is None or float(vloss) < best[0]:
                        best = (float(vloss), step, reads.detach().clone())
                if step == STEPS:
                    break
                with torch.enable_grad():
                    loss, _ = objective(reads, src, *fitp)
                    opt.zero_grad(); loss.backward(); opt.step()
            reads_b = best[2]
            with torch.no_grad():
                A = atoms(reads_b, src, *fitp[:3]); c = profile(A, fitp[3], fitp[4])
                Af = atoms(reads_b, src, zeros(fxn), zeros(fxn), fxn); fpred = fresh_parent + (Af * c[None]).sum(-1)
                cpred = cal_parent + (atoms(reads_b, src, zeros(xn), zeros(xn), xn) * c[None]).sum(-1)
            m = errors(fpred, fy, don, rec); mc = dict(value=((cpred - caly).square().sum(0) / caly.square().sum(0)).sqrt().tolist(), sensitivity_weighted=cal_weighted(cpred))
            key = f'size{size}_seed{seed}'; exports[key] = dict(reads=(reads_b / reads_b.norm(dim=-1, keepdim=True)).cpu(), src=src.cpu(), coefficients=c.cpu(), scale=scale, allocation=alloc, size=size, seed=seed)
            corrections[key] = (fpred - fresh_parent).cpu(); spans[key] = Af.cpu()
            rows.append(dict(size=size, seed=seed, fit_states=int(X.shape[0]), selected_step=best[1], selected_val_objective=best[0], history=history, fit_seconds=time.monotonic() - t0, fresh=m, calibration=mc))
            print(f'[{key}] step {best[1]} val {best[0]:.4f} | fresh small value {m["small_value_rms"]:.4f} response {m["small_response_rms"]:.4f} | calibration weighted small {(sum(v * v for v in mc["sensitivity_weighted"][4:]) / 12) ** .5:.4f} | outputs 0-3 {[round(v, 4) for v in m["value"][:4]]} | {time.monotonic() - t0:.0f}s', flush=True)
    audit = {}
    for size in SIZES:
        a, b = corrections[f'size{size}_seed{SEEDS[0]}'], corrections[f'size{size}_seed{SEEDS[1]}']
        cos = ((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-300)).tolist()
        A1, A2 = spans[f'size{size}_seed{SEEDS[0]}'], spans[f'size{size}_seed{SEEDS[1]}']
        cca = [span_cca(A1[:, g], A2[:, g]) for g in range(16)]
        audit[str(size)] = dict(cross_start_cosine=cos, canonical_correlations=cca, outputs_all_cc_over_bar=[g for g in range(16) if min(cca[g]) >= BARS['cca']])
        print(f'[audit size {size}] cosines 4-15 mean {sum(cos[4:]) / 12:.3f} min {min(cos[4:]):.3f} | min CC per output {[round(min(x), 3) for x in cca]}', flush=True)
    torch.save(exports, P / 'BIDEGREE_DATA_SCALING_EXPORTS_V1.pt')
    reloaded = torch.load(P / 'BIDEGREE_DATA_SCALING_EXPORTS_V1.pt', weights_only=False); export_err = 0.0
    for r in rows:
        e = reloaded[f'size{r["size"]}_seed{r["seed"]}']
        pred = fresh_parent + (atoms(e['reads'].to(dev), e['src'].to(dev), zeros(fxn), zeros(fxn), fxn) * e['coefficients'].to(dev)[None]).sum(-1)
        m = errors(pred, fy, don, rec); export_err = max(export_err, max(abs(x - y) for x, y in zip(m['value'] + m['response'], r['fresh']['value'] + r['fresh']['response'])))
    by = {(r['size'], r['seed']): r['fresh'] for r in rows}; big = SIZES[-1]; tcos = audit[str(big)]['cross_start_cosine'][4:]
    small_ok = [g for g in audit[str(big)]['outputs_all_cc_over_bar'] if g >= 4]
    preds = dict(pred_a_integrity=max(replay_cal, replay_fresh, replay_prev, replay_corpus, teacher_replay) <= BARS['replay'] and export_err <= BARS['export'],
                 pred_b_more_data_helps=all(by[big, s]['small_value_rms'] <= BARS['value'] and by[big, s]['small_response_rms'] <= BARS['response'] for s in SEEDS),
                 pred_c_monotone=all(by[3040, s]['small_value_rms'] <= by[1216, s]['small_value_rms'] <= prev608[s]['small_value_rms'] for s in SEEDS),
                 pred_d_cross_start_stability=(sum(tcos) / len(tcos) >= BARS['cos_mean'] and min(tcos) >= BARS['cos_min']),
                 pred_e_spans_identified=len(small_ok) >= BARS['cca_outputs'])
    assert forwards <= FORWARDS_MAX, forwards
    cost = dict(atoms=96, reads_per_atom=4, read_multiply_adds=96 * 4 * 1152, products=96 * 3, coefficient_multiplies=96, adds=96, added_coefficients=96 * 4 * 1152 + 96)
    receipt = dict(predictions=preds, controls=checks, manifest=manifest, replay=dict(calibration=replay_cal, fresh=replay_fresh, prev=replay_prev, corpus=replay_corpus, teacher=teacher_replay, export=export_err),
                   allocation=alloc, previous_608=prev608, baseline=baseline, rows=rows, audit=audit, scale=scale, literal_cost=cost, forwards=forwards, backwards=0,
                   seconds=time.monotonic() - start, peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
                   scope='Data scaling of the untyped correction with a cross-start span audit; no feature names, no intervention, no circuit claim.')
    out.write_text(json.dumps(receipt, indent=1)); print(json.dumps(dict(predictions=preds, replay=receipt['replay'], forwards=forwards), indent=1), flush=True)


if __name__ == '__main__':
    main()
