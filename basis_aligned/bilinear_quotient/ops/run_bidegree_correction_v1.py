#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_integrity pred_b_typed_beats_untyped pred_c_beats_hybrid pred_d_large_outputs_kept pred_e_cross_start_stability
"""Bidegree-typed correction of the CP512 parent at the hybrid experiment's capacity (BIDEGREE_CORRECTION_PLAN_V1.md).
Per output 6 atoms (96 total, 442,464 coefficients). TYPED arm: an atom of bidegree (p, q) is a product of p unit reads of the propagated
source state S and q unit reads of the context transport C (S + C = the normalized MLP16 input); the (p, q) counts per output are fixed
before fitting by largest-remainder rounding of 6 x the corrected census residual piece shares. UNTYPED control: four reads of x, same
loop, seeds, objective and budget. Coefficients profiled in closed form; reads trained by Adam on the sensitivity-weighted relative
squared error over all 16 outputs on 80 calibration documents, snapshot at the best of the 16 held-out documents; scored on the fresh
16,384-state panel and the 2,494 matched pairs exactly as the hybrid receipt.
pred_a_integrity: S+C replays the stored rows on both panels <= 1e-4; exports reloaded reproduce every fresh error <= 1e-8; controls pass.
pred_b_typed_beats_untyped: both seeds, typed fresh small-output value RMS and response RMS <= .85 x untyped.
pred_c_beats_hybrid: both seeds, typed small-output value RMS <= .40 and response RMS <= .42.
pred_d_large_outputs_kept: both seeds, typed fresh value error on outputs 0-3 <= parent's; output 3 <= .9 x parent's.
pred_e_cross_start_stability: typed cross-start fresh cosine over outputs 4-15: mean >= .9 and min >= .8.
Price: 12 + 32 forwards (bar 64); 0 model backwards; 4 fits x 400 Adam updates of 442,464 parameters.
"""
import os, sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; P = ROOT / 'basis_aligned/polynomial_causal/direct_tensor_match'
sys.path.insert(0, str(Path(__file__).resolve().parent))
FORWARDS_MAX = 64
ATOMS, SEEDS, STEPS, LR, RIDGE, VAL_DOCS, EVAL_EVERY = 6, (25001, 25002), 400, 0.05, 1e-8, 16, 10
BARS = dict(replay=1e-4, export=1e-8, typed_vs_untyped=.85, value=.40, response=.42, output3=.9, cos_mean=.9, cos_min=.8)
PIECES = ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4))
INPUTS = ['BIDEGREE_CORRECTION_PLAN_V1.md', 'SOURCE_GEOMETRY_CENSUS_PLAN_V1.md', 'SENSITIVE_ROOT_CALIBRATION_V2.pt', 'NATIVE_QUARTIC_COVARIANCE_V1.pt',
          'QUARTIC_ADDITIONAL_STATES_V1.pt', 'MIXED_CP_FEATURES_SEED1001_V1.pt', 'EXPANDED_ROOT_EMPIRICAL_V1.pt', 'RESIDUAL_FRESH_STATES_V1.pt',
          'RESIDUAL_FRESH_TOKENS_V1.pt', 'ALL_FEATURE_MATCHED_RESPONSES_V1.json']
TOKENS_CAL = ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n96_skip1200.pt'


def allocate(shares, n=ATOMS):
    """Largest-remainder rounding of n x shares (shares sum to 1) into non-negative integer counts summing to n."""
    raw = [n * s for s in shares]; base = [int(r) for r in raw]; rem = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def build_sources(alloc_per_output, typed):
    """src[g, k, i] in {0: S, 1: C, 2: x} for output g, atom k, read i."""
    import torch
    G = len(alloc_per_output); src = torch.full((G, ATOMS, 4), 2, dtype=torch.long)
    if typed:
        for g, alloc in enumerate(alloc_per_output):
            k = 0
            for (p, q), cnt in zip(PIECES, alloc):
                for _ in range(cnt):
                    src[g, k, :p] = 0; src[g, k, p:] = 1; k += 1
            assert k == ATOMS
    return src


def atoms(reads, src, S, C, X):
    """reads [G, K, 4, D] (any scale; normalized inside), src [G, K, 4]; returns atom values [N, G, K]."""
    import torch
    R = reads / reads.norm(dim=-1, keepdim=True)
    Z = torch.stack([torch.einsum('nd,gkid->ngki', M, R) for M in (S, C, X)])          # [3, N, G, K, 4]
    index = src[None, None].expand(1, Z.shape[1], *src.shape)                            # pick the source of every read
    return torch.gather(Z, 0, index)[0].prod(-1)


def profile(A, r, w, ridge=RIDGE):
    """Weighted least-squares coefficients per output: A [N, G, K], r [N, G], w [N, G] -> c [G, K]."""
    import torch
    G = torch.einsum('ngk,ng,ngl->gkl', A, w, A); b = torch.einsum('ngk,ng,ng->gk', A, w, r)
    G = G + ridge * torch.diagonal(G, dim1=1, dim2=2).mean(1)[:, None, None].clamp_min(1e-300) * torch.eye(A.shape[2], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(G, b)


def objective(reads, src, S, C, X, r, w):
    A = atoms(reads, src, S, C, X); c = profile(A, r, w); pred = (A * c[None]).sum(-1)
    num = (w * (pred - r).square()).sum(0); den = (w * r.square()).sum(0)
    return (num / den).mean(), c


def errors(pred, y, don, rec):
    """Per-output value error and matched-pair response error (all 16 outputs), plus small-output RMS of each."""
    v = ((pred - y).square().sum(0) / y.square().sum(0)).sqrt()
    dref = y[don] - y[rec]; dp = pred[don] - pred[rec]; r = ((dp - dref).square().sum(0) / dref.square().sum(0)).sqrt()
    return dict(value=v.tolist(), response=r.tolist(), small_value_rms=float(v[4:].square().mean().sqrt()), small_response_rms=float(r[4:].square().mean().sqrt()))


def controls():
    import torch
    torch.manual_seed(3); dt = torch.float64; N, D, G = 40, 12, 3
    S, C = torch.randn(N, D, dtype=dt), torch.randn(N, D, dtype=dt); X = S + C
    alloc = [allocate([.2, .3, .1, .25, .15]), allocate([1, 0, 0, 0, 0]), allocate([.05, .05, .05, .05, .8])]
    assert all(sum(a) == ATOMS and min(a) >= 0 for a in alloc), alloc
    src = build_sources(alloc, typed=True); reads = torch.randn(G, ATOMS, 4, D, dtype=dt)
    A = atoms(reads, src, S, C, X); A_noC = atoms(reads, src, S, torch.zeros_like(C), S); A_noS = atoms(reads, src, torch.zeros_like(S), C, C)
    e1 = 0.0
    for g in range(G):
        k = 0
        for (p, q), cnt in zip(PIECES, alloc[g]):
            for _ in range(cnt):
                e1 += float(A_noC[:, g, k].abs().sum()) if q > 0 else float((A_noC[:, g, k] - A[:, g, k]).abs().sum())
                e1 += float(A_noS[:, g, k].abs().sum()) if p > 0 else float((A_noS[:, g, k] - A[:, g, k]).abs().sum())
                k += 1
    untyped = build_sources(alloc, typed=False); Au = atoms(reads, untyped, S, C, X); Ax = torch.stack([torch.einsum('nd,gkd->ngk', X, reads[:, :, i] / reads[:, :, i].norm(dim=-1, keepdim=True)) for i in range(4)]).prod(0)
    e2 = float((Au - Ax).abs().max())
    cstar = torch.randn(G, ATOMS, dtype=dt); r = (A * cstar[None]).sum(-1); w = torch.rand(N, G, dtype=dt) + .5
    e3 = float((profile(A, r, w, ridge=0.0) - cstar).abs().max())
    y = torch.randn(N, G, dtype=dt); don = torch.tensor([0, 1, 2]); rec = torch.tensor([3, 4, 5]); m = errors(y, y, don, rec)
    e4 = max(m['value']) + max(m['response'])
    assert e1 < 1e-12 and e2 < 1e-12 and e3 < 1e-8 and e4 < 1e-12, (e1, e2, e3, e4)
    return dict(typed_atoms_vanish_correctly=e1, untyped_equals_product_of_x_reads=e2, profiled_coefficients_recover_planted=e3, error_of_identity=e4, allocations=alloc)


def split_states(model, tokens, dev, batch=8):
    """S, C, xn (float64 on dev, [N, 1152]) at the normalized MLP16 input; S + C = xn up to float32 capture error."""
    import torch, torch.nn.functional as F
    blocks = model.transformer.h; n, T = tokens.shape
    lam = torch.stack([b.lambdas.detach().double().cpu() for b in blocks]); cE = 1.0
    for l in range(17):
        cE = float(lam[l, 0]) * cE + float(lam[l, 1])
    coef = {f'attn{l}': float(torch.prod(lam[l + 1:17, 0])) for l in range(16)}; coef.update({f'mlp{l}': coef[f'attn{l}'] for l in range(16)})
    STATE = torch.zeros(n, T, 1152, dtype=torch.float64); CONTEXT = torch.zeros_like(STATE); XMID = torch.zeros_like(STATE); box = {'sel': None}
    def hook(name, dest):
        def save(module, args, output):
            y = output[0] if isinstance(output, (tuple, list)) else output
            dest[box['sel']] += coef[name] * y.double().cpu()
        return save
    handles = [blocks[l].attn.register_forward_hook(hook(f'attn{l}', CONTEXT)) for l in range(16)] + [blocks[l].mlp.register_forward_hook(hook(f'mlp{l}', STATE)) for l in range(16)]
    fw = 0
    try:
        for a in range(0, n, batch):
            box['sel'] = slice(a, a + batch); ids = tokens[a:a + batch].to(dev)
            x = F.rms_norm(model.transformer.wte(ids), (1152,)); x0 = x; v1 = None; STATE[box['sel']] += cE * x0.double().cpu()
            for l in range(16):
                x, v1 = blocks[l](x, v1, x0)
            b16 = blocks[16]; xin = b16.lambdas[0] * x + b16.lambdas[1] * x0
            a16 = b16.attn(F.rms_norm(xin, (1152,)), v1); a16 = a16[0] if isinstance(a16, (tuple, list)) else a16
            CONTEXT[box['sel']] += a16.double().cpu(); XMID[box['sel']] = (xin + a16).double().cpu(); fw += 1
    finally:
        for h in handles:
            h.remove()
    XMID = XMID.flatten(0, 1); S = STATE.flatten(0, 1); C = CONTEXT.flatten(0, 1); rms = XMID.square().mean(-1, keepdim=True).sqrt()
    return (S / rms).to(dev), (C / rms).to(dev), (XMID / rms).to(dev), fw


def main():
    import torch
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    plan = dict(candidate_id='bidegree_correction.v1', forwards_max=FORWARDS_MAX, model_backwards=0, model_updates=0, fit_parameters=4 * (96 * 4 * 1152 + 96),
                fits=4, steps=STEPS, gpu_accessed=False, model_loaded=False, execution_policy='managed_queue_only', bars=BARS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        plan['controls'] = controls(); print(json.dumps(plan, indent=2, sort_keys=True)); return
    sys.path.insert(0, str(P))
    from circuit_fast_screen_producer import Bilin18TorchBackend
    from run_source_geometry_census_v1 import quartic_pieces, parent_pieces
    torch.backends.cuda.matmul.allow_tf32 = False; start = time.monotonic(); dev = 'cuda'
    out = P / 'BIDEGREE_CORRECTION_V1.json'; assert not out.exists()
    manifest = {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in INPUTS}; manifest['tokens_calibration'] = hashlib.sha256(TOKENS_CAL.read_bytes()).hexdigest()
    checks = controls()
    model = Bilin18TorchBackend.load('cuda').model.float(); forwards = 0
    # ---- panels ------------------------------------------------------------------------------------------------------------------
    tokens = torch.load(TOKENS_CAL, weights_only=True)[:96, :64].contiguous()
    labels = torch.load(P / 'SENSITIVE_ROOT_CALIBRATION_V2.pt', weights_only=True); cov = torch.load(P / 'NATIVE_QUARTIC_COVARIANCE_V1.pt', weights_only=True)
    extra = torch.load(P / 'QUARTIC_ADDITIONAL_STATES_V1.pt', weights_only=True)
    digest = lambda z: hashlib.sha256(z.contiguous().numpy().tobytes()).hexdigest(); assert digest(tokens) == labels['token_sha256']['calibration']
    calx = torch.cat([cov['panels'][0]['rows'], extra['rows']]).to(dev).double(); caly = labels['panels'][0]['target'].to(dev).double()
    calw = labels['panels'][0]['weight'].to(dev).double(); calw = calw / calw.mean(0, keepdim=True)
    fresh = torch.load(P / 'RESIDUAL_FRESH_STATES_V1.pt', weights_only=True); fx = fresh['rows'].to(dev).double(); fy = fresh['target'].to(dev).double()
    ftok = torch.load(P / 'RESIDUAL_FRESH_TOKENS_V1.pt', weights_only=True)[:, :64].contiguous(); assert ftok.shape[0] == 256
    pairs = torch.tensor(json.loads((P / 'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs'], device=dev).T; don, rec = pairs[0], pairs[1]
    scale = float(caly[:, :4].abs().mean()); caly = caly / scale; fy = fy / scale                     # one common scale; every metric is relative
    p = torch.load(P / 'MIXED_CP_FEATURES_SEED1001_V1.pt', weights_only=True); pf = [f.to(dev).double() for f in p['factors']]; pC = p['coefficients'].to(dev).double() / scale
    parent = lambda x: torch.stack([x @ f.T for f in pf]).prod(0) @ pC.T
    cal_parent = parent(calx); fresh_parent = parent(fx); cal_r = caly - cal_parent; fresh_r = fy - fresh_parent
    # ---- source/context split on both panels ------------------------------------------------------------------------------------
    S, C, xn, fw = split_states(model, tokens, dev); forwards += fw; replay_cal = float(((S + C) - calx).norm() / calx.norm())
    fS, fC, fxn, fw = split_states(model, ftok, dev); forwards += fw; replay_fresh = float(((fS + fC) - fx).norm() / fx.norm())
    print(f'[split] calibration S+C vs stored rows {replay_cal:.2e} | fresh {replay_fresh:.2e} | forwards {forwards}', flush=True)
    # ---- allocation from the corrected census pieces (calibration panel) --------------------------------------------------------
    state = model.state_dict(); w = lambda l, n: state[f'transformer.h.{l}.mlp.{n}.weight'].double()
    w16 = (w(16, 'Left'), w(16, 'Right'), w(16, 'Down')); w17 = (w(17, 'Left'), w(17, 'Right'), w(17, 'Down')); lam17 = float(model.transformer.h[17].lambdas[0])
    writer = torch.load(P / 'EXPANDED_ROOT_EMPIRICAL_V1.pt', weights_only=True)['writer'].to(dev).double(); vocab = state['lm_head.weight'].double()
    uw = vocab @ writer; readers = vocab.T @ uw / uw.square().sum(0); del vocab, uw
    tp = quartic_pieces(S, C, w16, w17, lam17, readers) / scale; pp = parent_pieces(S, C, pf, pC); rp = tp - pp
    piece_identity = float((tp.sum(0) - caly).norm() / caly.norm()); parent_identity = float((pp.sum(0) - parent(S + C)).norm() / cal_parent.norm())
    share = (torch.diagonal(torch.einsum('ing,jng->gij', rp, rp), dim1=1, dim2=2)); share = (share / share.sum(1, keepdim=True)).tolist()
    alloc = [allocate(s) for s in share]; del tp, pp, rp
    print(f'[census] target piece identity {piece_identity:.2e} | parent identity {parent_identity:.2e} | allocation per output {alloc}', flush=True)
    # ---- fits ---------------------------------------------------------------------------------------------------------------------
    doc = torch.arange(96, device=dev).repeat_interleave(64); fit_m = doc < 96 - VAL_DOCS; val_m = ~fit_m
    panels = dict(fit=(S[fit_m], C[fit_m], xn[fit_m], cal_r[fit_m], calw[fit_m]), val=(S[val_m], C[val_m], xn[val_m], cal_r[val_m], calw[val_m]))
    baseline = errors(fresh_parent, fy, don, rec); rows = []; exports = {}
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
                A = atoms(reads_b, src, *panels['fit'][:3]); c = profile(A, panels['fit'][3], panels['fit'][4])          # coefficients profiled on the fit split only
                fpred = fresh_parent + (atoms(reads_b, src, fS, fC, fxn) * c[None]).sum(-1); cpred = cal_parent + (atoms(reads_b, src, S, C, xn) * c[None]).sum(-1)
            m = errors(fpred, fy, don, rec); mc = dict(value=((cpred - caly).square().sum(0) / caly.square().sum(0)).sqrt().tolist())
            key = f'{arm}_seed{seed}'; exports[key] = dict(reads=(reads_b / reads_b.norm(dim=-1, keepdim=True)).cpu(), src=src.cpu(), coefficients=c.cpu(), scale=scale, allocation=alloc, arm=arm, seed=seed)
            rows.append(dict(arm=arm, seed=seed, selected_step=best[1], selected_val_objective=best[0], history=history, fit_seconds=time.monotonic() - t0, fresh=m, calibration_value=mc['value'],
                             fresh_correction=(fpred - fresh_parent).cpu()))
            print(f'[{key}] step {best[1]} val {best[0]:.4f} | fresh small value {m["small_value_rms"]:.4f} response {m["small_response_rms"]:.4f} | outputs 0-3 value {[round(v, 4) for v in m["value"][:4]]} (parent {[round(v, 4) for v in baseline["value"][:4]]}) | {time.monotonic() - t0:.0f}s', flush=True)
    # ---- cross-start cosines, export replay ---------------------------------------------------------------------------------------
    def cosines(arm):
        a, b = [r['fresh_correction'] for r in rows if r['arm'] == arm]
        return ((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-300)).tolist()
    cos = {arm: cosines(arm) for arm in ('typed', 'untyped')}
    torch.save(exports, P / 'BIDEGREE_CORRECTION_EXPORTS_V1.pt')
    reloaded = torch.load(P / 'BIDEGREE_CORRECTION_EXPORTS_V1.pt', weights_only=False); export_err = 0.0
    for r in rows:
        e = reloaded[f'{r["arm"]}_seed{r["seed"]}']
        pred = fresh_parent + (atoms(e['reads'].to(dev), e['src'].to(dev), fS, fC, fxn) * e['coefficients'].to(dev)[None]).sum(-1)
        m = errors(pred, fy, don, rec); export_err = max(export_err, max(abs(x - y) for x, y in zip(m['value'] + m['response'], r['fresh']['value'] + r['fresh']['response'])))
        del r['fresh_correction']
    by = {(r['arm'], r['seed']): r['fresh'] for r in rows}
    ratio = {s: dict(value=by['typed', s]['small_value_rms'] / by['untyped', s]['small_value_rms'], response=by['typed', s]['small_response_rms'] / by['untyped', s]['small_response_rms']) for s in SEEDS}
    tcos = cos['typed'][4:]
    preds = dict(pred_a_integrity=replay_cal <= BARS['replay'] and replay_fresh <= BARS['replay'] and export_err <= BARS['export'],
                 pred_b_typed_beats_untyped=all(ratio[s]['value'] <= BARS['typed_vs_untyped'] and ratio[s]['response'] <= BARS['typed_vs_untyped'] for s in SEEDS),
                 pred_c_beats_hybrid=all(by['typed', s]['small_value_rms'] <= BARS['value'] and by['typed', s]['small_response_rms'] <= BARS['response'] for s in SEEDS),
                 pred_d_large_outputs_kept=all(all(by['typed', s]['value'][o] <= baseline['value'][o] for o in range(4)) and by['typed', s]['value'][3] <= BARS['output3'] * baseline['value'][3] for s in SEEDS),
                 pred_e_cross_start_stability=(sum(tcos) / len(tcos) >= BARS['cos_mean'] and min(tcos) >= BARS['cos_min']))
    assert forwards <= FORWARDS_MAX, forwards
    cost = dict(atoms=96, reads_per_atom=4, read_multiply_adds=96 * 4 * 1152, products=96 * 3, coefficient_multiplies=96, adds=96, added_coefficients=96 * 4 * 1152 + 96)
    receipt = dict(predictions=preds, controls=checks, manifest=manifest, replay=dict(calibration=replay_cal, fresh=replay_fresh, export=export_err, target_piece_identity=piece_identity, parent_identity=parent_identity),
                   allocation=alloc, residual_piece_shares=share, piece_order=['40', '31', '22', '13', '04'], baseline=baseline, rows=rows, typed_over_untyped=ratio, cross_start_cosines=cos, scale=scale,
                   literal_cost=cost, forwards=forwards, backwards=0, seconds=time.monotonic() - start, peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
                   scope='Matched-capacity typed vs untyped correction of the frozen parent; no feature names, no intervention, no circuit claim.')
    out.write_text(json.dumps(receipt, indent=1)); print(json.dumps(dict(predictions=preds, replay=receipt['replay'], typed_over_untyped=ratio, forwards=forwards), indent=1), flush=True)


if __name__ == '__main__':
    main()
