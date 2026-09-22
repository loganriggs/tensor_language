#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_instrument pred_b_state_piece_dominates pred_c_context_low_dimensional pred_d_mixed_pieces_live pred_e_residual_in_context_pieces
"""Source/context geometry census of the selected MLP16->MLP17 quartic branch (no fitting).
Registered in SOURCE_GEOMETRY_CENSUS_PLAN_V1.md. x = S + C exactly (STATE = embedding + MLP writes, CONTEXT = attention writes,
both propagated through the lambda recurrence to the MLP16 input and divided by its rms). F4 and the fixed CP512 parent split exactly
into five bidegree pieces by the number of S factors; the residual splits likewise.
pred_a_instrument: STATE+CONTEXT reconstruct x_mid rel<=1e-4; pieces sum to stored native targets rel<=1e-4; parent pieces sum to parent rel<=1e-8.
pred_b_state_piece_dominates: mean over outputs 4-15 of the pure-state share s_40 >= .5.
pred_c_context_low_dimensional: covariance participation ratio of C <= .25 x that of S.
pred_d_mixed_pieces_live: mean over outputs 4-15 of s_31+s_22+s_13 >= .2.
pred_e_residual_in_context_pieces: mean over outputs 4-15 of the residual share in pieces with >=1 context factor >= .6.
Price: 12 forwards of 8 documents through block 16 + attention 16, one 4-document replay forward; 0 backwards; 0 fits. Bar 16 forwards.
"""
import os, sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; P = ROOT / 'basis_aligned/polynomial_causal/direct_tensor_match'
FORWARDS_MAX = 16
INPUTS = ['SOURCE_GEOMETRY_CENSUS_PLAN_V1.md', 'SENSITIVE_ROOT_CALIBRATION_V2.pt', 'NATIVE_QUARTIC_COVARIANCE_V1.pt', 'QUARTIC_ADDITIONAL_STATES_V1.pt',
          'MIXED_CP_FEATURES_SEED1001_V1.pt', 'EXPANDED_ROOT_EMPIRICAL_V1.pt']
TOKENS = ROOT / 'basis_aligned/bilinear_quotient/.rowcache/fineweb_n96_skip1200.pt'
SOURCES = ['embedding'] + [f'mlp{l}' for l in range(16)] + [f'attn{l}' for l in range(17)]


def bilinear(u, v, left, right, down):
    return ((u @ left.T) * (v @ right.T)) @ down.T


def quartic_pieces(S, C, w16, w17, lam, readers, chunk=1024):
    """Five bidegree pieces of F4 = B17(m, m) @ readers with m = lam * B16(x, x), x = S + C. Returns [5, N, 16]."""
    import torch
    L16, R16, D16 = w16; L17, R17, D17 = w17; out = []
    for a in range(0, S.shape[0], chunk):
        s, c = S[a:a + chunk], C[a:a + chunk]
        m20 = lam * bilinear(s, s, L16, R16, D16); m02 = lam * bilinear(c, c, L16, R16, D16)
        m11 = lam * (bilinear(s, c, L16, R16, D16) + bilinear(c, s, L16, R16, D16))
        B = lambda u, v: bilinear(u, v, L17, R17, D17) @ readers
        out.append(torch.stack([B(m20, m20), B(m20, m11) + B(m11, m20), B(m20, m02) + B(m02, m20) + B(m11, m11), B(m11, m02) + B(m02, m11), B(m02, m02)]))
    return torch.cat(out, 1)


def parent_pieces(S, C, factors, coefficients, chunk=1024):
    """Five bidegree pieces of the CP parent sum_k c_k prod_s (a_ks . x): coefficient of t^i in prod_s (alpha_s t + beta_s). Returns [5, N, 16]."""
    import torch
    out = []
    for a in range(0, S.shape[0], chunk):
        s, c = S[a:a + chunk], C[a:a + chunk]
        poly = None
        for f in factors:
            al, be = s @ f.T, c @ f.T                       # [n, K] each
            term = torch.stack([be, al])                     # coefficients of t^0, t^1
            if poly is None:
                poly = term
            else:
                new = torch.zeros(poly.shape[0] + 1, *poly.shape[1:], dtype=poly.dtype, device=poly.device)
                new[:-1] += poly * be; new[1:] += poly * al; poly = new
        out.append(torch.einsum('ink,gk->ing', poly, coefficients))
    return torch.cat(out, 1).flip(0)                     # poly[i] is the coefficient of t^i (i state factors); flip so index 0 = pure state (40), matching quartic_pieces


def controls():
    """CPU identities on planted weights: pieces sum to the full quartic; parent pieces sum to the parent."""
    import torch
    torch.manual_seed(7); dt = torch.float64; d, h, K, n = 12, 20, 9, 31
    L16, R16 = torch.randn(h, d, dtype=dt), torch.randn(h, d, dtype=dt); D16 = torch.randn(d, h, dtype=dt)
    L17, R17 = torch.randn(h, d, dtype=dt), torch.randn(h, d, dtype=dt); D17 = torch.randn(d, h, dtype=dt); readers = torch.randn(d, 5, dtype=dt)
    S, C = torch.randn(n, d, dtype=dt), torch.randn(n, d, dtype=dt); lam = 0.7
    x = S + C; m = lam * bilinear(x, x, L16, R16, D16); full = bilinear(m, m, L17, R17, D17) @ readers
    pieces = quartic_pieces(S, C, (L16, R16, D16), (L17, R17, D17), lam, readers, chunk=10)
    e1 = float((pieces.sum(0) - full).norm() / full.norm())
    fac = [torch.randn(K, d, dtype=dt) for _ in range(4)]; coef = torch.randn(5, K, dtype=dt)
    parent = torch.stack([x @ f.T for f in fac]).prod(0) @ coef.T
    pp = parent_pieces(S, C, fac, coef, chunk=10); e2 = float((pp.sum(0) - parent).norm() / parent.norm())
    only_s = parent_pieces(S, torch.zeros_like(C), fac, coef, chunk=10); only_c = parent_pieces(torch.zeros_like(S), C, fac, coef, chunk=10)
    e4 = float(only_s[1:].abs().sum()) + float(only_c[:4].abs().sum())   # C = 0 leaves only index 0 (pure state); S = 0 leaves only index 4 (pure context)
    qs = quartic_pieces(S, torch.zeros_like(C), (L16, R16, D16), (L17, R17, D17), lam, readers, chunk=10); qc = quartic_pieces(torch.zeros_like(S), C, (L16, R16, D16), (L17, R17, D17), lam, readers, chunk=10)
    e5 = float(qs[1:].abs().sum()) + float(qc[:4].abs().sum())            # the same ordering test on the quartic pieces
    e6 = float((only_s[0] - parent_pieces(S, torch.zeros_like(C), fac, coef, chunk=10).sum(0)).abs().sum())
    assert e1 < 1e-12 and e2 < 1e-12 and e4 < 1e-12 and e5 < 1e-12 and e6 < 1e-12, (e1, e2, e4, e5, e6)
    return dict(quartic_piece_identity=e1, parent_piece_identity=e2, pure_state_only_when_context_zero=e4, quartic_pieces_ordered=e5, parent_pieces_ordered=e6)


def main():
    import torch
    torch.set_num_threads(2); torch.set_grad_enabled(False); sys.path.insert(0, str(P))
    plan = dict(candidate_id='source_geometry.census_v1', forwards_max=FORWARDS_MAX, model_backwards=0, model_updates=0, fit_parameters=0,
                gpu_accessed=False, model_loaded=False, execution_policy='managed_queue_only', panel='calibration 96x64', sources=len(SOURCES),
                bars=dict(instrument=1e-4, parent_identity=1e-8, state_share=.5, context_pr_ratio=.25, mixed_share=.2, residual_context_share=.6))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        plan['controls'] = controls(); print(json.dumps(plan, indent=2, sort_keys=True)); return
    import torch.nn.functional as F
    from native_feature_capture import capture
    from circuit_fast_screen_producer import Bilin18TorchBackend
    torch.backends.cuda.matmul.allow_tf32 = False; start = time.monotonic()
    out = P / 'SOURCE_GEOMETRY_CENSUS_V1.json'; archive = P / 'SOURCE_GEOMETRY_STATES_V1.pt'; assert not out.exists() and not archive.exists()
    manifest = {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in INPUTS}; manifest['tokens'] = hashlib.sha256(TOKENS.read_bytes()).hexdigest()
    checks = controls()
    model = Bilin18TorchBackend.load('cuda').model.float(); blocks = model.transformer.h; dev = 'cuda'
    tokens = torch.load(TOKENS, weights_only=True)[:96, :64].contiguous()
    labels = torch.load(P / 'SENSITIVE_ROOT_CALIBRATION_V2.pt', weights_only=True); cov = torch.load(P / 'NATIVE_QUARTIC_COVARIANCE_V1.pt', weights_only=True)
    extra = torch.load(P / 'QUARTIC_ADDITIONAL_STATES_V1.pt', weights_only=True)
    digest = lambda z: hashlib.sha256(z.contiguous().numpy().tobytes()).hexdigest()
    assert digest(tokens) == labels['token_sha256']['calibration']
    calx = torch.cat([cov['panels'][0]['rows'], extra['rows']]).to(dev).double(); target = labels['panels'][0]['target'].to(dev).double()
    assert calx.shape == (6144, 1152) and target.shape == (6144, 16)
    # ---- capture every propagated write at every position -----------------------------------------------------------------------
    writes = {name: torch.empty(96, 64, 1152, dtype=torch.float64) for name in SOURCES if name != 'embedding'}
    x0_all = torch.empty(96, 64, 1152, dtype=torch.float64); xmid_all = torch.empty_like(x0_all); box = {'sel': None}
    def hook(name):
        def save(module, args, output):
            y = output[0] if isinstance(output, (tuple, list)) else output
            writes[name][box['sel']] = y.double().cpu()
        return save
    handles = [blocks[l].attn.register_forward_hook(hook(f'attn{l}')) for l in range(16)] + [blocks[l].mlp.register_forward_hook(hook(f'mlp{l}')) for l in range(16)]
    forwards = 0
    try:
        for a in range(0, 96, 8):
            box['sel'] = slice(a, a + 8); ids = tokens[a:a + 8].to(dev)
            x = F.rms_norm(model.transformer.wte(ids), (1152,)); x0 = x; v1 = None; x0_all[box['sel']] = x0.double().cpu()
            for l in range(16):
                x, v1 = blocks[l](x, v1, x0)
            b16 = blocks[16]; xin = b16.lambdas[0] * x + b16.lambdas[1] * x0
            a16 = b16.attn(F.rms_norm(xin, (1152,)), v1); a16 = a16[0] if isinstance(a16, (tuple, list)) else a16
            writes['attn16'][box['sel']] = a16.double().cpu(); xmid_all[box['sel']] = (xin + a16).double().cpu(); forwards += 1
    finally:
        for h in handles: h.remove()
    # ---- replay: the normalized MLP16 input equals the stored calibration rows -----------------------------------------------------
    ref = capture(model, tokens[:4].to(dev))['x16'].flatten(0, 1).double(); forwards += 1
    xmid = xmid_all.flatten(0, 1).to(dev); rms = (xmid.square().mean(-1, keepdim=True)).sqrt(); xn = xmid / rms
    replay_x16 = float((xn[:256] - ref).norm() / ref.norm()); replay_rows = float((xn - calx).norm() / calx.norm())
    # ---- exact propagation coefficients into the MLP16 input --------------------------------------------------------------------
    lam = torch.stack([b.lambdas.detach().double().cpu() for b in blocks])          # [18, 2]
    cE = 1.0
    for l in range(17):
        cE = float(lam[l, 0]) * cE + float(lam[l, 1])
    coef = {'embedding': cE}
    for l in range(16):
        c = float(torch.prod(lam[l + 1:17, 0])); coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    coef['attn16'] = 1.0
    parts = {'embedding': cE * x0_all.flatten(0, 1)}
    parts.update({name: coef[name] * writes[name].flatten(0, 1) for name in writes})
    STATE = parts['embedding'] + sum(parts[f'mlp{l}'] for l in range(16)); CONTEXT = sum(parts[f'attn{l}'] for l in range(17))
    recon = float(((STATE + CONTEXT).to(dev) - xmid).norm() / xmid.norm())
    S = STATE.to(dev) / rms; C = CONTEXT.to(dev) / rms
    split_exact = float(((S + C) - xn).norm() / xn.norm())
    # ---- per-source geometry -----------------------------------------------------------------------------------------------------
    xe = float(xn.square().sum())
    source_energy = {name: float((parts[name].to(dev) / rms).square().sum() / xe) for name in SOURCES}
    def spectrum(Z):
        Zc = Z - Z.mean(0, keepdim=True); ev = torch.linalg.eigvalsh(Zc.T @ Zc / Z.shape[0]).flip(0).clamp_min(0)
        return dict(participation_ratio=float(ev.sum() ** 2 / ev.square().sum()), top64=ev[:64].tolist(), trace=float(ev.sum()))
    spec = dict(S=spectrum(S), C=spectrum(C), x=spectrum(xn))
    group_energy = dict(S=float(S.square().sum() / xe), C=float(C.square().sum() / xe), cross=float(2 * (S * C).sum() / xe))
    # ---- bidegree pieces of the target and the parent ---------------------------------------------------------------------------
    state = model.state_dict()
    w = lambda l, n: state[f'transformer.h.{l}.mlp.{n}.weight'].double()
    w16 = (w(16, 'Left'), w(16, 'Right'), w(16, 'Down')); w17 = (w(17, 'Left'), w(17, 'Right'), w(17, 'Down')); lam17 = float(lam[17, 0])
    writer = torch.load(P / 'EXPANDED_ROOT_EMPIRICAL_V1.pt', weights_only=True)['writer'].to(dev).double(); vocab = state['lm_head.weight'].double()
    uw = vocab @ writer; readers = vocab.T @ uw / uw.square().sum(0); del vocab, uw
    pieces = quartic_pieces(S, C, w16, w17, lam17, readers)                                   # [5, N, 16]
    direct = quartic_pieces(xn, torch.zeros_like(xn), w16, w17, lam17, readers)[0]           # F4(x) computed the same way
    replay_target = float((pieces.sum(0) - target).norm() / target.norm()); replay_direct = float((pieces.sum(0) - direct).norm() / direct.norm())
    p = torch.load(P / 'MIXED_CP_FEATURES_SEED1001_V1.pt', weights_only=True); pf = [f.to(dev).double() for f in p['factors']]; pC = p['coefficients'].to(dev).double()
    parent = torch.stack([xn @ f.T for f in pf]).prod(0) @ pC.T                                # the parent as the hybrid experiment evaluates it (at the stored row)
    parent_sc = torch.stack([(S + C) @ f.T for f in pf]).prod(0) @ pC.T                         # the parent at the split input the pieces are built from
    ppieces = parent_pieces(S, C, pf, pC)
    parent_identity = float((ppieces.sum(0) - parent_sc).norm() / parent_sc.norm())              # the bookkeeping identity (exact algebra)
    parent_identity_at_x = float((ppieces.sum(0) - parent).norm() / parent.norm())              # the same against the parent at x: bounded by split_exact (float32 capture), ~1e-7 in run 1
    rpieces = pieces - ppieces; residual = target - parent
    def accounting(Q, total):
        gram = torch.einsum('ing,jng->gij', Q, Q)                                          # [16, 5, 5]
        diag = torch.diagonal(gram, dim1=1, dim2=2); share = diag / diag.sum(1, keepdim=True)
        return dict(gram=gram.tolist(), diagonal_share=share.tolist(), total_energy=total.square().sum(0).tolist(), sum_of_piece_energies=diag.sum(1).tolist())
    acc_target = accounting(pieces, pieces.sum(0)); acc_parent = accounting(ppieces, ppieces.sum(0)); acc_resid = accounting(rpieces, rpieces.sum(0))
    share = torch.tensor(acc_target['diagonal_share'])[4:]; rshare = torch.tensor(acc_resid['diagonal_share'])[4:]
    s40 = float(share[:, 0].mean()); mixed = float(share[:, 1:4].sum(1).mean()); rctx = float(rshare[:, 1:].sum(1).mean())
    pr_ratio = spec['C']['participation_ratio'] / spec['S']['participation_ratio']
    resid_energy_share = float(residual[:, 4:].square().sum() / target[:, 4:].square().sum())
    per_output_small_residual = (residual.square().sum(0) / target.square().sum(0)).sqrt().tolist()
    preds = dict(pred_a_instrument=recon <= 1e-4 and replay_target <= 1e-4 and parent_identity <= 1e-8,
                 pred_b_state_piece_dominates=s40 >= .5, pred_c_context_low_dimensional=pr_ratio <= .25,
                 pred_d_mixed_pieces_live=mixed >= .2, pred_e_residual_in_context_pieces=rctx >= .6)
    print(json.dumps(dict(recon=recon, split_exact=split_exact, replay_x16=replay_x16, replay_rows=replay_rows, replay_target=replay_target, replay_direct=replay_direct, parent_identity=parent_identity,
                          group_energy=group_energy, pr=dict(S=spec['S']['participation_ratio'], C=spec['C']['participation_ratio'], x=spec['x']['participation_ratio']),
                          s40=s40, mixed=mixed, residual_context_share=rctx, mean_small_share=share.mean(0).tolist(), mean_small_residual_share=rshare.mean(0).tolist(),
                          resid_energy_share_small=resid_energy_share, predictions=preds, forwards=forwards), indent=1), flush=True)
    if forwards > FORWARDS_MAX:
        raise SystemExit(f'price exceeded: {forwards} > {FORWARDS_MAX}')
    torch.save(dict(S=S.float().cpu(), C=C.float().cpu(), target_pieces=pieces.cpu(), parent_pieces=ppieces.cpu(), piece_order=['40', '31', '22', '13', '04'],
                    token_sha256=labels['token_sha256']['calibration'], parent='MIXED_CP_FEATURES_SEED1001_V1.pt', readers_from='EXPANDED_ROOT_EMPIRICAL_V1.pt',
                    scope='Calibration 96x64 states; exact state/context split of the normalized MLP16 input and exact bidegree pieces of F4 and the CP512 parent. No fitting.'), archive)
    out.write_text(json.dumps(dict(predictions=preds, controls=checks, manifest=manifest, replay=dict(parent_identity_at_x=parent_identity_at_x, state_context_reconstruction=recon, split_exact=split_exact, x16_capture=replay_x16, calibration_rows=replay_rows, target=replay_target, direct=replay_direct, parent_identity=parent_identity),
                                   propagation_coefficients=coef, source_energy_share=source_energy, group_energy_share=group_energy, spectra=spec, piece_order=['40', '31', '22', '13', '04'],
                                   target_accounting=acc_target, parent_accounting=acc_parent, residual_accounting=acc_resid, summary=dict(mean_small_state_share=s40, mean_small_mixed_share=mixed, residual_context_share=rctx, context_over_state_pr=pr_ratio,
                                   residual_energy_share_outputs_4_15=resid_energy_share, per_output_residual_relative_rms=per_output_small_residual),
                                   forwards=forwards, backwards=0, seconds=time.monotonic() - start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                   scope='Exact census, no fitting, calibration panel only; licenses (if pred_e passes) a bidegree-typed correction rung, not a circuit claim.'), indent=2) + '\n')
    print(json.dumps(preds))


if __name__ == '__main__':
    main()
