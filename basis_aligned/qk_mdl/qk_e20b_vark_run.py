"""E20b VARIABLE-K CODEBOOK SLOTS -- per-slot pursuit depth from the E20
error decomposition (fresh single-epoch batch-16, recipe conventions;
results -> qk_e20b.json).

MOTIVATION (E20 error decomposition, BRAINSTORM_STATE 2026-08-07): E20a's
+0.134 CE cost vs the continuous E19a parent is token-heavy-tailed and its
slot split shows mid/late ATTENTION slots leaving ~45-46% of content RMS
unexplained after 2 pursuit codes vs 24-25% for MLP/early slots --
attention messages are ~2x harder to quantize at k=2. E20b spends the code
budget where the residuals are: k=4 pursuit steps for ATTENTION slots (even
slot indices 0,2,..,22), k=2 for MLP slots (odd indices), same n=256
unit-norm codebooks per slot, same EMA/commitment/dead-reinit machinery,
same E19a base architecture (E15c bandwidth reinvestment 24x15, stream 360,
compute 264, hidden 1056, group-lasso 1e-4, full readable recipe), same
straight-through matching pursuit (scales = continuous inner products; the
content-bits term counts code identities only).

GLOBAL BUDGET DOCUMENTED (not equal by construction): content bits/token =
sum over quantized slots of the joint code-tuple usage entropy; E20a's
measured budget was 342.101 bits/token (23 slots x k=2 pairs); E20b's
4-tuples on the 12 attention-write slots RAISE the identity budget --
reported next to E20a's number, cost-per-bit is part of the verdict, and
slot 23 stays excluded (readout-only, never quantized -- E20 exemption).

CONTROLS BEFORE TRAINING (hard gates, the E20 controls rerun through this
class):
  1. BYPASS: quantization off == the E19a parent (make_e15c) bit-exactly at
     init (parameter identity + forward identity) AND over a 3-step
     training run (aux hook + buffers inert);
  2. CAPACITY: n >= #distinct vectors with k=15 pursuit -> ~0 error;
  3. PLANTED TOY at k=4: the EMA + dead-reinit + commitment machinery must
     recover 10 planted cluster centers through the exact mp_quantize/
     ema_update functions with the attention-slot pursuit depth;
  4. STRUCTURAL k-wiring: a collected forward must yield 4 code columns on
     every even (attention) slot and 2 on every odd (MLP) slot;
  5. COV PIPELINE: with quantization off, the quantized-content covariance
     pass reproduces qk_e18_probe_upgrades.gen_slot_covariances exactly.

REGISTERED PREDICTIONS (merged before training):
  (i)   CE cost vs the continuous E19a parent (paired, fresh held) drops
        below +0.07 (E20a: +0.134);
  (ii)  the ATTENTION-slot pursuit residual fraction after its k codes
        (final residual RMS / content RMS) drops to ~MLP levels: mean over
        attention slots <= 0.30 (E20a mid/late attention: 0.45-0.46);
  (iii) readability stays >= 0.85 (covariance-composed wiring Spearman with
        quantized content; E20a: 0.8866).

MEASURE: everything E20a measured -- paired fresh CE vs E19a/E15c/E9a/E0a/
E0b + vs the E20a scratch arm, per-slot code usage/dead fraction/entropies,
per-pursuit-step residual RMS ladders, content bits/token, heldloss.npy +
per-seq file, wiring-trajectory logging (standing harness), codebook
snapshots + dead-code event log (qk_e20b_* files), generalized light probe
+ covariance-composed wiring on the quantized forward. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, C, DEPTH, F, nn, torch
import qk_e7_evenout_run as E7R
import qk_e15_reinvest_run as E15R
import qk_e18_probe_upgrades as E18U
import qk_e17_composed_wiring as E17
import qk_e20_codebook_run as E20R
import qk_deeproute_train_2 as R2

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e20b.json')
GC20B = 1e-4                              # the E19a lasso (unchanged)
QZ_N = E20R.QZ_N
K_ATTN, K_MLP = 4, 2
NG = E.NGROUP
E20A_BITS = 342.101                       # measured E20a budget (qk_e20.json)
E20A_COST = 0.1344                        # E20a minus_e19a paired


class E20bRoute(E20R.E20Route):
    """E20Route with per-slot pursuit depth: k=4 on attention-write slots
    (even indices), k=2 on MLP-write slots (odd). Codebooks, EMA,
    commitment, dead-reinit, caching and the quantization interface are
    inherited unchanged."""

    def __init__(self, variant, depth, s, Dc, NH, HD, hidden,
                 n_codes=QZ_N):
        super().__init__(variant, depth, s, Dc, NH, HD, hidden, n_codes)
        self.qz_ksteps = [K_ATTN if k % 2 == 0 else K_MLP
                          for k in range(2 * depth)]

    def _qz_slot(self, k, v, collect, cache):
        """E20Route._qz_slot with ksteps = self.qz_ksteps[k] and a
        generalized residual ladder (content, after code 1, after code 2,
        final, count)."""
        ks = self.qz_ksteps[k]
        upd = self.training and torch.is_grad_enabled() and cache is not None
        with torch.autocast('cpu' if v.device.type == 'cpu' else 'cuda',
                            enabled=False):
            x32 = v.detach().float().reshape(-1, self.s)
            recon, idxs, alphas, resids = E20R.mp_quantize(
                x32, self.qz_codebook[k], ks)
            if upd:
                _, reinit = E20R.ema_update(
                    self.qz_codebook[k], self.qz_ema[k],
                    self.qz_last_used[k], self.qz_usage[k],
                    int(self.qz_step), x32, idxs, alphas, resids)
                if reinit is not None:
                    self.qz_dead_events.append(
                        {'step': int(self.qz_step), 'slot': k,
                         'n_reinit': len(reinit[0]),
                         'codes': reinit[0][:40],
                         'source_flat_token_idx': reinit[1][:40]})
            q = recon.view(v.shape)
        out = v + (q.to(v.dtype) - v).detach()
        if upd:
            commit = (v.float() - q.detach()).pow(2).mean()
            self._commit_sum = commit if self._commit_sum is None \
                else self._commit_sum + commit
            self._commit_n += 1
        if collect is not None:
            collect.setdefault('qcontent', {})[k] = q.detach()
            if 'codes' in collect:
                collect['codes'].setdefault(k, []).append(
                    torch.stack(idxs, 1).to(torch.int32).cpu())
            if 'scales' in collect:
                collect['scales'].setdefault(k, []).append(
                    torch.stack(alphas, 1).float().cpu())
            if 'resid_stats' in collect:
                r_fin = x32 - recon
                a1 = (resids[1].pow(2).sum() if len(resids) > 1
                      else r_fin.pow(2).sum())
                a2 = (resids[2].pow(2).sum() if len(resids) > 2
                      else r_fin.pow(2).sum())
                collect['resid_stats'].setdefault(k, []).append([
                    float(x32.pow(2).sum()), float(a1), float(a2),
                    float(r_fin.pow(2).sum()), float(x32.shape[0])])
        if cache is not None:
            cache[k] = out
        return out


def make_e20b(s=None, qz_on=True):
    hidden = 4 * Q.D
    if s is None:
        s, _ = E15R.solve_slot_c(hidden)
    torch.manual_seed(Q.SEED)
    m = E20bRoute('E20b', DEPTH, s, Q.D, Q.NH, Q.HD, hidden).to(E.DEV)
    m.qz_on = qz_on
    return m


# ---------------- E20b-specific training-time logging ----------------
_SNAPSHOTS_B = {}


def flush_dead_events_b(model, phase='train'):
    ev = getattr(model, 'qz_dead_events', None)
    if ev:
        with open(E.jpath('qk_e20b_deadcode_events.jsonl'), 'a') as f:
            for e_ in ev:
                e_['phase'] = phase
                f.write(json.dumps(e_) + '\n')
        model.qz_dead_events = []


def qz_step_cb_b(step, model):
    if not getattr(model, 'qz_on', False):
        return
    flush_dead_events_b(model)
    if step % 1000 == 0:
        _SNAPSHOTS_B[f'step{step:05d}'] = \
            model.qz_codebook.detach().cpu().numpy()
        np.savez(E.jpath('qk_e20b_codebook_snapshots.npz'), **_SNAPSHOTS_B)


qz_step_cb_b.every = 250


def trainer_b(lr, gc, steps, **kw):
    kw.setdefault('step_cb', qz_step_cb_b)
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)


# ---------------- controls ----------------
def control_bypass(s_c):
    key = 'control1_bypass'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m19 = E15R.make_e15c(s=s_c).eval().float()
    m20 = make_e20b(s=s_c, qz_on=False).eval().float()
    p19 = dict(m19.named_parameters())
    pdiff = max(float((p - p19[nm]).abs().max())
                for nm, p in m20.named_parameters())
    idx = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        d_fwd = float((m20(idx) - m19(idx)).abs().max())
        m20.qz_on = True
        d_qz = float((m20(idx) - m19(idx)).abs().max())
        m20.qz_on = False
    del m19, m20
    torch.cuda.empty_cache()
    parent = E20R.three_step(lambda: E15R.make_e15c(s=s_c), GC20B)
    mine = E20R.three_step(lambda: make_e20b(s=s_c, qz_on=False), GC20B)
    step_diff = max(abs(a - b) for a, b in
                    zip(parent['per_step_ce'], mine['per_step_ce']))
    held_diff = abs(parent['held100_ce'] - mine['held100_ce'])
    rec = {'param_identity_max_abs_diff': pdiff,
           'forward_bypass_max_logit_diff': d_fwd,
           'forward_quantized_max_logit_diff_at_init_informational': d_qz,
           'train3_max_per_step_abs_diff': step_diff,
           'train3_held100_abs_diff': held_diff,
           'pass': bool(pdiff == 0.0 and d_fwd == 0.0 and step_diff == 0.0
                        and held_diff < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: params {pdiff:.1e}, forward {d_fwd:.1e}, 3-step "
          f"{step_diff:.1e}/{held_diff:.1e} (quantized-at-init shift "
          f"{d_qz:.3f}) -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_capacity(s_c):
    key = 'control2_capacity'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m = make_e20b(s=s_c, qz_on=True).eval().float()
    b = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        e = F.rms_norm(m.wte(b), (m.Ws,))
        v = F.rms_norm(e[..., :m.s], (m.s,)).reshape(-1, m.s).float()[:128]
        uniq = torch.unique(v, dim=0)
        Cb = uniq / uniq.norm(dim=1, keepdim=True).clamp_min(1e-8)
        recon, _, _, _ = E20R.mp_quantize(v, Cb, 15)
        rel = float((v - recon).norm() / v.norm().clamp_min(1e-12))
    del m
    torch.cuda.empty_cache()
    rec = {'n_distinct': int(uniq.shape[0]), 'k_steps': 15,
           'rel_error': rel, 'pass': bool(rel < 1e-4)}
    E.merge(JP, key, rec)
    print(f"{key}: {rec['n_distinct']} distinct vectors, k=15, rel error "
          f"{rel:.2e} -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_toy_k4():
    """Planted-toy EMA recovery at the ATTENTION pursuit depth (k=4),
    through the exact mp_quantize/ema_update functions the model runs."""
    key = 'control3_toy_ema_k4'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    dim, ncen, ncode = 15, 10, 64
    g = torch.Generator().manual_seed(7)
    cen = torch.randn(ncen, dim, generator=g)
    cen = cen / cen.norm(dim=1, keepdim=True) * math.sqrt(dim)
    Cb = torch.randn(ncode, dim, generator=g)
    Cb = Cb / Cb.norm(dim=1, keepdim=True)
    M = Cb.clone()
    last_used = torch.zeros(ncode, dtype=torch.long)
    usage = torch.zeros(ncode, dtype=torch.long)
    steps = 120 if E.SMOKE else 600
    commit0 = commit_end = None
    for t in range(1, steps + 1):
        lab = torch.randint(0, ncen, (512,), generator=g)
        x = cen[lab] + 0.01 * torch.randn(512, dim, generator=g)
        recon, idxs, alphas, resids = E20R.mp_quantize(x, Cb, K_ATTN)
        cv = float((x - recon).pow(2).mean())
        if t == 1:
            commit0 = cv
        commit_end = cv
        E20R.ema_update(Cb, M, last_used, usage, t, x, idxs, alphas, resids)
    lab = torch.randint(0, ncen, (2048,), generator=g)
    x = cen[lab] + 0.01 * torch.randn(2048, dim, generator=g)
    recon, idxs, _, _ = E20R.mp_quantize(x, Cb, K_ATTN)
    rel_mse = float((x - recon).pow(2).mean() / x.pow(2).mean())
    cosmat = (cen / cen.norm(dim=1, keepdim=True)) @ Cb.t()
    center_maxcos = cosmat.max(1).values
    samp_cos = (Cb[idxs[0]] * (cen[lab] / cen[lab].norm(dim=1, keepdim=True))
                ).sum(1)
    frac_matched = float((samp_cos > 0.99).float().mean())
    rec = {'n_centers': ncen, 'n_codes': ncode, 'k_steps': K_ATTN,
           'steps': steps,
           'center_max_cos_min': float(center_maxcos.min()),
           'frac_samples_step1_code_cos_gt_0.99': frac_matched,
           'final_rel_mse': rel_mse,
           'commit_first_step': commit0, 'commit_last_step': commit_end,
           'pass': bool(float(center_maxcos.min()) > 0.99
                        and frac_matched >= 0.99 and rel_mse < 1e-2
                        and commit_end < 0.1 * commit0)}
    E.merge(JP, key, rec)
    print(f"{key}: center cos min {rec['center_max_cos_min']:.4f}, matched "
          f"{frac_matched:.3f}, rel MSE {rel_mse:.2e}, commitment "
          f"{commit0:.3f} -> {commit_end:.5f} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_k_wiring(s_c):
    """Gate 4: a collected forward yields 4 code columns on every even slot,
    2 on every odd slot (among slots quantized at that sequence length)."""
    key = 'control4_k_wiring'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m = make_e20b(s=s_c, qz_on=True).eval().float()
    with torch.no_grad():
        col = {'codes': {}}
        m(E.OLD_HELD[:2, :Q.T], collect=col)
    got = {k: int(v[0].shape[1]) for k, v in col['codes'].items()}
    ok = all(got[k] == (K_ATTN if k % 2 == 0 else K_MLP) for k in got)
    rec = {'code_columns_per_slot': {str(k): got[k] for k in sorted(got)},
           'expected': f'{K_ATTN} on even (attention) slots, {K_MLP} on '
                       'odd (MLP) slots', 'pass': bool(ok and len(got) > 0)}
    E.merge(JP, key, rec)
    print(f"{key}: {len(got)} quantized slots, wiring "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    torch.cuda.empty_cache()


def control_cov_pipeline(s_c):
    key = 'control5_cov_pipeline'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    dims = [s_c] * NG
    m = make_e20b(s=s_c, qz_on=False)
    held = np.load(E.HELD_PATH)[33000:33032].astype(np.int64)
    rows = torch.from_numpy(held)
    Cb_g, _, Cr_g, n_g = E18U.gen_slot_covariances(m, rows, E.DEV, dims,
                                                   remnant=False)
    Cb_e, Cr_e, n_e = E20R.e20_slot_covariances(m, rows, E.DEV, dims)
    d_b = max(float((a - b).abs().max()) for a, b in zip(Cb_g, Cb_e))
    d_r = max(float((a - b).abs().max()) for a, b in zip(Cr_g, Cr_e))
    rec = {'max_abs_diff_content_cov': d_b, 'max_abs_diff_readout_cov': d_r,
           'pass': bool(n_g == n_e and d_b < 1e-6 and d_r < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: content {d_b:.2e}, readout {d_r:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    torch.cuda.empty_cache()


# ---------------- variable-k code statistics ----------------
def entropy_bits(counts):
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def tuple_entropy_bits(codes):
    """Joint usage entropy of the full code tuple (rows of `codes`)."""
    _, cnt = np.unique(codes, axis=0, return_counts=True)
    return entropy_bits(cnt)


def code_stats_b(s_c):
    key = 'code_stats_E20b'
    if key in E.loadj(JP):
        print(f"{key}: already done -- skip", flush=True)
        return
    stem = 'qk_e20b_a'
    if E.SMOKE:
        m = make_e20b(s=s_c)
    else:
        if not os.path.exists(E.ckpath(stem)):
            return
        m, _ = E.load_arm(stem, lambda: make_e20b(s=s_c))
    m.eval().float()
    rows = Q.HELD
    per_slot = {k: [] for k in range(NG)}
    resid_acc = {k: np.zeros(5) for k in range(NG)}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            col = {'codes': {}, 'resid_stats': {}}
            m(b[:, :Q.T], collect=col)
            for k, chunks in col['codes'].items():
                per_slot[k].append(torch.cat(chunks, 0))
            for k, sums in col['resid_stats'].items():
                for s5 in sums:
                    resid_acc[k] += np.asarray(s5)
            if (i // 8) % 40 == 0:
                print(f"  code stats: {i + b.shape[0]}/{len(rows)} seqs "
                      f"({time.time() - t0:.0f}s)", flush=True)
    quantized_slots = sorted(k for k in per_slot if per_slot[k])
    slots_rec = {}
    bits_total = 0.0
    dead_num = dead_den = 0
    attn_fracs, mlp_fracs = [], []
    for k in quantized_slots:
        codes = torch.cat(per_slot[k], 0).long().numpy()  # (Ntok, k_slot)
        ntok, ks = codes.shape
        step_h = [round(entropy_bits(np.bincount(codes[:, jj],
                                                 minlength=QZ_N)), 3)
                  for jj in range(ks)]
        hj = tuple_entropy_bits(codes)
        used = np.zeros(QZ_N, dtype=bool)
        for jj in range(ks):
            used |= np.bincount(codes[:, jj], minlength=QZ_N) > 0
        bits_total += hj
        dead_num += int(QZ_N - used.sum())
        dead_den += QZ_N
        sx, sr1, sr2, srf, ncnt = resid_acc[k]
        rms_c = float(np.sqrt(sx / ncnt))
        rms_f = float(np.sqrt(srf / ncnt))
        frac = rms_f / max(rms_c, 1e-12)
        (attn_fracs if k % 2 == 0 else mlp_fracs).append(frac)
        slots_rec[str(k)] = {
            'module': R2.stream_name(k + 1), 'k_steps': ks,
            'n_tokens': int(ntok),
            'distinct_codes_used': int(used.sum()),
            'dead_fraction': round(float(1.0 - used.sum() / QZ_N), 4),
            'entropy_per_step_bits': step_h,
            'entropy_joint_tuple_bits': round(hj, 3),
            'pursuit_residual_rms': {
                'content': round(rms_c, 4),
                'after_code1': round(float(np.sqrt(sr1 / ncnt)), 4),
                'after_code2': round(float(np.sqrt(sr2 / ncnt)), 4),
                'final': round(rms_f, 4)},
            'final_residual_rms_fraction': round(frac, 4),
            'energy_captured_total': round(1.0 - srf / sx, 4)}
    dead_frac = dead_num / max(dead_den, 1)
    mean_attn = float(np.mean(attn_fracs)) if attn_fracs else float('nan')
    mean_mlp = float(np.mean(mlp_fracs)) if mlp_fracs else float('nan')
    rec = {'rows': f'fresh held ({len(rows)} seqs x {Q.T} tokens), fp32',
           'quantized_slots': quantized_slots,
           'excluded_slot_23': 'mlp11 write, readout-only, never quantized '
                               '(E20 exemption)',
           'codebook_n': QZ_N,
           'k_select': {'attention_slots': K_ATTN, 'mlp_slots': K_MLP},
           'dead_code_fraction_overall': round(dead_frac, 4),
           'content_bits_per_token_sum_joint_entropy': round(bits_total, 3),
           'e20a_budget_bits_per_token': E20A_BITS,
           'budget_delta_vs_e20a_bits': round(bits_total - E20A_BITS, 3),
           'mean_final_residual_rms_fraction_attention': round(mean_attn, 4),
           'max_final_residual_rms_fraction_attention': round(
               float(np.max(attn_fracs)), 4) if attn_fracs else None,
           'mean_final_residual_rms_fraction_mlp': round(mean_mlp, 4),
           'per_slot': slots_rec}
    E.merge(JP, key, rec)
    print(f"code stats: dead {dead_frac:.3f}, bits/token {bits_total:.1f} "
          f"(E20a {E20A_BITS}), attention final-resid RMS fraction mean "
          f"{mean_attn:.3f} vs MLP {mean_mlp:.3f}", flush=True)
    del m
    torch.cuda.empty_cache()


# ---------------- probes (E20's generalized probe on the E20b forward) ----------------
def probe_e20b(s_c):
    stem = 'qk_e20b_a'
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    dims = [s_c] * NG
    j = E.loadj(JP)
    if 'light_probe_E20b_var_dims' not in j:
        print('E20b light probe (quantized forward) ...', flush=True)
        m, _ = E.load_arm(stem, lambda: make_e20b(s=s_c))
        Ws = m.wte.weight.shape[1]
        base, dce = E18U.gen_consumption(m, Ws)
        wp = E18U.wpairs(m, dims)
        G = E18U.gen_gram_table(m, dims)
        sup = E18U.score(G, wp)
        cau = [dce[li][si] for li, si in wp]
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        agr = E17.agreement(sup, cau, eff)
        rec = {'checkpoint': f'{stem}.pt',
               'base_ce_fp32_abl_oldheld': round(base, 5),
               'wiring_n_pairs': len(wp),
               'wiring_spearman_all': agr['spearman_all'],
               'wiring_n_effectual': len(eff),
               'wiring_spearman_effectual': agr['spearman_effectual'],
               'wiring_top10_precision': agr['top10_precision'],
               'consumption_matrix': {str(li): {str(si): round(v, 6)
                                                for si, v in dce[li].items()}
                                      for li in dce}}
        E.merge(JP, 'light_probe_E20b_var_dims', rec)
        print(f"E20b wiring Spearman all {agr['spearman_all']} "
              f"effectual({len(eff)}) {agr['spearman_effectual']} "
              f"top10 {agr['top10_precision']}", flush=True)
        del m
        torch.cuda.empty_cache()
    j = E.loadj(JP)
    if 'composed_wiring_E20b' not in j:
        lp = j['light_probe_E20b_var_dims']
        print('E20b covariance-composed re-scoring ...', flush=True)
        m, _ = E.load_arm(stem, lambda: make_e20b(s=s_c))
        wp = E18U.wpairs(m, dims)
        cau = E18U.stored_cau(lp, wp)
        G = E18U.gen_gram_table(m, dims)
        plain = E18U.score(G, wp)
        held = np.load(E.HELD_PATH)[33000:33000 + E18U.N_COV].astype(np.int64)
        Cb, Cr, n_samp = E20R.e20_slot_covariances(
            m, torch.from_numpy(held), E.DEV, dims)
        cov = E18U.score(G, wp, lambda li, si: Cb[si - 1])
        cov_ro = E18U.score(G, wp, lambda li, si:
                            Cr[si - 1] if li == DEPTH else Cb[si - 1])
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        tables = {'plain': E17.agreement(plain, cau, eff),
                  'cov_composed': E17.agreement(cov, cau, eff),
                  'cov_composed_readout_globalnorm':
                      E17.agreement(cov_ro, cau, eff)}
        chk = abs(tables['plain']['spearman_all'] - lp['wiring_spearman_all'])
        assert chk <= 1e-3, \
            f'E20b plain does not reproduce the light probe ({chk})'
        E.merge(JP, 'composed_wiring_E20b', {
            'checkpoint': f'{stem}.pt',
            'plain_reproduction_abs_diff': round(chk, 6),
            'n_samples': n_samp, 'tables': tables})
        print(f"E20b plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
        del m
        torch.cuda.empty_cache()


def summarize_b():
    j = E.loadj(JP)
    summary = {'parents': {'E19a': 4.9742, 'E20a_scratch_cost': E20A_COST}}
    if 'E20b' in j:
        row = {'final_held_ce_fresh_bf16':
               j['E20b']['final_held_ce_fresh_bf16'],
               'diverged': j['E20b']['diverged']}
        dce = None
        if 'E20b_minus_e19a_fresh' in j:
            dce = round(j['E20b_minus_e19a_fresh']['minus_e19a'], 4)
            row['minus_e19a_paired'] = dce
            row['minus_e19a_se_seq'] = round(
                j['E20b_minus_e19a_fresh']['minus_e19a_se_seq'], 5)
        if 'E20b_minus_e20a_fresh' in j:
            row['minus_e20a_scratch_paired'] = round(
                j['E20b_minus_e20a_fresh']['minus_e20a'], 4)
        if 'composed_wiring_E20b' in j:
            t = j['composed_wiring_E20b']['tables']
            row['plain'] = t['plain']['spearman_all']
            row['cov_composed_quantized'] = t['cov_composed']['spearman_all']
        cs = j.get('code_stats_E20b', {})
        if cs:
            row['content_bits_per_token'] = \
                cs['content_bits_per_token_sum_joint_entropy']
            row['dead_code_fraction'] = cs['dead_code_fraction_overall']
            row['attn_final_resid_rms_fraction_mean'] = \
                cs['mean_final_residual_rms_fraction_attention']
        summary['E20b'] = row
        if dce is not None:
            summary['prediction_i_verdict'] = (
                f"{'CONFIRMED' if dce < 0.07 else 'REFUTED'}: CE cost "
                f"{dce} vs E19a against the registered < +0.07 threshold "
                f"(E20a was +{E20A_COST})")
        if cs:
            ma = cs['mean_final_residual_rms_fraction_attention']
            summary['prediction_ii_verdict'] = (
                f"{'CONFIRMED' if ma <= 0.30 else 'REFUTED'}: mean "
                f"attention final-residual RMS fraction {ma} vs the "
                f"registered <= 0.30 (MLP-level) threshold")
        if 'composed_wiring_E20b' in j:
            cv = j['composed_wiring_E20b']['tables']['cov_composed'][
                'spearman_all']
            summary['prediction_iii_verdict'] = (
                f"{'CONFIRMED' if cv >= 0.85 else 'REFUTED'}: "
                f"covariance-composed Spearman {cv} vs the registered "
                f">= 0.85 threshold (E20a: 0.8866)")
    E.merge(JP, 'summary_E20b', summary)
    print(json.dumps({'summary_E20b': summary}, indent=2), flush=True)


def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') \
                and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


if __name__ == '__main__':
    E.setup()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c

    # ---- hard-gate controls ----
    control_bypass(s_c)
    control_capacity(s_c)
    control_toy_k4()
    control_k_wiring(s_c)
    if not E.SMOKE:
        control_cov_pipeline(s_c)

    # ---- registered predictions (before training) ----
    if 'E20b_prediction' not in E.loadj(JP):
        E.merge(JP, 'E20b_prediction', {
            'registered_before_training': True,
            'design': 'E20a codebook slots with per-slot pursuit depth '
                      'from the E20 error decomposition: k=4 on attention '
                      'slots (even), k=2 on MLP slots (odd); n=256 '
                      'unit-norm codes per slot, EMA 0.99, commitment '
                      '0.25, dead-reinit 200, straight-through; E19a base '
                      '(E15c 24x15, lasso 1e-4, readable recipe)',
            'i_ce_cost': 'paired fresh CE minus the continuous E19a parent '
                         f'drops below +0.07 (E20a: +{E20A_COST})',
            'ii_attention_residuals': 'mean over attention slots of the '
                                      'final pursuit residual RMS fraction '
                                      '<= 0.30 (~MLP level; E20a mid/late '
                                      'attention 0.45-0.46)',
            'iii_readability': 'covariance-composed wiring Spearman (all '
                               'pairs, quantized content) >= 0.85 (E20a: '
                               '0.8866; parent E19a continuous: 0.8259)',
            'budget_note': 'content bits/token is NOT matched by '
                           'construction: E20a measured 342.101; the '
                           'attention 4-tuples raise the identity budget '
                           '-- documented in code_stats_E20b as '
                           'budget_delta_vs_e20a_bits'})

    # ---- train ----
    E.train_arm('qk_e20b_a', JP, 'E20b', lambda: make_e20b(s=s_c), GC20B,
                lr=E7R.muon_lr(), trainer=trainer_b,
                extra={'optimizer': 'muon', 'slot': s_c,
                       'parent': 'E19a (qk_e19_a); sibling qk_e20_a (k=2)',
                       'design': 'variable-k codebook: attention slots k=4, '
                                 'MLP slots k=2, n=256',
                       'logging': 'qk_e20b_codebook_snapshots.npz + '
                                  'qk_e20b_deadcode_events.jsonl'})
    E.oldheld_record('qk_e20b_a', lambda: make_e20b(s=s_c), JP,
                     'E20b_oldheld')
    E.paired_fresh('qk_e20b_a', JP, 'E20b')
    pair_extra('qk_e20b_a', 'E20b', (('qk_e19_a', 'e19a'),
                                     ('qk_e15_c', 'e15c'),
                                     ('qk_e9_a', 'e9a'),
                                     ('qk_e20_a', 'e20a')))
    E20R.save_seq_heldloss('qk_e20b_a')

    # ---- measurement ----
    code_stats_b(s_c)
    probe_e20b(s_c)
    summarize_b()
    print('e20b vark run done', flush=True)
