"""E18 PROBE UPGRADES (checkpoint-only diagnostics + data exports; NO training).

Four jobs, results -> qk_e18.json (+ qk_e9_a_heldloss.npy):

1. SCALE REQUEST -- per-token held-loss export for the readable recipe
   qk_e9_a.pt on the standard fresh held rows (fresh34k[33000:34500], 1500
   seqs x 512 tokens), matching the existing *_heldloss.npy convention
   exactly: flat (768000,) float32 from Q.eval_held(per_token=True) under
   bf16 autocast at batch 16 (the train_muon save path).

2. SCALE REQUEST -- neck-information reference on the recipe: the E12 funnel
   neck_info_probe (qk_e12_funnel_run.neck_info_probe, reused verbatim,
   funnel=False branch) on qk_e9_a.pt: ridge top-1 token recovery from
   (block-0 outputs, stream entries at blocks 3/7/11) on fresh held rows.
   Recorded under 'neck_info_reference_E9a'.

3. VARIABLE-SLOT-DIM WIRING PROBE -- the light probe's two halves
   (weight-support wiring + mean-ablation consumption graph) generalized to
   arbitrary per-slot dims (the qk_e_common versions assume slot dim
   Q.D // 24 = 11), applied to the bandwidth-reinvestment arm E15c
   (qk_e15_c.pt: 24 slots x 15 dims, stream 360, compute 264).
   POSITIVE CONTROL (hard gate): the generalized weight support at uniform
   dims [11]*24 on qk_e9_a.pt must (a) match E.weight_support to 1e-6
   relative and (b) reproduce the stored wiring_spearman_all 0.7711 exactly
   against the stored light_probe_E9a consumption matrix.

4. COVARIANCE-COMPOSED RE-SCORING (qk_e17_composed_wiring method, functions
   reused where dim-agnostic, gram/covariance machinery generalized to
   variable slot dims and to E16's PER-CONSUMER remnant stream 0):
   cov-composed wiring Spearman for E16a, E16b (stored causal vectors from
   qk_e16.json) and E15c (job 3's fresh causal vector), reported next to
   plain. E16 slot content at a consumer's interface = the module's write
   plus, where the remnant overlaps a claimed visible slot (E16b only:
   consumer 11 x slots 20-21, readout x slots 20-23), that consumer's
   remnant restricted to the slot -- covariance computed PER (consumer,
   slot) on the overlap pairs, per slot elsewhere. E15c/E9a slot content =
   embedding slot + write (bottom injection), exactly e17.
   POSITIVE CONTROL (hard gate): the generalized cov-composed pipeline at
   uniform 11 on qk_e9_a.pt must reproduce qk_e17.json's stored
   cov_composed spearman_all 0.8575 to 1e-3.

Both gates run FIRST; nothing downstream is computed or reported off an
ungated path. GPU etiquette: wait politely for >= 2500 MiB free before the
first GPU touch (the diagnostics fall back to CPU if it never frees).
Idempotent on qk_e18.json keys."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time

import numpy as np

# Neutralize the import-time BLOCKING gpu_guard (qk_tokenline_probe calls
# Q.gpu_guard() at module import) -- checkpoint-only diagnostics, same
# rationale as qk_e17_composed_wiring.
import qk_tokenline_train as _Q0
_Q0.gpu_guard = lambda *a, **k: None

import qk_e_common as E
from qk_e_common import Q, C, DEPTH, F, torch
import qk_deeproute_train_2 as R2
import qk_e7_evenout_run as E7R
# e17: reuse the dim-agnostic pieces (agreement, wait_gpu). NOTE its import
# sets E.DEV='cpu' for its own weight-only use -- restored to cuda below.
import qk_e17_composed_wiring as E17
import qk_e12_funnel_run as E12R          # neck_info_probe (reused verbatim)
import qk_e15_reinvest_run as E15R        # make_e15c
import qk_e16_shrinkemb_run as E16R       # make_e16a / make_e16b

JP = E.jpath('qk_e18.json')
NG = E.NGROUP                              # 24 slot groups
ABL_N = R2.ABL_N                           # 96 old cooc held seqs (causal)
N_COV = 300                                # fresh held seqs for covariances
GATE_TOL = 1e-3
DEV = 'cuda'                               # set for real in main() after guard


def offsets(dims):
    o = [0]
    for d in dims:
        o.append(o[-1] + d)
    return o


def reader_mats(model, li):
    """EXACTLY weight_support's matrix list for consumer li."""
    return ([model.wte.weight] if li == DEPTH else
            [getattr(model.h[li], nm).weight for nm in E.READ_NAMES
             if getattr(model.h[li], nm, None) is not None])


def wpairs(model, dims):
    """The (li, si) pairs weight_support keeps (visibility skip rule)."""
    out = []
    for li in range(DEPTH + 1):
        for k in range(NG):
            si = k + 1
            if si not in model.vis[li] or si > 2 * min(li, DEPTH):
                continue
            out.append((li, si))
    return out


def gen_gram_table(model, dims):
    """G[(li,si)] = sum over the reader's matrices of M[:,slot]^T M[:,slot]
    (dims[k] x dims[k] float64) -- variable per-slot dims."""
    offs = offsets(dims)
    G = {}
    for li in range(DEPTH + 1):
        mats = [M.detach().double().cpu() for M in reader_mats(model, li)]
        for k in range(NG):
            si = k + 1
            if si not in model.vis[li] or si > 2 * min(li, DEPTH):
                continue
            sl = slice(offs[k], offs[k + 1])
            g = torch.zeros(dims[k], dims[k], dtype=torch.float64)
            for M in mats:
                A = M[:, sl]
                g += A.t() @ A
            G[(li, si)] = g
    return G


def score(G, wp, keyfn=None):
    """plain = sqrt(tr G); composed = sqrt(tr(G K)) with K = keyfn(li, si)."""
    out = []
    for li, si in wp:
        g = G[(li, si)]
        if keyfn is None:
            out.append(float(torch.diagonal(g).sum()) ** 0.5)
        else:
            K = keyfn(li, si)
            out.append(max(float((g * K.t()).sum()), 0.0) ** 0.5)
    return out


# ---------------- generalized causal half (mean-ablation, old cooc held) ----------------
@torch.no_grad()
def gen_base_and_means(model, Ws):
    """fp32 pass over OLD cooc held [:ABL_N]: base CE + per-stream means
    (R2.base_and_means semantics at arbitrary stream width Ws)."""
    model.eval().float()
    sums = torch.zeros(1 + 2 * DEPTH, Ws, device=E.OLD_HELD.device,
                       dtype=torch.float64)
    n, pts = 0, []
    for i in range(0, ABL_N, 8):
        b = E.OLD_HELD[i:i + 8]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        logits = model(b[:, :Q.T], collect=col)
        ce = F.cross_entropy(logits.reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        pts.append(ce.cpu())
        e = F.rms_norm(model.wte(b[:, :Q.T]), (Ws,))
        sums[0] += e.double().sum((0, 1))
        for l in range(DEPTH):
            sums[1 + 2 * l] += col['attn_write'][l].double().sum((0, 1))
            sums[2 + 2 * l] += col['mlp_write'][l].double().sum((0, 1))
        n += b.shape[0] * Q.T
    return float(torch.cat(pts).mean()), (sums / n).float()


@torch.no_grad()
def gen_ce_with(model, li, si, mv, Ws):
    tot, n = 0.0, 0
    for i in range(0, ABL_N, 8):
        b = E.OLD_HELD[i:i + 8]
        se = {li: {si: mv[None, None, :].expand(b.shape[0], Q.T, Ws)}}
        logits = model(b[:, :Q.T], sub_entry=se)
        ce = F.cross_entropy(logits.reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        tot += ce.sum().item()
        n += ce.numel()
    return tot / n


@torch.no_grad()
def gen_consumption(model, Ws):
    base, means = gen_base_and_means(model, Ws)
    dce = {}
    t0 = time.time()
    for li in range(DEPTH + 1):
        dce[li] = {}
        for si in model.vis[li]:
            dce[li][si] = gen_ce_with(model, li, si, means[si], Ws) - base
        top = sorted(dce[li], key=lambda s: -dce[li][s])[:4]
        print(f"  consumption row {li}: " + ", ".join(
            f"{R2.stream_name(si)} {dce[li][si]:+.4f}" for si in top) +
            f"  ({time.time() - t0:.0f}s)", flush=True)
    return base, dce


# ---------------- generalized covariance half ----------------
def overlap_pairs(model, dims):
    """(li, k) pairs where the per-consumer remnant overlaps a VISIBLE
    claimed slot (E16b's disjointness exception; empty for E16a)."""
    if not hasattr(model, 'rem_dims'):
        return []
    Ws = model.wte.weight.shape[1]
    offs = offsets(dims)
    out = []
    for li in range(1, DEPTH + 1):
        r = model.rem_dims[li]
        for k in range(NG):
            si = k + 1
            if si not in model.vis[li] or si > 2 * min(li, DEPTH):
                continue
            if r > 0 and offs[k] >= Ws - r:
                out.append((li, k))
    return out


@torch.no_grad()
def gen_slot_covariances(model, rows, dev, dims, remnant=False):
    """Centered covariance of each slot's POST-NORM content at the consumer
    interface, generalized:
      remnant=False (E9a/E15c): content_k = e_slot + write_k (bottom
        embedding injection; consumer-independent) -- exactly e17.
      remnant=True (E16): content_k = write_k, plus PER-(consumer, slot)
        covariances where the remnant overlaps a visible claimed slot; the
        readout-globalnorm variant's final entry includes the readout
        remnant (E16b) or nothing (E16a).
    Returns (Cbase[k], Cover[(li,k)], Cro[k], n_samples)."""
    model = model.to(dev).eval().float()
    offs = offsets(dims)
    ov = overlap_pairs(model, dims) if remnant else []
    n = 0
    s_b = [torch.zeros(dims[k], dtype=torch.float64, device=dev)
           for k in range(NG)]
    ss_b = [torch.zeros(dims[k], dims[k], dtype=torch.float64, device=dev)
            for k in range(NG)]
    s_o = {p: torch.zeros(dims[p[1]], dtype=torch.float64, device=dev)
           for p in ov}
    ss_o = {p: torch.zeros(dims[p[1]], dims[p[1]], dtype=torch.float64,
                           device=dev) for p in ov}
    s_r = [torch.zeros(dims[k], dtype=torch.float64, device=dev)
           for k in range(NG)]
    ss_r = [torch.zeros(dims[k], dims[k], dtype=torch.float64, device=dev)
            for k in range(NG)]
    Ws = model.wte.weight.shape[1]
    t0 = time.time()
    for i in range(0, len(rows), 4):
        b = rows[i:i + 4].to(dev)
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b[:, :Q.T], collect=col)
        e = F.rms_norm(model.wte(b[:, :Q.T]), (Ws,))
        writes = []
        for j in range(DEPTH):
            writes.append(col['attn_write'][j])
            writes.append(col['mlp_write'][j])
        if remnant:
            rem = model.remnants(e)
            xf = sum(writes)
            if rem[DEPTH] is not None:
                xf = xf + rem[DEPTH]
        else:
            rem = None
            xf = e + sum(writes)
        gro = F.rms_norm(xf, (Ws,))                # readout: GLOBAL norm
        for k in range(NG):
            sl = slice(offs[k], offs[k + 1])
            S = dims[k]
            cont = writes[k][..., sl] if remnant else \
                e[..., sl] + writes[k][..., sl]
            pn = F.rms_norm(cont, (S,)).reshape(-1, S).double()
            s_b[k] += pn.sum(0)
            ss_b[k] += pn.t() @ pn
            gs = gro[..., sl].reshape(-1, S).double()
            s_r[k] += gs.sum(0)
            ss_r[k] += gs.t() @ gs
        for (li, k) in ov:
            sl = slice(offs[k], offs[k + 1])
            S = dims[k]
            cont = writes[k][..., sl] + rem[li][..., sl]
            pn = F.rms_norm(cont, (S,)).reshape(-1, S).double()
            s_o[(li, k)] += pn.sum(0)
            ss_o[(li, k)] += pn.t() @ pn
        n += b.shape[0] * Q.T
        if (i // 4) % 25 == 0:
            print(f"  cov pass: {i + b.shape[0]}/{len(rows)} seqs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if dev == 'cuda':
        torch.cuda.empty_cache()

    def fin(s_, ss_):
        mu = s_ / n
        return (ss_ / n - torch.outer(mu, mu)).cpu()
    return ([fin(s_b[k], ss_b[k]) for k in range(NG)],
            {p: fin(s_o[p], ss_o[p]) for p in ov},
            [fin(s_r[k], ss_r[k]) for k in range(NG)], n)


def composed_tables(model, dims, cau_vec, wp, dev, remnant=False,
                    plain_vec=None):
    """plain / cov_composed / cov_composed_readout_globalnorm agreement
    tables against cau_vec (+ the raw vectors)."""
    G = gen_gram_table(model, dims)
    plain = score(G, wp) if plain_vec is None else plain_vec
    held = np.load(E.HELD_PATH)[33000:33000 + N_COV].astype(np.int64)
    Cb, Co, Cr, n_samp = gen_slot_covariances(
        model, torch.from_numpy(held), dev, dims, remnant=remnant)

    def key_cov(li, si):
        return Co.get((li, si - 1), Cb[si - 1])

    def key_cov_ro(li, si):
        if li == DEPTH:
            return Cr[si - 1]
        return key_cov(li, si)
    cov = score(G, wp, key_cov)
    cov_ro = score(G, wp, key_cov_ro)
    eff_idx = [k for k in range(len(wp)) if cau_vec[k] > C.EFFECTUAL]
    tables = {'plain': E17.agreement(plain, cau_vec, eff_idx),
              'cov_composed': E17.agreement(cov, cau_vec, eff_idx),
              'cov_composed_readout_globalnorm':
                  E17.agreement(cov_ro, cau_vec, eff_idx)}
    meta = {'n_pairs': len(wp), 'n_effectual': len(eff_idx),
            'overlap_pairs_per_consumer_cov':
                [[li, R2.stream_name(k + 1)] for li, k in sorted(Co)],
            'cov_pass': {'device': dev, 'n_seq': N_COV, 'n_samples': n_samp,
                         'rows': f'fresh34k[33000:{33000 + N_COV}]',
                         'centered': True}}
    return tables, meta, {'plain': plain, 'cov': cov, 'cov_ro': cov_ro}


def stored_cau(lp, wp):
    dce = {int(li): {int(si): v for si, v in row.items()}
           for li, row in lp['consumption_matrix'].items()}
    return [dce[li][si] for li, si in wp]


def main():
    global DEV
    done = E.loadj(JP)
    DEV = 'cuda' if E17.wait_gpu(min_free=2500) else 'cpu'
    E.DEV = DEV
    print(f"E18 on {DEV}", flush=True)

    e9lp = E.loadj(E.jpath('qk_e9.json'))['light_probe_E9a']
    e16j = E.loadj(E.jpath('qk_e16.json'))
    e17j = E.loadj(E.jpath('qk_e17.json'))

    print('loading qk_e9_a.pt ...', flush=True)
    m9, _ = E.load_arm('qk_e9_a', E7R.make_e7m1)
    dims11 = [E.SUB] * NG                        # uniform 11
    wp9 = wpairs(m9, dims11)
    cau9 = stored_cau(e9lp, wp9)
    eff9 = [k for k in range(len(wp9)) if cau9[k] > C.EFFECTUAL]

    # ==== GATE 1 (job 3 control): generalized probe reproduces uniform-11 ====
    G9 = gen_gram_table(m9, dims11)
    plain9 = score(G9, wp9)
    ws = E.weight_support(m9)
    ws_vec = [ws[li][si] for li, si in wp9]
    rel = max(abs(a - b) / max(abs(b), 1e-12)
              for a, b in zip(plain9, ws_vec))
    agr9 = E17.agreement(plain9, cau9, eff9)
    gate1 = {'n_pairs': len(wp9), 'stored_n_pairs': e9lp['wiring_n_pairs'],
             'gen_vs_weight_support_max_rel_diff': float(f'{rel:.3e}'),
             'gen_spearman_all': agr9['spearman_all'],
             'stored_wiring_spearman_all': e9lp['wiring_spearman_all'],
             'abs_diff': round(abs(agr9['spearman_all']
                                   - e9lp['wiring_spearman_all']), 6),
             'gen_spearman_effectual': agr9['spearman_effectual'],
             'stored_spearman_effectual': e9lp['wiring_spearman_effectual'],
             'gen_top10_precision': agr9['top10_precision'],
             'stored_top10_precision': e9lp['wiring_top10_precision']}
    gate1['pass'] = bool(len(wp9) == e9lp['wiring_n_pairs'] and rel < 1e-6
                         and gate1['abs_diff'] == 0.0)
    print(f"GATE 1 (uniform-11 generalization): Spearman "
          f"{agr9['spearman_all']} vs stored {e9lp['wiring_spearman_all']}, "
          f"weight_support rel {rel:.1e} -> "
          f"{'PASS' if gate1['pass'] else 'FAIL'}", flush=True)
    E.merge(JP, 'gate1_uniform11_weight_support', gate1)
    if not gate1['pass']:
        raise SystemExit('GATE 1 FAILED -- fix before any downstream job')

    # ==== GATE 2 (job 4 control): cov-composed reproduces qk_e17 0.8575 ====
    if 'gate2_cov_composed_E9a' not in done or \
            not done['gate2_cov_composed_E9a'].get('pass'):
        t9, meta9, _ = composed_tables(m9, dims11, cau9, wp9, DEV,
                                       remnant=False, plain_vec=plain9)
        stored = e17j['tables']['cov_composed']['spearman_all']
        gate2 = {'cov_composed_spearman_all':
                     t9['cov_composed']['spearman_all'],
                 'stored_e17_cov_composed_spearman_all': stored,
                 'abs_diff': round(abs(t9['cov_composed']['spearman_all']
                                       - stored), 6),
                 'cov_composed_readout_globalnorm_spearman_all':
                     t9['cov_composed_readout_globalnorm']['spearman_all'],
                 'stored_e17_readout_globalnorm':
                     e17j['tables']['cov_composed_readout_globalnorm'][
                         'spearman_all'],
                 'tables': t9, 'meta': meta9}
        gate2['pass'] = bool(gate2['abs_diff'] <= GATE_TOL)
        print(f"GATE 2 (cov-composed E9a): {gate2['cov_composed_spearman_all']}"
              f" vs stored {stored} (diff {gate2['abs_diff']}) -> "
              f"{'PASS' if gate2['pass'] else 'FAIL'}", flush=True)
        E.merge(JP, 'gate2_cov_composed_E9a', gate2)
        if not gate2['pass']:
            raise SystemExit('GATE 2 FAILED -- fix before any downstream job')

    # ==== JOB 1: per-token held-loss export for qk_e9_a ====
    if 'heldloss_export_E9a' not in E.loadj(JP):
        held = np.load(E.HELD_PATH)[33000:34500].astype(np.int64)
        Q.HELD = torch.from_numpy(held).to(DEV)
        Q.BATCH = 16                              # train-time eval convention
        m9.to(DEV)
        hce, pt = Q.eval_held(m9, per_token=True)
        m9.eval()
        pt = pt.astype(np.float32)
        ref = f'{E.QK}/qk_e12_L_heldloss.npy'
        rf = np.load(ref)
        assert pt.shape == rf.shape and pt.dtype == rf.dtype, \
            (pt.shape, pt.dtype, rf.shape, rf.dtype)
        prev_path = f'{E.QK}/qk_e9_a_heldloss.npy'
        prev_diff = None
        if os.path.exists(prev_path):
            prev = np.load(prev_path)
            prev_diff = float(np.abs(prev - pt).max())
        np.save(prev_path, pt)
        rec = {'file': 'qk_e9_a_heldloss.npy', 'shape': list(pt.shape),
               'dtype': str(pt.dtype), 'mean_ce_bf16': round(float(hce), 5),
               'rows': 'fresh34k[33000:34500] (1500 seqs x 512 tokens, '
                       'flat, seq-major)',
               'convention': 'Q.eval_held(per_token=True), bf16 autocast, '
                             'batch 16 (train_muon save path)',
               'max_abs_diff_vs_preexisting_file': prev_diff}
        E.merge(JP, 'heldloss_export_E9a', rec)
        print(f"JOB 1: heldloss export mean {hce:.5f}, max diff vs "
              f"pre-existing file {prev_diff}", flush=True)

    # ==== JOB 2: neck_info reference on the recipe ====
    if 'neck_info_reference_E9a' not in E.loadj(JP):
        held = np.load(E.HELD_PATH)[33000:34500].astype(np.int64)
        Q.HELD = torch.from_numpy(held).to(DEV)
        res = E12R.neck_info_probe(m9, funnel=False)
        res['checkpoint'] = 'qk_e9_a.pt'
        res['sites'] = ('neck_block0_outputs = attn0+mlp0 write (the E9a '
                        'quantity the funnel necks compress); stream_blockN '
                        '= full entry at block N')
        res['rows'] = 'fresh34k[33000:33064] (fit 48 / eval 16, E12 probe '\
                      'convention)'
        E.merge(JP, 'neck_info_reference_E9a', res)
        print('JOB 2: neck_info_reference_E9a recorded', flush=True)
    del m9
    torch.cuda.empty_cache()

    # ==== JOB 3: E15c variable-slot-dim wiring probe ====
    j = E.loadj(JP)
    if 'light_probe_E15c_var_dims' not in j:
        print('loading qk_e15_c.pt (24 slots x 15 dims, stream 360) ...',
              flush=True)
        m15, ck15 = E.load_arm('qk_e15_c', lambda: E15R.make_e15c(s=15))
        Ws15 = m15.wte.weight.shape[1]
        dims15 = [m15.s] * NG
        assert Ws15 == 360 and m15.s == 15
        base15, dce15 = gen_consumption(m15, Ws15)
        wp15 = wpairs(m15, dims15)
        G15 = gen_gram_table(m15, dims15)
        sup15 = score(G15, wp15)
        cau15 = [dce15[li][si] for li, si in wp15]
        eff15 = [k for k in range(len(wp15)) if cau15[k] > C.EFFECTUAL]
        agr15 = E17.agreement(sup15, cau15, eff15)
        totals = {}
        for jj in range(DEPTH):
            totals[str(jj)] = round(sum(
                dce15[li].get(si, 0.0) for li in dce15
                for si in (1 + 2 * jj, 2 + 2 * jj) if si in dce15[li]), 5)
        pairs_sorted = sorted([(li, si, dce15[li][si]) for li in dce15
                               for si in dce15[li]], key=lambda p: -p[2])
        rec = {'checkpoint': 'qk_e15_c.pt', 'stream_width': Ws15,
               'slot_dims': dims15, 'compute_width': Q.D,
               'base_ce_fp32_abl_oldheld': round(base15, 5),
               'wiring_n_pairs': len(wp15),
               'wiring_spearman_all': agr15['spearman_all'],
               'wiring_n_effectual': len(eff15),
               'wiring_spearman_effectual': agr15['spearman_effectual'],
               'wiring_top10_precision': agr15['top10_precision'],
               'per_block_total_consumption': totals,
               'dead_blocks_below_0.001':
                   [b for b in totals if totals[b] < 0.001],
               'consumption_top20': [
                   {'consumer': ('readout' if li == DEPTH else f'block{li}'),
                    'source': R2.stream_name(si), 'dce': round(v, 5)}
                   for li, si, v in pairs_sorted[:20]],
               'consumption_matrix': {str(li): {str(si): round(v, 6)
                                                for si, v in dce15[li].items()}
                                      for li in dce15},
               'weight_support_matrix': {
                   str(li): {str(si): round(sup15[i], 3)
                             for i, (l2, si) in enumerate(wp15) if l2 == li}
                   for li in range(DEPTH + 1)},
               'note': 'generalized variable-slot-dim probe (gate 1 passed '
                       'on the uniform-11 recipe); causal = mean-ablation '
                       'on old cooc held [:96], fp32, light-probe '
                       'conventions'}
        E.merge(JP, 'light_probe_E15c_var_dims', rec)
        print(f"JOB 3: E15c wiring Spearman all {agr15['spearman_all']} "
              f"effectual({len(eff15)}) {agr15['spearman_effectual']} "
              f"top10 {agr15['top10_precision']}", flush=True)
        del m15
        torch.cuda.empty_cache()

    # ==== JOB 4: covariance-composed re-scoring ====
    def rescore(key, stem, factory, dims, lp, remnant, stored_spearman):
        if key in E.loadj(JP):
            return
        print(f'loading {stem}.pt ...', flush=True)
        m, _ = E.load_arm(stem, factory)
        wp = wpairs(m, dims)
        cau = stored_cau(lp, wp)
        tables, meta, _ = composed_tables(m, dims, cau, wp, DEV,
                                          remnant=remnant)
        # consistency: plain must reproduce the stored light-probe Spearman
        chk = abs(tables['plain']['spearman_all'] - stored_spearman)
        rec = {'checkpoint': f'{stem}.pt', 'slot_dims_uniform': dims[0],
               'causal_vector': lp.get('note', 'stored light-probe '
                                               'consumption matrix'),
               'stored_plain_spearman_all': stored_spearman,
               'plain_reproduction_abs_diff': round(chk, 6),
               'tables': tables}
        rec.update(meta)
        assert chk <= GATE_TOL, \
            f'{key}: plain does not reproduce stored ({chk})'
        E.merge(JP, key, rec)
        print(f"JOB 4 {key}: plain {tables['plain']['spearman_all']} -> "
              f"cov {tables['cov_composed']['spearman_all']}", flush=True)
        del m
        torch.cuda.empty_cache()

    rescore('composed_wiring_E16a', 'qk_e16_a', E16R.make_e16a, dims11,
            e16j['light_probe_E16a'], True,
            e16j['light_probe_E16a']['wiring_spearman_all'])
    rescore('composed_wiring_E16b', 'qk_e16_b', E16R.make_e16b, dims11,
            e16j['light_probe_E16b'], True,
            e16j['light_probe_E16b']['wiring_spearman_all'])
    lp15 = E.loadj(JP)['light_probe_E15c_var_dims']
    rescore('composed_wiring_E15c', 'qk_e15_c',
            lambda: E15R.make_e15c(s=15), [15] * NG, lp15, False,
            lp15['wiring_spearman_all'])

    # rebuild the summary table from the stored records (robust on resume)
    jj = E.loadj(JP)
    summary = {}
    for arm in ('E16a', 'E16b', 'E15c'):
        t = jj[f'composed_wiring_{arm}']['tables']
        summary[arm] = {
            'plain': t['plain']['spearman_all'],
            'cov_composed': t['cov_composed']['spearman_all'],
            'plain_effectual': t['plain']['spearman_effectual'],
            'cov_effectual': t['cov_composed']['spearman_effectual'],
            'plain_top10': t['plain']['top10_precision'],
            'cov_top10': t['cov_composed']['top10_precision']}
    summary['E9a_reference'] = {
        'plain': e9lp['wiring_spearman_all'],
        'cov_composed': e17j['tables']['cov_composed']['spearman_all'],
        'plain_effectual': e9lp['wiring_spearman_effectual'],
        'cov_effectual': e17j['tables']['cov_composed']['spearman_effectual'],
        'plain_top10': e9lp['wiring_top10_precision'],
        'cov_top10': e17j['tables']['cov_composed']['top10_precision']}
    E.merge(JP, 'summary_plain_vs_cov', summary)
    print(json.dumps({'summary_plain_vs_cov': summary}, indent=2), flush=True)
    print('e18 probe upgrades done', flush=True)


if __name__ == '__main__':
    main()
