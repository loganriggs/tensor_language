"""E17 COMPOSED-WIRING DIAGNOSTIC (checkpoint-only; no training).

Logan's question: the wiring table scores edge (reader li, writer stream si) by
the Frobenius norm of the reader's 11 slot columns alone. The per-slot RMSNorm
fixes the magnitude gauge at the slot interface but NOT directional alignment
inside the 11-dim slot -- a reader can carry large-norm columns orthogonal to
the few directions the writer actually populates. Does composing the reader
columns with the writer improve agreement with causal ablation?

On the readable-recipe checkpoint qk_e9_a.pt (E9a = per-slot RMSNorm + Muon +
in-loss lasso 3e-5, width 264, slot dim 11), three edge-strength tables are
scored against the SAME causal mean-ablation consumption vector (the stored
light_probe_E9a consumption_matrix in qk_e9.json, produced by
qk_e_common.light_probe -> R2.base_and_means/consumption_graph on the old cooc
held rows [:96]):

  1. PLAIN        sqrt(sum_mats ||M[:, slot]||_F^2)          (current table)
  2. DEC-COMPOSED sqrt(sum_mats ||M[:, slot] @ What_k||_F^2), What_k = writer
                  k's decoder rows into its slot (c_proj.weight[slot,:] for
                  attention writes, Down.weight[slot,:] for MLP writes; the
                  wmask zeroes every other output row, so these rows ARE the
                  full write projection), normalized to unit Frobenius norm
                  because the per-slot RMSNorm makes the interface invariant
                  to the writer's overall scale.
  3. COV-COMPOSED sqrt(sum_mats ||M[:, slot] @ C_k^{1/2}||_F^2), C_k = the
                  centered empirical covariance of slot k's POST-NORM content
                  rms_norm(e_slot + write_k_slot, (11,)) over N_COV fresh held
                  sequences (centered = matches mean-ablation semantics: the
                  causal probe replaces content with its mean, so variance
                  around the mean is the causally-relevant signal).
     variant 3b   same, but readout rows (li = 12) use the covariance of the
                  GLOBALLY-normed final entry's slot coordinates -- the readout
                  interface really is the global pre-readout norm, not the
                  per-slot norm.

All three reduce to G[li,k] = sum_mats M[:,slot]^T M[:,slot] (11x11):
  plain = sqrt(tr G), dec = sqrt(tr(G What What^T)), cov = sqrt(tr(G C)),
so all tables share EXACTLY the aggregation of qk_e_common.weight_support
(concatenated 7 read matrices per block consumer; tied wte for the readout;
identical visibility skip rule).

POSITIVE CONTROL (gate): plain derived from G must equal E.weight_support(model)
to 1e-6 relative, and its Spearman against the stored causal vector must match
the stored wiring_spearman_all for E9a (0.7711) to ~1e-3. Do not trust the
composed numbers if this gate fails.

GPU etiquette: weight-only math on CPU; the single activations pass (N_COV
sequences, micro-batch 4, no_grad) waits for >= 2500 MiB free VRAM (up to 2 h)
and falls back to CPU if the GPU never frees. Results -> qk_e17.json.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import subprocess
import time

import numpy as np

# Neutralize the import-time BLOCKING gpu_guard (qk_tokenline_probe calls
# Q.gpu_guard() at module import; with a training job holding the card it
# sleeps 15 min then RAISES). This is a checkpoint-only CPU-side diagnostic --
# same rationale as the non-blocking chain-guard fix in qk_e_common.setup().
import qk_tokenline_train as _Q0
_Q0.gpu_guard = lambda *a, **k: None

import qk_e_common as E
from qk_e_common import Q, W, C, DEPTH, F, torch
import qk_e7_evenout_run as E7R

# Minimal width patch (checkpoint-only diagnostic: no training corpus load,
# no preflight -- just the 264/6/44/slot-11 geometry the factories need).
if not E.SMOKE:
    W.patch_width()
E.SUB = Q.D // E.NGROUP
E.WIDTH = Q.D
E.DEV = 'cpu'                       # load + weight math on CPU (GPU etiquette)
SUB, NG = E.SUB, E.NGROUP
D = Q.D
JP = E.jpath('qk_e17.json')
STEM = 'qk_e9_a'
N_COV = 300                         # fresh held sequences for the covariances
GATE_TOL = 1e-3

import qk_deeproute_train_2 as R2   # spearman + stream_name (probe conventions)


def reader_mats(model, li):
    """EXACTLY weight_support's matrix list for consumer li."""
    return ([model.wte.weight] if li == DEPTH else
            [getattr(model.h[li], nm).weight for nm in E.READ_NAMES
             if getattr(model.h[li], nm, None) is not None])


def gram_table(model):
    """G[li][k] = sum over the reader's matrices of M[:,slot_k]^T M[:,slot_k]
    (11x11 float64), over exactly the (li, si=k+1) pairs weight_support keeps."""
    G = {}
    for li in range(DEPTH + 1):
        mats = reader_mats(model, li)
        row = {}
        for k in range(NG):
            si = k + 1
            if si not in model.vis[li] or si > 2 * min(li, DEPTH):
                continue
            dims = slice(SUB * k, SUB * (k + 1))
            g = torch.zeros(SUB, SUB, dtype=torch.float64)
            for M in mats:
                A = M[:, dims].detach().double()
                g += A.t() @ A
            row[si] = g
        G[li] = row
    return G


def writer_outer(model):
    """What_k What_k^T (11x11) per stream k: the writer's decoder rows into its
    slot, unit-Frobenius-normalized (per-slot RMSNorm interface is invariant to
    the writer's overall scale)."""
    outs = {}
    for k in range(NG):
        j, is_attn = k // 2, (k % 2 == 0)
        blk = model.h[j]
        Wd = (blk.c_proj.weight if is_attn else blk.Down.weight)
        Rows = Wd[SUB * k: SUB * (k + 1), :].detach().double()
        fn = float(Rows.norm())
        outs[k] = (Rows @ Rows.t()) / (fn ** 2) if fn > 0 else \
            torch.zeros(SUB, SUB, dtype=torch.float64)
    return outs


def wait_gpu(min_free=2500, max_s=7200, poll=60):
    t0 = time.time()
    while time.time() - t0 < max_s:
        try:
            free = int(subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits']).decode().split('\n')[0])
        except Exception:
            return False
        if free >= min_free:
            print(f"gpu: {free} MiB free -- using cuda", flush=True)
            return True
        print(f"gpu: only {free} MiB free -- waiting ({int(time.time()-t0)}s)",
              flush=True)
        time.sleep(poll)
    return False


@torch.no_grad()
def slot_covariances(model, rows, dev):
    """Centered covariance of each slot's post-norm content (per-slot RMSNorm
    of e_slot + write_k_slot) plus, for the readout interface, of the GLOBALLY
    normed final entry's slot coordinates. One forward per micro-batch."""
    model = model.to(dev).eval().float()
    n = 0
    s = torch.zeros(NG, SUB, dtype=torch.float64, device=dev)
    ss = torch.zeros(NG, SUB, SUB, dtype=torch.float64, device=dev)
    s_ro = torch.zeros(NG, SUB, dtype=torch.float64, device=dev)
    ss_ro = torch.zeros(NG, SUB, SUB, dtype=torch.float64, device=dev)
    t0 = time.time()
    for i in range(0, len(rows), 4):
        b = rows[i:i + 4].to(dev)
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b[:, :Q.T], collect=col)
        e = F.rms_norm(model.wte(b[:, :Q.T]), (D,))
        writes = []
        for j in range(DEPTH):
            writes.append(col['attn_write'][j])
            writes.append(col['mlp_write'][j])
        xf = e + sum(writes)                       # final entry (full visibility)
        gro = F.rms_norm(xf, (D,))                 # readout: GLOBAL norm
        for k in range(NG):
            dims = slice(SUB * k, SUB * (k + 1))
            cont = e[..., dims] + writes[k][..., dims]
            pn = F.rms_norm(cont, (SUB,)).reshape(-1, SUB).double()
            s[k] += pn.sum(0)
            ss[k] += pn.t() @ pn
            gs = gro[..., dims].reshape(-1, SUB).double()
            s_ro[k] += gs.sum(0)
            ss_ro[k] += gs.t() @ gs
        n += b.shape[0] * Q.T
        if (i // 4) % 25 == 0:
            print(f"  cov pass: {i + b.shape[0]}/{len(rows)} seqs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.to('cpu')
    if dev == 'cuda':
        torch.cuda.empty_cache()

    def finish(s_, ss_):
        mu = (s_ / n)
        Cv = ss_ / n - torch.einsum('ki,kj->kij', mu, mu)
        return Cv.cpu()
    return finish(s, ss), finish(s_ro, ss_ro), n


def score_vec(G, wp, key_mats=None):
    """sqrt(tr(G[li][si] @ K)) over wp; K=None -> plain sqrt(tr G)."""
    out = []
    for li, si in wp:
        g = G[li][si]
        if key_mats is None:
            out.append(float(torch.diagonal(g).sum()) ** 0.5)
        else:
            K = key_mats[si - 1]
            out.append(max(float((g * K.t()).sum()), 0.0) ** 0.5)
    return out


def agreement(sup, cau, eff_idx):
    top_s = set(np.argsort(sup)[::-1][:10].tolist())
    top_c = set(np.argsort(cau)[::-1][:10].tolist())
    return {'spearman_all': round(R2.spearman(sup, cau), 4),
            'spearman_effectual': round(R2.spearman(
                [sup[k] for k in eff_idx], [cau[k] for k in eff_idx]), 4),
            'top10_precision': len(top_s & top_c) / 10.0}


def main():
    done = E.loadj(JP)
    if 'tables' in done:
        print('qk_e17.json already complete -- skip', flush=True)
        return

    # ---- causal vector: the SAME consumption graph the stored probe computed
    e9 = E.loadj(E.jpath('qk_e9.json'))
    lp = e9['light_probe_E9a']
    dce = {int(li): {int(si): v for si, v in row.items()}
           for li, row in lp['consumption_matrix'].items()}

    print('loading qk_e9_a.pt (cpu) ...', flush=True)
    model, ck = E.load_arm(STEM, E7R.make_e7m1)

    # ---- table geometry + gram tensors (all weight math on CPU)
    G = gram_table(model)
    wp = [(li, si) for li in G for si in G[li]]
    cau = [dce[li][si] for li, si in wp]
    eff_idx = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]

    # ---- POSITIVE CONTROL / gate: plain-from-G == weight_support, and its
    # Spearman vs the stored causal vector == stored wiring_spearman_all
    plain = score_vec(G, wp)
    ws = E.weight_support(model)
    ws_vec = [ws[li][si] for li, si in wp]
    rel = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(plain, ws_vec))
    ctl = agreement(plain, cau, eff_idx)
    control = {
        'n_pairs': len(wp), 'stored_n_pairs': lp['wiring_n_pairs'],
        'n_effectual': len(eff_idx),
        'stored_n_effectual': lp['wiring_n_effectual'],
        'plain_vs_weight_support_max_rel_diff': float(f'{rel:.3e}'),
        'plain_spearman_all': ctl['spearman_all'],
        'stored_wiring_spearman_all': lp['wiring_spearman_all'],
        'abs_diff': round(abs(ctl['spearman_all']
                              - lp['wiring_spearman_all']), 6),
        'plain_spearman_effectual': ctl['spearman_effectual'],
        'stored_wiring_spearman_effectual': lp['wiring_spearman_effectual'],
        'plain_top10_precision': ctl['top10_precision'],
        'stored_wiring_top10_precision': lp['wiring_top10_precision'],
        'causal_vector': 'qk_e9.json light_probe_E9a consumption_matrix '
                         '(mean-ablation, old cooc held [:96], the exact '
                         'stored probe output)'}
    control['pass'] = bool(
        len(wp) == lp['wiring_n_pairs'] and rel < 1e-6
        and control['abs_diff'] <= GATE_TOL)
    print(f"CONTROL: plain Spearman {ctl['spearman_all']} vs stored "
          f"{lp['wiring_spearman_all']} (diff {control['abs_diff']}), "
          f"weight_support rel diff {rel:.1e} -> "
          f"{'PASS' if control['pass'] else 'FAIL'}", flush=True)
    E.merge(JP, 'control', control)
    if not control['pass']:
        raise SystemExit('control gate FAILED -- not computing composed tables')

    # ---- table 2: decoder-composed (CPU, weights only)
    Wout = writer_outer(model)
    dec = score_vec(G, wp, Wout)

    # ---- table 3: covariance-composed (one activations pass on fresh held)
    held = np.load(E.HELD_PATH)[33000:33000 + N_COV].astype(np.int64)
    rows = torch.from_numpy(held)
    dev = 'cuda' if (not E.SMOKE and wait_gpu()) else 'cpu'
    print(f"cov pass on {dev}: {N_COV} fresh held seqs "
          f"(fresh34k rows [33000:{33000 + N_COV}])", flush=True)
    Cslot, Cro, n_samp = slot_covariances(model, rows, dev)
    cov = score_vec(G, wp, {k: Cslot[k] for k in range(NG)})
    cov_ro = [max(float((G[li][si] * (Cro[si - 1] if li == DEPTH
                                      else Cslot[si - 1]).t()).sum()), 0.0) ** 0.5
              for li, si in wp]

    # ---- agreement + pairwise
    tables = {'plain': agreement(plain, cau, eff_idx),
              'decoder_composed': agreement(dec, cau, eff_idx),
              'cov_composed': agreement(cov, cau, eff_idx),
              'cov_composed_readout_globalnorm':
                  agreement(cov_ro, cau, eff_idx)}
    vecs = {'plain': plain, 'decoder_composed': dec, 'cov_composed': cov,
            'cov_composed_readout_globalnorm': cov_ro}
    names = list(vecs)
    pairwise = {f'{a}__vs__{b}': round(R2.spearman(vecs[a], vecs[b]), 4)
                for i, a in enumerate(names) for b in names[i + 1:]}

    # ---- concrete examples: edges whose rank moves most (plain -> cov)
    def ranks(v):
        return np.argsort(np.argsort(-np.asarray(v)))
    rp, rc, rcau = ranks(plain), ranks(cov), ranks(cau)
    moved = np.argsort(-np.abs(rp - rc))[:12]
    examples = []
    for k in moved[:8]:
        li, si = wp[k]
        examples.append({
            'consumer': 'readout' if li == DEPTH else f'block{li}',
            'source': R2.stream_name(si),
            'dce': round(cau[k], 5),
            'rank_causal': int(rcau[k]), 'rank_plain': int(rp[k]),
            'rank_cov_composed': int(rc[k]),
            'plain': round(plain[k], 3), 'cov_composed': round(cov[k], 4)})

    out = {
        'checkpoint': f'{STEM}.pt', 'width': D, 'slot_dim': SUB,
        'n_pairs': len(wp), 'n_effectual': len(eff_idx),
        'effectual_threshold': C.EFFECTUAL,
        'cov_pass': {'device': dev, 'n_seq': N_COV, 'n_samples': n_samp,
                     'rows': 'fresh34k[33000:33300] (standard fresh held '
                             'prefix)', 'centered': True},
        'decoder_normalization': 'unit Frobenius per writer (per-slot RMSNorm '
                                 'interface is writer-scale invariant)',
        'tables': tables, 'pairwise_spearman': pairwise,
        'largest_rank_moves_plain_to_cov': examples}
    for k, v in out.items():
        E.merge(JP, k, v)
    print(json.dumps({'control_pass': control['pass'], 'tables': tables,
                      'pairwise': pairwise}, indent=2), flush=True)


if __name__ == '__main__':
    main()
