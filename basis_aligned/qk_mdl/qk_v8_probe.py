"""V8 interpretability scoring (companion to qk_v8_train.py; machinery reused from
qk_deeproute_train_2.py). Measures, on the trained V8:

1. CAUSAL CONSUMPTION GRAPH - mean-ablate each visible (source stream, consumer)
   pair in that consumer's assembly only, delta CE on 96 held seqs, fp32 (79 pairs
   under the V6 dilation mask: 74 write pairs + 5 embedding pairs).
2. WIRING AGREEMENT (the decisive number) - weight-support prediction (per-slot
   Frobenius of the consumer's seven input matrices; readout: tied wte) vs the
   causal graph: Spearman over all visible write pairs, Spearman over effectual
   pairs (causal delta CE > 0.005 nats), and top-10-edge precision (overlap of the
   top 10 support-ranked and top 10 causally-ranked edges). Plus per-source-block
   total consumption (dead-block check: no block below 0.001 nats total).
3. EXACT-TERM DECOMPOSABILITY at blocks 4, 6, 9 - reassembly gate, terms-to-95%
   energy, causal recovery of the top-k95 truncation vs the mean-entry floor.
4. STANDARD PROBES - token-determined fraction of each mlp write (group-mean R^2 +
   shuffle control) and linear-in-normed-embedding (held variance explained +
   causal recovered fraction vs global-mean floor, full held slice, fp32).

Merges everything into qk_v8.json under 'probe'.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, time
import numpy as np
import torch

import qk_tokenline_train as Q
import qk_v8_train as V8T                 # registers 'V8' in the deeproute tables
import qk_deeproute_train_2 as R2
from qk_deeproute_train import DEPTH

QK = Q.QK
EFFECTUAL = 0.005
TERM_BLOCKS = [4, 6, 9]


@torch.no_grad()
def standard_probes_guarded(model):
    """qk_deeproute_train_2.standard_probes with zero-variance guards: a collapsed
    (constant) module write has tot == 0, which crashed the shared version."""
    import torch.nn.functional as F
    import qk_tokenline_probe as TP
    r = {}
    mws = []
    for i in range(0, R2.N_COLLECT, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        col = {'mlp_write': [], 'attn_write': [], 'entry_norm': []}
        model(b, collect=col)
        mws.append(torch.stack([w.float().cpu() for w in col['mlp_write']], 1))
    mw = torch.cat(mws).transpose(1, 2).reshape(-1, DEPTH, Q.D)
    del mws
    gids = Q.HELD[:R2.N_COLLECT, :Q.T].reshape(-1).cpu().numpy().astype(np.int64)
    elig = np.ones(len(gids), bool)
    gmeans, lin = R2.build_linear_tables(model)
    lin_cpu = lin.float().cpu()
    gid_t = torch.from_numpy(gids)
    td, ve = [], []
    for l in range(DEPTH):
        X64 = mw[:, l, :].double()
        tot = float((X64 - X64.mean(0)).pow(2).sum())
        if tot < 1e-10:                       # constant write: probes undefined
            td.append({'r2': None, 'shuffle_ctl': None, 'constant_write': True})
            ve.append(None)
            continue
        r2, G, cov = TP.group_r2(X64, gids, elig, R2.MIN_COUNT)
        ctl = TP.shuffle_control(X64, gids, cov)
        td.append({'r2': round(r2, 4), 'shuffle_ctl': round(ctl, 4)})
        pred = lin_cpu[l][gid_t].double()
        resid = float((X64 - pred).pow(2).sum())
        ve.append(round(1 - resid / tot, 4))
    r['token_determined_mlp'] = td
    r['linear_held_variance_explained'] = ve
    del mw
    base = R2.held_ce_sub(model)
    r['base_ce_fp32_fullheld'] = round(base, 5)
    causal = []
    for l in range(DEPTH):
        floor_tab = gmeans[l][None].expand(Q.V, Q.D).contiguous()
        cef = R2.held_ce_sub(model, floor_tab, l)
        cel = R2.held_ce_sub(model, lin[l], l)
        dfloor, dlin = cef - base, cel - base
        rec = round(1 - dlin / dfloor, 4) if dfloor > 1e-6 else None
        causal.append({'dce_floor': round(dfloor, 5), 'dce_lin': round(dlin, 5),
                       'lin_recovered': rec})
        tds = td[l]['r2']
        print(f"  L{l}: tokdet {tds} linVE {ve[l]} "
              f"dCEfloor {dfloor:.4f} rec {rec}", flush=True)
    r['causal_per_layer'] = causal
    del gmeans, lin
    torch.cuda.empty_cache()
    return r


def save(out, r):
    out['probe'] = r
    json.dump(out, open(f'{QK}/qk_v8.json', 'w'), indent=2)


def main():
    Q.gpu_guard(min_free=7000)
    model, ck = V8T.load_v8()
    out = json.load(open(f'{QK}/qk_v8.json'))
    r = {'held_ce_bf16': ck['log']['final_held_ce'], 'lr': ck['config']['lr'],
         'group_coeff': ck['config']['group_coeff'], 'spikes': ck['log']['spikes']}
    t0 = time.time()

    base, means = R2.base_and_means(model)
    r['base_ce_fp32_abl'] = round(base, 5)

    # ---- 1. causal consumption graph ----
    dce = R2.consumption_graph(model, base, means)
    pairs = [(li, si, dce[li][si]) for li in dce for si in dce[li]]
    pairs.sort(key=lambda p: -p[2])
    r['consumption_top40'] = [
        {'consumer': ('readout' if li == DEPTH else f'block{li}'),
         'source': R2.stream_name(si), 'dce': round(v, 5)}
        for li, si, v in pairs[:40]]
    r['consumption_matrix'] = {str(li): {str(si): round(v, 6)
                                         for si, v in dce[li].items()}
                               for li in dce}
    save(out, r)

    # per-source-block total consumption (dead-block check)
    totals = {}
    for j in range(DEPTH):
        tot = sum(dce[li].get(si, 0.0) for li in dce
                  for si in (1 + 2 * j, 2 + 2 * j) if si in dce[li])
        totals[j] = round(tot, 5)
    r['per_block_total_consumption'] = {str(j): totals[j] for j in totals}
    r['min_block_total'] = min(totals.values())
    r['dead_blocks_below_0.001'] = [j for j in totals if totals[j] < 0.001]
    print(f"per-block totals: {totals}", flush=True)

    # ---- 2. wiring agreement ----
    scores, _ = R2.weight_wiring(model, 'V8')
    wp = [(li, si) for li in scores for si in scores[li]]     # visible write pairs
    sup = [scores[li][si] for li, si in wp]
    cau = [dce[li][si] for li, si in wp]
    r['wiring_n_pairs'] = len(wp)
    r['wiring_spearman_all'] = round(R2.spearman(sup, cau), 4)
    eff = [k for k in range(len(wp)) if cau[k] > EFFECTUAL]
    r['wiring_n_effectual'] = len(eff)
    r['wiring_spearman_effectual'] = round(
        R2.spearman([sup[k] for k in eff], [cau[k] for k in eff]), 4)
    top_sup = set(np.argsort(sup)[::-1][:10].tolist())
    top_cau = set(np.argsort(cau)[::-1][:10].tolist())
    r['wiring_top10_precision'] = len(top_sup & top_cau) / 10.0
    r['wiring_top10_support_edges'] = [
        {'consumer': ('readout' if wp[k][0] == DEPTH else f'block{wp[k][0]}'),
         'source': R2.stream_name(wp[k][1]), 'support': round(sup[k], 3),
         'dce': round(cau[k], 5)} for k in sorted(top_sup, key=lambda k: -sup[k])]
    r['weight_support_matrix'] = {str(li): {str(si): round(v, 3)
                                            for si, v in scores[li].items()}
                                  for li in scores}
    print(f"wiring: n {len(wp)}, Spearman all {r['wiring_spearman_all']}, "
          f"effectual ({len(eff)} pairs) {r['wiring_spearman_effectual']}, "
          f"top-10 precision {r['wiring_top10_precision']}", flush=True)
    save(out, r)

    # ---- 3. routing sparsity (norm-weighted effective inputs per consumer) ----
    sp = R2.routing_sparsity(model)
    r['routing_sparsity'] = {str(k): v for k, v in sp.items()}
    r['mean_effective_inputs'] = round(
        float(np.mean([v['effective_inputs'] for v in sp.values()])), 3)

    # ---- 4. exact-term decomposability at blocks 4, 6, 9 ----
    r['terms'] = {}
    for mid in TERM_BLOCKS:
        R2.MID = mid
        tr = R2.term_decomposition(model, 'V8', base)
        r['terms'][f'block{mid}'] = tr
        print(f"terms block{mid}: n {tr['n_terms']} "
              f"rel_err {tr['reassembly_rel_err']:.2e} k95 {tr['k95_energy']} "
              f"recovered {tr['topk95_recovered']}", flush=True)
    save(out, r)

    # ---- 5. standard probes ----
    r['probes'] = standard_probes_guarded(model)

    print(f"probe done in {time.time() - t0:.0f}s", flush=True)
    save(out, r)
    print('saved qk_v8.json (probe)', flush=True)


if __name__ == '__main__':
    main()
