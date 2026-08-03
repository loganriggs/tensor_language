"""DEPTH-ROUTING PROTOTYPES, interpretability scoring (companion to
qk_deeproute_train.py). For every surviving variant V1..V7:

1. CAUSAL CONSUMPTION GRAPH: for every (source stream i, consumer li) pair - li =
   block entry 0..11 or readout 12, i = embedding (0) or write stream (1..2*li) -
   mean-ablate stream i IN CONSUMER li's ASSEMBLY ONLY (stream mean over 96 held
   seqs x 512 positions), fp32 forward, delta CE on those 96 seqs. All 169 pairs.
2. WEIGHT-READABLE WIRING (V5/V7, where a weight prediction exists): predicted
   strength of consumer l reading write k = Frobenius support of block l's seven
   input matrices (c_q, c_k, c_q2, c_k2, c_v, Left, Right; readout: tied wte) on
   the fixed 16-dim subspace P_k; for V7 also the routing prediction
   ||w_hat_l[P_k]||^2 and the product. AGREEMENT = Spearman rank correlation with
   the causal delta CE over write-source pairs. For V6/V7 also the binary mask
   check: causal effect mass on invisible pairs is exactly zero by construction.
3. ROUTING SPARSITY: realized per-position contribution shares p_i propto
   |coef_i| * ||v_i(pos)|| per consumer; mean entropy -> effective number of
   inputs; plus structural support counts.
4. EXACT-TERM DECOMPOSABILITY at mid block 6: named per-source terms
   term_i = coef_i(v_i) * v_i must reassemble the captured entry (rel err < 1e-5,
   fp32); k95 = fewest terms carrying 95 percent of the entry energy; causal
   check: entry_override with the top-k95 reconstruction vs mean-entry floor.
   For V2 the normalizer is a bubble: naive expansion freezes the denominator at
   its held mean; the reassembly error of that naive expansion = the leakage.
5. STANDARD PROBES per layer (machinery verbatim from qk_denseform_2.py /
   qk_tokenline_probe.py): token-determined fraction of each mlp write
   (group-mean R^2 + shuffle control) and linear-in-normed-embedding (held
   variance explained + causal recovered fraction vs global-mean floor, full
   held slice, fp32).

Everything merges into qk_deeproute.json (keys 'ce', 'lrsweep', then per-variant).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_tokenline_probe as TP
import qk_deeproute_train as R
from qk_deeproute_train import DeepRoute, DEPTH, SUBDIM

QK = Q.QK
DEV = 'cuda'
D, V, T = Q.D, Q.V, Q.T
HELD, TRAIN = Q.HELD, Q.TRAIN
ABL_N = 96                # held seqs for the consumption graph + term causal checks
N_COLLECT = 200           # held seqs for token-determined / variance probes
N_TABLE = 1500            # train seqs for the linear tables
MIN_COUNT = 5
MID = 6                   # the mid block for term decomposability
NSTREAM = 1 + 2 * DEPTH


def load_model(variant):
    ck = torch.load(f'{QK}/qk_deeproute_{variant}.pt', map_location=DEV,
                    weights_only=False)
    torch.manual_seed(Q.SEED)
    m = DeepRoute(variant, DEPTH, unit_norm=ck['config'].get('unit_norm'))
    m = m.to(DEV)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


def stream_name(i):
    if i == 0:
        return 'emb'
    j, kind = (i - 1) // 2, ('attn' if (i - 1) % 2 == 0 else 'mlp')
    return f'{kind}{j}'


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    dn = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum()) / dn if dn > 0 else 0.0


@torch.no_grad()
def base_and_means(model):
    """One fp32 pass over HELD[:ABL_N]: per-token base CE + per-stream mean vectors."""
    sums = torch.zeros(NSTREAM, D, device=DEV, dtype=torch.float64)
    n = 0
    pts = []
    for i in range(0, ABL_N, 8):
        b = HELD[i:i + 8]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        logits = model(b[:, :T], collect=col)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        pts.append(ce.cpu())
        e = F.rms_norm(model.wte(b[:, :T]), (D,))
        sums[0] += e.double().sum((0, 1))
        for l in range(DEPTH):
            sums[1 + 2 * l] += col['attn_write'][l].double().sum((0, 1))
            sums[2 + 2 * l] += col['mlp_write'][l].double().sum((0, 1))
        n += b.shape[0] * T
    pt = torch.cat(pts).numpy()
    return float(pt.mean()), (sums / n).float()


@torch.no_grad()
def ce_with(model, sub_entry=None, entry_override_fn=None):
    """fp32 CE over HELD[:ABL_N]. sub_entry: {li: {i: (D,) mean}} broadcast per batch.
    entry_override_fn: batch_slice -> {li: tensor} (for term-truncation checks)."""
    tot, n = 0.0, 0
    for i in range(0, ABL_N, 8):
        b = HELD[i:i + 8]
        B = b.shape[0]
        se = None
        if sub_entry is not None:
            se = {li: {si: mv[None, None, :].expand(B, T, D)
                       for si, mv in d.items()} for li, d in sub_entry.items()}
        eo = entry_override_fn(i) if entry_override_fn is not None else None
        logits = model(b[:, :T], sub_entry=se, entry_override=eo)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        tot += ce.sum().item(); n += ce.numel()
    return tot / n


@torch.no_grad()
def consumption_graph(model, base, means):
    """dce[li][i] for every existing (visible) pair."""
    dce = {}
    t0 = time.time()
    for li in range(DEPTH + 1):
        dce[li] = {}
        for si in model.vis[li]:
            ce = ce_with(model, sub_entry={li: {si: means[si]}})
            dce[li][si] = ce - base
        print(f"  consumption row {li}: " + ", ".join(
            f"{stream_name(si)} {dce[li][si]:+.4f}" for si in
            sorted(dce[li], key=lambda s: -dce[li][s])[:4]) +
            f"  ({time.time() - t0:.0f}s)", flush=True)
    return dce


def weight_wiring(model, variant):
    """Support scores score[l][write k] from weights alone (V5/V7 meaningful)."""
    scores, route = {}, {}
    for l in range(DEPTH + 1):
        mats = ([model.wte.weight] if l == DEPTH else
                [model.h[l].c_q.weight, model.h[l].c_k.weight,
                 model.h[l].c_q2.weight, model.h[l].c_k2.weight,
                 model.h[l].c_v.weight, model.h[l].Left.weight,
                 model.h[l].Right.weight])
        row, rrow = {}, {}
        for k in range(2 * DEPTH):
            si = k + 1
            if si not in model.vis[l] or si > 2 * min(l, DEPTH):
                continue
            dims = slice(SUBDIM * k, SUBDIM * (k + 1))
            row[si] = float(sum(M[:, dims].float().pow(2).sum().item()
                                for M in mats) ** 0.5)
            if variant == 'V7':
                w = model.row_w(l).detach()
                rrow[si] = float(w[dims].pow(2).sum().item())
        scores[l] = row
        route[l] = rrow
    return scores, route


@torch.no_grad()
def routing_sparsity(model, n_seq=48):
    """Realized contribution-share entropy per consumer over HELD[:n_seq]."""
    Hsum = {li: 0.0 for li in range(DEPTH + 1)}
    npos = 0
    for i in range(0, n_seq, 8):
        b = HELD[i:i + 8, :T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        coef = {}
        model(b, collect=col, coef_out=coef)
        e = F.rms_norm(model.wte(b), (D,))
        norms = [e.norm(dim=-1)]
        for l in range(DEPTH):
            norms.append(col['attn_write'][l].norm(dim=-1))
            norms.append(col['mlp_write'][l].norm(dim=-1))
        for li in range(DEPTH + 1):
            idxs, c = coef[li]
            contrib = c.abs() * torch.stack([norms[si] for si in idxs], -1)
            p = contrib / (contrib.sum(-1, keepdim=True) + 1e-12)
            H = -(p * (p + 1e-12).log()).sum(-1)
            Hsum[li] += H.sum().item()
        npos += b.numel()
    out = {}
    for li in range(DEPTH + 1):
        H = Hsum[li] / npos
        out[li] = {'support': len(model.vis[li]), 'mean_entropy_nats': round(H, 4),
                   'effective_inputs': round(math.exp(H), 3)}
    return out


@torch.no_grad()
def term_decomposition(model, variant, base):
    """Exact per-source term expansion at block MID + causal top-k check."""
    r = {}
    # capture streams, coefficients, and the true entry on HELD[:16]
    b = HELD[:16, :T]
    col = {'entry_norm': [], 'attn_write': [], 'mlp_write': [], 'entry': []}
    coef = {}
    model(b, collect=col, coef_out=coef)
    e = F.rms_norm(model.wte(b), (D,))
    streams = [e]
    for l in range(DEPTH):
        streams.append(col['attn_write'][l])
        streams.append(col['mlp_write'][l])
    h_true = col['entry'][MID]
    idxs, c = coef[MID]
    terms = [c[..., k, None] * streams[si] for k, si in enumerate(idxs)]
    recon = sum(terms)
    rel = float((recon - h_true).norm() / (h_true.norm() + 1e-12))
    r['n_terms'] = len(idxs)
    r['term_names'] = [stream_name(si) for si in idxs]
    r['reassembly_rel_err'] = rel
    r['exact'] = bool(rel < 1e-5)
    if variant == 'V2':
        # leakage of the naive (denominator-frozen) polynomial expansion
        w = model.row_w(MID)
        dots = torch.stack([s.float() @ w for s in streams[:len(streams)]], -1)
        num = dots[..., idxs].pow(2)
        den_mean = float(num.sum(-1).mean())
        naive = sum((num[..., k, None] / den_mean) * streams[si]
                    for k, si in enumerate(idxs))
        r['naive_frozen_denominator_rel_err'] = float(
            (naive - h_true).norm() / (h_true.norm() + 1e-12))
    # rank terms by mean squared norm; k95 by energy of the reconstruction
    en = torch.tensor([float(t.float().pow(2).sum()) for t in terms])
    order = torch.argsort(en, descending=True).tolist()
    tot_err_prev = None
    k95 = len(idxs)
    for k in range(1, len(idxs) + 1):
        part = sum(terms[j] for j in order[:k])
        ve = 1 - float((part - h_true).pow(2).sum() / h_true.pow(2).sum())
        if ve >= 0.95:
            k95 = k
            break
    r['k95_energy'] = k95
    r['term_energy_ranked'] = [[stream_name(idxs[j]), round(float(en[j]), 2)]
                               for j in order]
    # causal check over HELD[:ABL_N]: entry_override with top-k95 terms vs mean floor
    keep = [idxs[j] for j in order[:k95]]

    def override_topk(i0):
        b2 = HELD[i0:i0 + 8, :T]
        col2 = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        coef2 = {}
        model(b2, collect=col2, coef_out=coef2)
        e2 = F.rms_norm(model.wte(b2), (D,))
        st = [e2]
        for l in range(DEPTH):
            st.append(col2['attn_write'][l])
            st.append(col2['mlp_write'][l])
        ii, cc = coef2[MID]
        h = sum(cc[..., k, None] * st[si] for k, si in enumerate(ii) if si in keep)
        return {MID: h}

    ce_topk = ce_with(model, entry_override_fn=override_topk)
    # floor: replace block MID's entry with its held mean vector
    hsum, n = torch.zeros(D, device=DEV, dtype=torch.float64), 0
    for i in range(0, ABL_N, 8):
        b2 = HELD[i:i + 8, :T]
        col2 = {'entry_norm': [], 'attn_write': [], 'mlp_write': [], 'entry': []}
        model(b2, collect=col2)
        hsum += col2['entry'][MID].double().sum((0, 1))
        n += b2.numel()
    hmean = (hsum / n).float()

    def override_mean(i0):
        B = HELD[i0:i0 + 8].shape[0]
        return {MID: hmean[None, None, :].expand(B, T, D)}

    ce_floor = ce_with(model, entry_override_fn=override_mean)
    d_topk, d_floor = ce_topk - base, ce_floor - base
    r['dce_topk95'] = round(d_topk, 5)
    r['dce_entry_mean_floor'] = round(d_floor, 5)
    r['topk95_recovered'] = (round(1 - d_topk / d_floor, 4)
                             if d_floor > 1e-6 else None)
    return r


# ---------------- standard probes (adapted verbatim from qk_denseform_2.py) ----------------
@torch.no_grad()
def build_linear_tables(model):
    sums = torch.zeros(DEPTH, V, D, device=DEV)
    cnts = torch.zeros(V, device=DEV)
    for i in range(0, N_TABLE, 8):
        b = TRAIN[i:i + 8, :T]
        col = {'mlp_write': [], 'attn_write': [], 'entry_norm': []}
        model(b, collect=col)
        idsf = b.reshape(-1)
        for l in range(DEPTH):
            sums[l].index_add_(0, idsf, col['mlp_write'][l].float().reshape(-1, D))
        cnts.index_add_(0, idsf, torch.ones_like(idsf, dtype=torch.float32))
    gmeans = sums.sum(1) / cnts.sum()
    E = F.rms_norm(model.wte.weight.detach(), (D,))
    Ea = torch.cat([E, torch.ones(V, 1, device=DEV)], 1)
    A = Ea.t() @ (cnts[:, None] * Ea)
    lam = 1e-4 * torch.trace(A) / (D + 1)
    Areg = A + lam * torch.eye(D + 1, device=DEV)
    lin = torch.empty_like(sums)
    for l in range(DEPTH):
        W = torch.linalg.solve(Areg, Ea.t() @ sums[l])
        lin[l] = Ea @ W
    return gmeans, lin


@torch.no_grad()
def held_ce_sub(model, sub_table=None, layer=None):
    tot, n = 0.0, 0
    for i in range(0, len(HELD), 8):
        b = HELD[i:i + 8]
        inp = b[:, :T]
        sub = None if sub_table is None else {layer: sub_table[inp]}
        logits = model(inp, mlp_sub=sub)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        tot += ce.sum().item(); n += ce.numel()
    return tot / n


@torch.no_grad()
def standard_probes(model):
    r = {}
    mws = []
    for i in range(0, N_COLLECT, 8):
        b = HELD[i:i + 8, :T]
        col = {'mlp_write': [], 'attn_write': [], 'entry_norm': []}
        model(b, collect=col)
        mws.append(torch.stack([w.float().cpu() for w in col['mlp_write']], 1))
    mw = torch.cat(mws).transpose(1, 2).reshape(-1, DEPTH, D)
    del mws
    gids = HELD[:N_COLLECT, :T].reshape(-1).cpu().numpy().astype(np.int64)
    elig = np.ones(len(gids), bool)
    gmeans, lin = build_linear_tables(model)
    lin_cpu = lin.float().cpu()
    gid_t = torch.from_numpy(gids)
    td, ve = [], []
    for l in range(DEPTH):
        X64 = mw[:, l, :].double()
        r2, G, cov = TP.group_r2(X64, gids, elig, MIN_COUNT)
        ctl = TP.shuffle_control(X64, gids, cov)
        td.append({'r2': round(r2, 4), 'shuffle_ctl': round(ctl, 4)})
        pred = lin_cpu[l][gid_t].double()
        resid = float((X64 - pred).pow(2).sum())
        tot = float((X64 - X64.mean(0)).pow(2).sum())
        ve.append(round(1 - resid / tot, 4))
    r['token_determined_mlp'] = td
    r['linear_held_variance_explained'] = ve
    del mw
    base = held_ce_sub(model)
    r['base_ce_fp32_fullheld'] = round(base, 5)
    causal = []
    for l in range(DEPTH):
        floor_tab = gmeans[l][None].expand(V, D).contiguous()
        cef = held_ce_sub(model, floor_tab, l)
        cel = held_ce_sub(model, lin[l], l)
        dfloor, dlin = cef - base, cel - base
        rec = round(1 - dlin / dfloor, 4) if dfloor > 1e-6 else None
        causal.append({'dce_floor': round(dfloor, 5), 'dce_lin': round(dlin, 5),
                       'lin_recovered': rec})
        print(f"  L{l}: tokdet {td[l]['r2']:.3f} linVE {ve[l]:.3f} "
              f"dCEfloor {dfloor:.4f} rec {rec}", flush=True)
    r['causal_per_layer'] = causal
    del gmeans, lin
    torch.cuda.empty_cache()
    return r


def analyze(variant):
    print(f"==== analyzing {variant} ====", flush=True)
    t0 = time.time()
    model, ck = load_model(variant)
    r = {'held_ce_bf16': ck['log']['final_held_ce'], 'lr': ck['config']['lr'],
         'spikes': ck['log']['spikes'], 'unit_norm': ck['config'].get('unit_norm')}

    base, means = base_and_means(model)
    r['base_ce_fp32_abl'] = round(base, 5)

    # 1. causal consumption graph
    dce = consumption_graph(model, base, means)
    pairs = [(li, si, dce[li][si]) for li in dce for si in dce[li]]
    pairs.sort(key=lambda p: -p[2])
    r['consumption_top40'] = [
        {'consumer': ('readout' if li == DEPTH else f'block{li}'),
         'source': stream_name(si), 'dce': round(v, 5)} for li, si, v in pairs[:40]]
    r['consumption_matrix'] = {str(li): {str(si): round(v, 6)
                                         for si, v in dce[li].items()} for li in dce}

    # 2. weight-readable wiring
    if variant in ('V5', 'V7'):
        scores, route = weight_wiring(model, variant)
        cw, cs, cr, cc = [], [], [], []
        for li in scores:
            for si, sc in scores[li].items():
                cw.append(dce[li][si]); cs.append(sc)
                if variant == 'V7':
                    cr.append(route[li][si]); cc.append(sc * route[li][si])
        r['wiring_n_pairs'] = len(cw)
        r['wiring_spearman_support'] = round(spearman(cs, cw), 4)
        if variant == 'V7':
            r['wiring_spearman_route'] = round(spearman(cr, cw), 4)
            r['wiring_spearman_support_x_route'] = round(spearman(cc, cw), 4)
        r['weight_support_matrix'] = {str(li): {str(si): round(v, 3)
                                                for si, v in scores[li].items()}
                                      for li in scores}
        print(f"  wiring Spearman(support, causal) = {r['wiring_spearman_support']}"
              + (f", route = {r.get('wiring_spearman_route')}" if variant == 'V7'
                 else ""), flush=True)

    # 3. routing sparsity
    sp = routing_sparsity(model)
    r['routing_sparsity'] = {str(k): v for k, v in sp.items()}
    r['mean_effective_inputs'] = round(
        float(np.mean([v['effective_inputs'] for v in sp.values()])), 3)

    # 4. exact-term decomposability at block MID
    r['terms_block6'] = term_decomposition(model, variant, base)
    print(f"  terms block6: n {r['terms_block6']['n_terms']} "
          f"rel_err {r['terms_block6']['reassembly_rel_err']:.2e} "
          f"k95 {r['terms_block6']['k95_energy']} "
          f"recovered {r['terms_block6']['topk95_recovered']}", flush=True)

    # 5. standard probes
    r['probes'] = standard_probes(model)

    print(f"  {variant} done in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return r


if __name__ == '__main__':
    Q.gpu_guard()
    out = json.load(open(f'{QK}/qk_deeproute.json'))
    for variant in R.VARIANTS:
        if not os.path.exists(f'{QK}/qk_deeproute_{variant}.pt'):
            print(f"{variant}: no checkpoint (diverged?) -- skip", flush=True)
            continue
        if variant in out and 'terms_block6' in out.get(variant, {}):
            print(f"{variant}: already analyzed -- skip", flush=True)
            continue
        out[variant] = analyze(variant)
        json.dump(out, open(f'{QK}/qk_deeproute.json', 'w'), indent=2)
    print('saved qk_deeproute.json', flush=True)
