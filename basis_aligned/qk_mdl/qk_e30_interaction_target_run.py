"""E30 INTERACTION-ADJUSTED CAUSAL TARGET + RE-SCORING OF EVERY STORED WIRING
TABLE (checkpoint-only; NO training).

WHY (BRAINSTORM_STATE "FOUNDATIONS CORRECTION 2026-08-07", item b): E26 found
only 18% of module pairs near-additive, 148/300 superadditive, and set
causal_ground_truth_changes_materially = TRUE. Every readability number in the
program is a Spearman of a WEIGHT table against a FIRST-ORDER single-ablation
causal vector, and that vector is mis-specified. This run replaces it with
interaction-aware importances and RE-SCORES every stored wiring table so the
whole frontier is comparable again on one target.

TWO ESTIMATORS, BOTH REPORTED (never silently one):

  (A) SHAPLEY-2 (second-order truncation of the Shapley value)
        adj_A(x) = dCE(x) + 0.5 * sum_{y != x} I(x, y),  I = measured pairwise
      interaction (E26's definition: I(x,y) = dCE(x,y) - dCE(x) - dCE(y)).
      INTERPRETATION: importance of x in the INTACT model, with each pairwise
      surplus/deficit split evenly between its two members. It is an
      extrapolation: it assumes the interaction expansion truncates at order 2,
      so it is exact only if triples and higher vanish. It stays near the
      first-order vector by construction (it is first-order plus a correction).

  (B) LEAVE-ONE-IN-CONTEXT (LOIC), K = 8 random contexts
        loic(x) = mean_c [ CE(ablate S_c u {x}) - CE(ablate S_c \\ {x}) ],
      S_c = a fixed random half of the 25 module sources. INTERPRETATION:
      importance of x in a HALF-DESTROYED model. It measures the true marginal
      effect deep in the ablation lattice, so it captures interactions of ALL
      orders -- but only at one coalition size, and with sampling noise over
      the 8 contexts. Where A and B disagree, the disagreement is exactly the
      contribution of third-and-higher-order terms plus context variance.
      Evaluation trick that keeps this at ~26 evals per context (208 total,
      not 400): CE(S_c) is shared -- for x in S_c the "without" term IS
      CE(S_c), for x not in S_c the "with" term IS CE(S_c).

EDGE-LEVEL TARGETS (the wiring tables have 156 (consumer, writer) edges, and
E26 measured interactions only at the 25-source MODULE tier and the readout's
24 edges). Per checkpoint:
  - measure TRUE per-consumer edge-pair interactions for the THREE consumers
    with the largest total first-order consumption (that checkpoint's own
    consumption matrix, writer sources only);
  - for every other consumer use the MODULE-TIER correction distributed over
    that module's edges in proportion to their first-order consumption share:
        adj_A(li, si) = dCE(li, si) + corr_mod(si) * share(li, si),
        corr_mod(si) = 0.5 * sum_j I_mod(si, j),
        share(li, si) = dCE(li, si) / sum_l' dCE(l', si)   (relu'd; uniform if
                        the module's total is non-positive).
    LIMITATION, stated explicitly in the JSON: this assumes a module's total
    interaction correction is carried by its edges in the same proportion as
    its first-order consumption. It is VALIDATED, not assumed, on the three
    measured consumers (exact vs approximated adjusted values, Spearman +
    max abs deviation) and, on the reference checkpoint only, against a
    direct EDGE-level LOIC (156 edges x 8 contexts).
  The LOIC target is mapped to edges by the same share rule (module tier only
  -- an edge-level LOIC on every checkpoint is out of budget).

CONTROLS (hard gates):
  1. E26 REPRODUCTION: recompute qk_e9_a's 25 module singles through this
     runner's own joint-ablation path and match qk_e26.json's stored
     singles_dce to 1e-6; the base CE must match too. (E26's cached pair CEs
     in qk_e26_partial.json are then reused for the reference checkpoint --
     the gate is what licenses the reuse.)
  2. STORED-SPEARMAN GATE, per checkpoint: this runner's weight table and
     causal vector must reproduce the checkpoint's STORED plain
     wiring_spearman_all to 1e-3 before any adjusted number is computed.
  3. SHUFFLED-INTERACTION NULL: permute the interaction matrix over pairs and
     rebuild the adjusted target. Under a permutation every source draws its
     correction from the same pool, so the correction becomes near-constant
     across sources and the adjusted target must COLLAPSE toward first order
     (Spearman(adj_null, first) -> 1). Reported as the mean over 200
     permutations next to the true Spearman(adj, first).

EVERY Spearman in this run carries a percentile bootstrap CI over edges
(reviewer-2 R1: n = 156 -> SE ~ 0.08, so bare point estimates are not
interpretable). B = 2000 resamples.

OUTPUT qk_e30.json: the two adjusted targets per checkpoint, the full
comparison table (plain and covariance-composed weight tables x first-order /
Shapley-2 / LOIC targets, each with CI), the arm ORDERING under each target,
which arms move most, and the approximation-validation blocks. Pair CEs cached
in qk_e30_partial.json (resume-safe); idempotent on qk_e30.json keys."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, C, DEPTH, F, torch
import qk_deeproute_train_2 as R2
import qk_e7_evenout_run as E7R
import qk_e15_reinvest_run as E15R
import qk_e16_shrinkemb_run as E16R
import qk_e17_composed_wiring as E17
import qk_e18_probe_upgrades as E18U

E.DEV = 'cpu' if E.SMOKE else 'cuda'
if E.SMOKE:
    R2.DEV = 'cpu'
    R2.ABL_N = 8
    E18U.N_COV = 24

JP = E.jpath('qk_e30.json')
PARTIAL = E.jpath('qk_e30_partial.json')
NS = 1 + 2 * DEPTH                 # 25 module sources (emb + 24 writers)
NG = E.NGROUP                      # 24 slot groups
ABL_N = R2.ABL_N                   # 96 old-cooc held seqs (the family convention)
GATE_TOL = 1e-3
E26_TOL = 1e-6
K_LOIC = 8                         # LOIC contexts
LOIC_SEED = 30
N_BOOT = 2000
N_PERM = 200
N_TOPC = 3                         # consumers with measured edge-pair maps
sname = R2.stream_name


# ============================ ablation machinery ============================
@torch.no_grad()
def base_means_rem(model, Ws):
    """One fp32 pass over the old-cooc held rows [:ABL_N]: base CE, per-stream
    mean vectors, and (E16 family only) the PER-CONSUMER remnant means used as
    stream 0's ablation value. Same rows / same ABL_N / same fp32 convention as
    R2.base_and_means, generalized to arbitrary stream width."""
    model.eval().float()
    dev = E.OLD_HELD.device
    sums = torch.zeros(NS, Ws, device=dev, dtype=torch.float64)
    rsums, n, pts = None, 0, []
    has_rem = hasattr(model, 'remnants')
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
        if has_rem:
            rems = model.remnants(e)
            if rsums is None:
                rsums = [None if r is None else
                         torch.zeros(Ws, dtype=torch.float64, device=dev)
                         for r in rems]
            for li, r in enumerate(rems):
                if r is not None:
                    rsums[li] += r.double().sum((0, 1))
        n += b.shape[0] * Q.T
    rem = None if rsums is None else \
        [None if s is None else (s / n).float() for s in rsums]
    return float(torch.cat(pts).mean()), (sums / n).float(), rem


@torch.no_grad()
def ce_subs(model, subs, Ws):
    """fp32 CE over the old-cooc held rows with an arbitrary set of
    (consumer, source) mean substitutions applied in ONE forward -- downstream
    writes are recomputed, so ablations propagate exactly as in the stored
    singles (R2.ce_with semantics, plural)."""
    tot, n = 0.0, 0
    for i in range(0, ABL_N, 8):
        b = E.OLD_HELD[i:i + 8]
        B = b.shape[0]
        se = {li: {si: mv[None, None, :].expand(B, Q.T, Ws)
                   for si, mv in d.items()} for li, d in subs.items()}
        logits = model(b[:, :Q.T], sub_entry=se)
        ce = F.cross_entropy(logits.reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        tot += ce.sum().item()
        n += ce.numel()
    return tot / n


def module_subs(model, srcs, means, rem, consumers=None):
    """Sources mean-substituted JOINTLY at EVERY visible consumer (E26's
    module-tier convention)."""
    cons = range(DEPTH + 1) if consumers is None else consumers
    subs = {}
    for li in cons:
        d = {}
        for si in srcs:
            if si not in model.vis[li]:
                continue
            if si == 0 and rem is not None:
                if rem[li] is None:
                    continue
                d[0] = rem[li]
            else:
                d[si] = means[si]
        if d:
            subs[li] = d
    return subs


def run_jobs(tag, jobs, fn):
    """Resume-safe evaluation cache: jobs = [(job_id, payload), ...]."""
    part = E.loadj(PARTIAL)
    d = part.get(tag, {})
    todo = [(jid, p) for jid, p in jobs if jid not in d]
    if todo:
        t0 = time.time()
        for k, (jid, p) in enumerate(todo):
            d[jid] = fn(p)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                part = E.loadj(PARTIAL)
                part[tag] = d
                json.dump(part, open(PARTIAL, 'w'))
                el = time.time() - t0
                print(f"  {tag}: {k + 1}/{len(todo)} evals ({el:.0f}s, "
                      f"{el / (k + 1):.2f}s/eval, "
                      f"~{el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min "
                      f"left)", flush=True)
    return d


# ============================ statistics ====================================
def boot_ci(a, b, seed=0, B=N_BOOT):
    """Percentile bootstrap CI of the Spearman over EDGES (reviewer-2 R1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, n)
        vals[k] = R2.spearman(a[idx], b[idx])
    return {'spearman': round(R2.spearman(a, b), 4),
            'ci95': [round(float(np.percentile(vals, 2.5)), 4),
                     round(float(np.percentile(vals, 97.5)), 4)],
            'boot_sd': round(float(vals.std()), 4), 'n': int(n)}


def shares(dce, si, edges):
    """Edge shares of module si's total first-order consumption (relu'd;
    uniform fallback when the module's total is non-positive)."""
    ed = [(li, s) for li, s in edges if s == si]
    vals = [max(dce[li][s], 0.0) for li, s in ed]
    tot = sum(vals)
    if tot <= 0:
        return {e: 1.0 / max(len(ed), 1) for e in ed}
    return {e: v / tot for e, v in zip(ed, vals)}


# ============================ checkpoint registry ===========================
def registry():
    """(tag, checkpoint stem, factory, slot dim, remnant flag, stored light
    probe location, tier). tier 'full' = module map + LOIC + measured edge maps
    for the top-3 consumers; tier 'module' = module map + LOIC only (edge
    corrections entirely through the documented share approximation)."""
    if E.SMOKE:
        return [dict(tag='E9a_recipe_s0', stem=None,
                     factory=E7R.make_e7m1, slot=E.SUB, remnant=False,
                     lp=None, tier='full', arm='recipe', seed=0)]
    import qk_e20_codebook_run as E20R
    import qk_e20b_vark_run as E20BR
    import qk_e22_predbasis_run as E22R
    import qk_e23_idwiring_run as E23R
    S = 15
    return [
        # ---- tier "full": the arms the frontier claim rests on ----
        dict(tag='E9a_recipe_s0', stem='qk_e9_a', factory=E7R.make_e7m1,
             slot=E.SUB, remnant=False, lp=('qk_e9.json', 'light_probe_E9a'),
             tier='full', arm='recipe', seed=0),
        dict(tag='E19a_frontier_s0', stem='qk_e19_a',
             factory=lambda: E15R.make_e15c(s=S), slot=S, remnant=False,
             lp=('qk_e19.json', 'light_probe_E19a_var_dims'),
             tier='full', arm='frontier', seed=0),
        dict(tag='E27b_recipe_s1', stem='qk_e27_recipe_s1',
             factory=E7R.make_e7m1, slot=E.SUB, remnant=False,
             lp=('qk_e27.json', 'light_probe_E27b_recipe'),
             tier='full', arm='recipe', seed=1),
        dict(tag='E27a_frontier_s1', stem='qk_e27_frontier_s1',
             factory=lambda: E15R.make_e15c(s=S), slot=S, remnant=False,
             lp=('qk_e27.json', 'light_probe_E27a_frontier'),
             tier='full', arm='frontier', seed=1),
        dict(tag='E22a_predbasis_s0', stem='qk_e22_a',
             factory=lambda: E22R.make_e22(s=S), slot=S, remnant=False,
             lp=('qk_e22.json', 'light_probe_E22a_var_dims'),
             tier='full', arm='predicate_basis', seed=0),
        # ---- tier "module": the rest of the stored frontier ----
        dict(tag='E20a_codebook', stem='qk_e20_a',
             factory=lambda: E20R.make_e20(s=S), slot=S, remnant=False,
             lp=('qk_e20.json', 'light_probe_E20a_var_dims'),
             tier='module', arm='codebook', seed=0),
        dict(tag='E20b_vark_codebook', stem='qk_e20b_a',
             factory=lambda: E20BR.make_e20b(s=S), slot=S, remnant=False,
             lp=('qk_e20b.json', 'light_probe_E20b_var_dims'),
             tier='module', arm='codebook_vark', seed=0),
        dict(tag='E23a_idwiring', stem='qk_e23_a',
             factory=lambda: E23R.make_e23(s=S), slot=S, remnant=False,
             lp=('qk_e23.json', 'light_probe_E23a_var_dims'),
             tier='module', arm='identifiable_wiring', seed=0,
             fold=E23R.fold_effective),
        dict(tag='E15c_bandwidth_3e5', stem='qk_e15_c',
             factory=lambda: E15R.make_e15c(s=S), slot=S, remnant=False,
             lp=('qk_e18.json', 'light_probe_E15c_var_dims'),
             tier='module', arm='bandwidth_3e5', seed=0),
        dict(tag='E16a_shrink', stem='qk_e16_a', factory=E16R.make_e16a,
             slot=E.SUB, remnant=True,
             lp=('qk_e16.json', 'light_probe_E16a'),
             tier='module', arm='shrink_emb', seed=0),
        dict(tag='E16b_floor', stem='qk_e16_b', factory=E16R.make_e16b,
             slot=E.SUB, remnant=True,
             lp=('qk_e16.json', 'light_probe_E16b'),
             tier='module', arm='shrink_floor', seed=0),
        dict(tag='E19b_floor_1e4', stem='qk_e19_b', factory=E16R.make_e16b,
             slot=E.SUB, remnant=True,
             lp=('qk_e19.json', 'light_probe_E19b'),
             tier='module', arm='shrink_floor_1e4', seed=0),
    ]


# ============================ per-checkpoint pipeline =======================
def load_ckpt(spec):
    if spec['stem'] is None:                       # SMOKE: fresh model
        return spec['factory']().eval().float()
    m, _ = E.load_arm(spec['stem'], spec['factory'])
    if spec.get('fold') is not None:
        mf = spec['fold'](m)
        idx = E.OLD_HELD[:2, :Q.T]
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.no_grad():
                d = float((mf(idx) - m(idx)).abs().max())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        assert d < 1e-5, f"{spec['tag']} fold identity failed ({d})"
        print(f"  {spec['tag']}: fold identity max |logit diff| {d:.2e}",
              flush=True)
        del m
        m = mf
    return m.eval().float()


def stored_lp(spec):
    if spec['lp'] is None:
        return None
    j = E.loadj(E.jpath(spec['lp'][0]))
    return j.get(spec['lp'][1])


def causal_first_order(spec, m, Ws, wp):
    """The checkpoint's first-order 156-edge causal vector: the STORED one
    where it exists (so the re-scoring sits on the same numbers the frontier
    was built from), otherwise recomputed with the identical machinery."""
    lp = stored_lp(spec)
    if lp is not None:
        dce = {int(li): {int(si): v for si, v in row.items()}
               for li, row in lp['consumption_matrix'].items()}
        return dce, lp, 'stored'
    print(f"  {spec['tag']}: no stored light probe -- computing the "
          f"consumption graph", flush=True)
    _, dce = E18U.gen_consumption(m, Ws)
    return dce, None, 'recomputed'


def module_tier(spec, m, Ws, base, means, rem, srcs):
    """25 module singles + all C(25,2) pairs at this checkpoint (E26's
    convention: joint mean substitution at every visible consumer)."""
    tag = spec['tag']
    sing = run_jobs(f'{tag}:module_singles',
                    [(f's{i}', [i]) for i in srcs],
                    lambda p: ce_subs(m, module_subs(m, p, means, rem), Ws))
    dmod = {i: sing[f's{i}'] - base for i in srcs}
    pairs = [(i, j) for a, i in enumerate(srcs) for j in srcs[a + 1:]]
    pce = run_jobs(f'{tag}:module_pairs',
                   [(f'p{i}_{j}', [i, j]) for i, j in pairs],
                   lambda p: ce_subs(m, module_subs(m, p, means, rem), Ws))
    inter = {(i, j): (pce[f'p{i}_{j}'] - base) - dmod[i] - dmod[j]
             for i, j in pairs}
    return dmod, inter


def loic_module(spec, m, Ws, base, means, rem, srcs):
    """Leave-one-in-context at the module tier: K random half-contexts; the
    shared CE(S_c) halves the eval count (26 per context, not 50)."""
    tag = spec['tag']
    rng = np.random.default_rng(LOIC_SEED)
    ctxs = []
    for c in range(K_LOIC):
        half = sorted(rng.choice(srcs, size=max(len(srcs) // 2, 1),
                                 replace=False).tolist())
        ctxs.append(half)
    jobs, meta = [], []
    for c, S in enumerate(ctxs):
        jobs.append((f'c{c}', list(S)))
        for x in srcs:
            if x in S:
                jobs.append((f'c{c}_wo{x}', [s for s in S if s != x]))
            else:
                jobs.append((f'c{c}_wi{x}', sorted(S + [x])))
        meta.append(S)
    ce = run_jobs(f'{tag}:loic',
                  jobs,
                  lambda p: (base if not p else
                             ce_subs(m, module_subs(m, p, means, rem), Ws)))
    per = {x: [] for x in srcs}
    for c, S in enumerate(ctxs):
        cs = ce[f'c{c}']
        for x in srcs:
            if x in S:
                per[x].append(cs - ce[f'c{c}_wo{x}'])
            else:
                per[x].append(ce[f'c{c}_wi{x}'] - cs)
    out = {x: float(np.mean(per[x])) for x in srcs}
    sd = {x: float(np.std(per[x])) for x in srcs}
    return out, sd, ctxs, per


def edge_tier(spec, m, Ws, base, means, rem, dce, li, srcs_li):
    """True pairwise EDGE interactions at one consumer (writer sources only)."""
    tag = spec['tag']
    pairs = [(i, j) for a, i in enumerate(srcs_li) for j in srcs_li[a + 1:]]
    pce = run_jobs(f'{tag}:edge{li}',
                   [(f'p{i}_{j}', [i, j]) for i, j in pairs],
                   lambda p: ce_subs(
                       m, module_subs(m, p, means, rem, consumers=[li]), Ws))
    return {(i, j): (pce[f'p{i}_{j}'] - base) - dce[li][i] - dce[li][j]
            for i, j in pairs}


def build_targets(dce, wp, inter_mod, loic, edge_inter, srcs):
    """The three 156-edge target vectors, in wp order.
      first    : stored/recomputed first-order dCE
      shapley  : exact within the measured consumers, share-approximated
                 elsewhere
      shapley_approx_only : share approximation EVERYWHERE (kept so the
                 checkpoints without measured consumers are comparable)
      loic     : module LOIC distributed by the same share rule
    """
    corr_mod = {i: 0.5 * sum(v for (a, b), v in inter_mod.items()
                             if a == i or b == i) for i in srcs}
    sh = {}
    for si in {s for _, s in wp}:
        sh.update(shares(dce, si, wp))
    first = [dce[li][si] for li, si in wp]
    appr = [dce[li][si] + corr_mod.get(si, 0.0) * sh[(li, si)]
            for li, si in wp]
    lo = [loic.get(si, 0.0) * sh[(li, si)] for li, si in wp]
    exact = list(appr)
    for k, (li, si) in enumerate(wp):
        if li in edge_inter:
            I = edge_inter[li]
            s = sum(v for (a, b), v in I.items() if a == si or b == si)
            exact[k] = dce[li][si] + 0.5 * s
    return {'first': first, 'shapley': exact,
            'shapley_approx_only': appr, 'loic': lo}, corr_mod, sh


def validate_approximation(dce, wp, edge_inter, corr_mod, sh):
    """FREE validation of the share approximation: on the consumers where the
    edge interactions were MEASURED, compare exact adjusted values against the
    module-tier approximation."""
    out = {}
    for li, I in edge_inter.items():
        idx = [k for k, (l2, _) in enumerate(wp) if l2 == li]
        if len(idx) < 3:
            out[f'consumer{li}'] = {'n_edges': len(idx),
                                    'note': 'too few edges to correlate'}
            continue
        ex, ap = [], []
        for k in idx:
            l2, si = wp[k]
            s = sum(v for (a, b), v in I.items() if a == si or b == si)
            ex.append(dce[l2][si] + 0.5 * s)
            ap.append(dce[l2][si] + corr_mod.get(si, 0.0) * sh[(l2, si)])
        out[f'consumer{li}'] = {
            'n_edges': len(idx),
            'spearman_exact_vs_approx': round(R2.spearman(ex, ap), 4),
            'max_abs_deviation': round(
                float(np.max(np.abs(np.array(ex) - np.array(ap)))), 5),
            'mean_abs_exact': round(float(np.mean(np.abs(ex))), 5)}
    return out


def shuffled_null(dce, wp, inter_mod, srcs, seed=0):
    """Permute the interaction matrix over pairs and rebuild the adjusted
    target: the correction becomes near-constant across sources, so the
    adjusted target must COLLAPSE toward first order."""
    keys = list(inter_mod)
    vals = np.array([inter_mod[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    first = [dce[li][si] for li, si in wp]
    sh = {}
    for si in {s for _, s in wp}:
        sh.update(shares(dce, si, wp))
    rs = []
    for _ in range(N_PERM):
        pv = vals[rng.permutation(len(vals))]
        im = {k: float(v) for k, v in zip(keys, pv)}
        corr = {i: 0.5 * sum(v for (a, b), v in im.items()
                             if a == i or b == i) for i in srcs}
        adj = [dce[li][si] + corr.get(si, 0.0) * sh[(li, si)]
               for li, si in wp]
        rs.append(R2.spearman(adj, first))
    corr_t = {i: 0.5 * sum(v for (a, b), v in inter_mod.items()
                           if a == i or b == i) for i in srcs}
    adj_t = [dce[li][si] + corr_t.get(si, 0.0) * sh[(li, si)]
             for li, si in wp]
    true_r = R2.spearman(adj_t, first)
    return {'n_permutations': N_PERM,
            'spearman_adjusted_vs_first_TRUE': round(true_r, 4),
            'spearman_adjusted_vs_first_SHUFFLED_mean': round(
                float(np.mean(rs)), 4),
            'spearman_adjusted_vs_first_SHUFFLED_p2.5_p97.5': [
                round(float(np.percentile(rs, 2.5)), 4),
                round(float(np.percentile(rs, 97.5)), 4)],
            'collapses_toward_first_order': bool(np.mean(rs) > true_r),
            'note': 'under a permutation every source draws its correction '
                    'from the same pool, so the correction is near-constant '
                    'across sources and the adjusted target must sit CLOSER '
                    'to first order than the true adjustment does'}


def seed_e26_cache(tag):
    """qk_e26.json's gate licenses reusing E26's cached pair CEs for the
    reference checkpoint (identical model, rows, ABL_N and substitution
    convention). Copy them into this run's cache under the E30 job ids."""
    src = E.loadj(E.jpath('qk_e26_partial.json'))
    if not src:
        return {}
    part = E.loadj(PARTIAL)
    moved = {}
    for k_src, k_dst in (('module_singles', f'{tag}:module_singles'),
                         ('module_pairs', f'{tag}:module_pairs'),
                         ('edge_pairs', f'{tag}:edge{DEPTH}')):
        if k_src in src and k_dst not in part:
            part[k_dst] = src[k_src]
            moved[k_dst] = len(src[k_src])
    if moved:
        json.dump(part, open(PARTIAL, 'w'))
        print(f"  seeded E30 cache from qk_e26_partial.json: {moved}",
              flush=True)
    return moved


def edge_loic(spec, m, Ws, base, means, rem, wp):
    """Direct EDGE-level LOIC (156 edges x K contexts) -- the strongest test of
    the module->edge share approximation. Reference checkpoint only."""
    tag = spec['tag']
    rng = np.random.default_rng(LOIC_SEED + 1)
    n = len(wp)
    ctxs = [sorted(rng.choice(n, size=n // 2, replace=False).tolist())
            for _ in range(K_LOIC)]

    def subs_of(idxs):
        d = {}
        for k in idxs:
            li, si = wp[k]
            d.setdefault(li, {})[si] = means[si]
        return d
    jobs = []
    for c, S in enumerate(ctxs):
        Ss = set(S)
        jobs.append((f'c{c}', S))
        for k in range(n):
            if k in Ss:
                jobs.append((f'c{c}_wo{k}', [x for x in S if x != k]))
            else:
                jobs.append((f'c{c}_wi{k}', sorted(S + [k])))
    ce = run_jobs(f'{tag}:edge_loic', jobs,
                  lambda p: (base if not p else ce_subs(m, subs_of(p), Ws)))
    per = [[] for _ in range(n)]
    for c, S in enumerate(ctxs):
        Ss, cs = set(S), ce[f'c{c}']
        for k in range(n):
            if k in Ss:
                per[k].append(cs - ce[f'c{c}_wo{k}'])
            else:
                per[k].append(ce[f'c{c}_wi{k}'] - cs)
    return [float(np.mean(p)) for p in per]


def run_ckpt(spec):
    key = f"ckpt_{spec['tag']}"
    if key in E.loadj(JP):
        print(f"{key}: done -- skip", flush=True)
        return
    if spec['stem'] is not None and not os.path.exists(E.ckpath(spec['stem'])):
        E.merge(JP, key, {'skipped': f"{spec['stem']}.pt missing"})
        return
    t0 = time.time()
    print(f"\n==== {spec['tag']} ({spec['stem']}, tier {spec['tier']}) ====",
          flush=True)
    m = load_ckpt(spec)
    Ws = m.wte.weight.shape[1]
    dims = [spec['slot']] * NG
    assert Ws == sum(dims), (spec['tag'], Ws, sum(dims))
    wp = E18U.wpairs(m, dims)

    # ---- first-order causal vector + STORED-SPEARMAN GATE (control 2) ----
    dce, lp, origin = causal_first_order(spec, m, Ws, wp)
    cau = [dce[li][si] for li, si in wp]
    G = E18U.gen_gram_table(m, dims)
    plain = E18U.score(G, wp)
    gate = {'origin_of_causal_vector': origin, 'n_edges': len(wp),
            'recomputed_plain_spearman': round(R2.spearman(plain, cau), 4)}
    if lp is not None:
        gate['stored_plain_spearman'] = lp['wiring_spearman_all']
        gate['abs_diff'] = round(abs(gate['recomputed_plain_spearman']
                                     - lp['wiring_spearman_all']), 6)
        gate['pass'] = bool(gate['abs_diff'] < GATE_TOL)
        assert gate['pass'] or E.SMOKE, \
            f"{spec['tag']} stored-Spearman gate FAILED: {gate}"
    else:
        gate['pass'] = None

    # ---- weight tables (plain + covariance-composed) ----
    tables, meta, vecs = E18U.composed_tables(
        m, dims, cau, wp, E.DEV, remnant=spec['remnant'], plain_vec=plain)
    weight_vecs = {'plain': vecs['plain'], 'cov_composed': vecs['cov'],
                   'cov_composed_readout_globalnorm': vecs['cov_ro']}

    # ---- interaction structure on THIS checkpoint ----
    base, means, rem = base_means_rem(m, Ws)
    if lp is not None and 'base_ce_fp32_abl_oldheld' in lp:
        gate['base_ce_recomputed'] = round(base, 5)
        gate['base_ce_stored'] = lp['base_ce_fp32_abl_oldheld']
        gate['base_ce_abs_diff'] = round(
            abs(base - lp['base_ce_fp32_abl_oldheld']), 6)
    srcs = list(range(7)) if E.SMOKE else list(range(NS))
    if spec['tag'] == 'E9a_recipe_s0' and not E.SMOKE:
        seed_e26_cache(spec['tag'])
    dmod, inter_mod = module_tier(spec, m, Ws, base, means, rem, srcs)
    loic, loic_sd, ctxs, _ = loic_module(spec, m, Ws, base, means, rem, srcs)

    # ---- measured edge maps at the top-N consumers (tier "full") ----
    tot_cons = {li: sum(dce[li][si] for si in dce[li] if si >= 1)
                for li in dce}
    order = sorted(tot_cons, key=lambda li: -tot_cons[li])
    edge_inter, measured = {}, []
    if spec['tier'] == 'full':
        for li in order:
            if len(measured) >= N_TOPC:
                break
            sl = sorted(si for (l2, si) in wp if l2 == li)
            if len(sl) < 2:
                continue
            edge_inter[li] = edge_tier(spec, m, Ws, base, means, rem,
                                       dce, li, sl)
            measured.append(li)

    targets, corr_mod, sh = build_targets(dce, wp, inter_mod, loic,
                                          edge_inter, srcs)
    approx_val = validate_approximation(dce, wp, edge_inter, corr_mod, sh)
    null = shuffled_null(dce, wp, inter_mod, srcs)

    # ---- optional: direct edge-level LOIC on the reference checkpoint ----
    eloic = None
    if spec['tag'] == 'E9a_recipe_s0' and not spec['remnant'] \
            and os.environ.get('QK_E30_EDGE_LOIC', '1') == '1':
        print("  reference checkpoint: direct EDGE-level LOIC "
              f"({len(wp)} edges x {K_LOIC} contexts) ...", flush=True)
        ev = edge_loic(spec, m, Ws, base, means, rem, wp)
        eloic = {
            'n_edges': len(wp), 'contexts': K_LOIC,
            'spearman_vs_module_mapped_loic': round(
                R2.spearman(ev, targets['loic']), 4),
            'spearman_vs_first_order': round(R2.spearman(ev, cau), 4),
            'spearman_vs_shapley': round(
                R2.spearman(ev, targets['shapley']), 4),
            'agreement_with_weight_tables': {
                nm: boot_ci(v, ev, seed=7)
                for nm, v in weight_vecs.items()},
            'note': 'measured directly at the edge tier: validates (or not) '
                    'the module->edge share mapping used for the LOIC target '
                    'on every other checkpoint'}
        targets['edge_loic_measured'] = ev

    # ---- the comparison numbers, every one with a bootstrap CI ----
    agree = {}
    for wnm, wv in weight_vecs.items():
        for tnm, tv in targets.items():
            agree[f'{wnm}__vs__{tnm}'] = boot_ci(wv, tv, seed=11)
    tvt = {a: {b: round(R2.spearman(targets[a], targets[b]), 4)
               for b in targets} for a in targets}

    rec = {
        'checkpoint': (f"{spec['stem']}.pt" if spec['stem'] else 'SMOKE'),
        'arm': spec['arm'], 'seed': spec['seed'], 'tier': spec['tier'],
        'slot_dim': spec['slot'], 'stream_width': Ws, 'remnant': spec['remnant'],
        'control_stored_spearman_gate': gate,
        'stored_wiring_tables': tables,
        'cov_meta': meta,
        'module_tier': {
            'sources': [sname(i) for i in srcs],
            'singles_dce': {sname(i): round(dmod[i], 6) for i in srcs},
            'interaction_matrix': {f'{sname(i)}|{sname(j)}': round(v, 6)
                                   for (i, j), v in inter_mod.items()},
            'shapley2_correction': {sname(i): round(corr_mod[i], 6)
                                    for i in srcs},
            'n_pairs': len(inter_mod)},
        'loic_module': {
            'K': K_LOIC, 'context_seed': LOIC_SEED,
            'contexts': [[sname(i) for i in S] for S in ctxs],
            'loic': {sname(i): round(loic[i], 6) for i in srcs},
            'loic_sd_over_contexts': {sname(i): round(loic_sd[i], 6)
                                      for i in srcs},
            'spearman_loic_vs_first_order_module': round(
                R2.spearman([loic[i] for i in srcs],
                            [dmod[i] for i in srcs]), 4),
            'spearman_loic_vs_shapley2_module': round(
                R2.spearman([loic[i] for i in srcs],
                            [dmod[i] + corr_mod[i] for i in srcs]), 4)},
        'measured_edge_consumers': [
            ('readout' if li == DEPTH else f'block{li}') for li in measured],
        'per_consumer_total_first_order_consumption': {
            ('readout' if li == DEPTH else f'block{li}'): round(v, 5)
            for li, v in tot_cons.items()},
        'edge_interaction_maps': {
            ('readout' if li == DEPTH else f'block{li}'): {
                f'{sname(i)}|{sname(j)}': round(v, 6)
                for (i, j), v in I.items()} for li, I in edge_inter.items()},
        'approximation_validation': approx_val,
        'shuffled_interaction_null': null,
        'target_vectors': {k: [round(x, 6) for x in v]
                           for k, v in targets.items()},
        'edge_order': [[('readout' if li == DEPTH else f'block{li}'),
                        sname(si)] for li, si in wp],
        'target_vs_target_spearman': tvt,
        'agreement': agree,
        'edge_loic_validation': eloic,
        'elapsed_s': round(time.time() - t0, 1)}
    E.merge(JP, key, rec)
    print(f"  {spec['tag']}: plain vs first "
          f"{agree['plain__vs__first']['spearman']}, plain vs shapley "
          f"{agree['plain__vs__shapley']['spearman']}, cov vs first "
          f"{agree['cov_composed__vs__first']['spearman']}, cov vs shapley "
          f"{agree['cov_composed__vs__shapley']['spearman']}, cov vs loic "
          f"{agree['cov_composed__vs__loic']['spearman']} "
          f"({rec['elapsed_s']:.0f}s)", flush=True)
    del m, G
    if E.DEV == 'cuda':
        torch.cuda.empty_cache()


# ============================ controls ======================================
def control_e26_reproduction():
    """Gate 1: this runner's own joint-ablation path must reproduce E26's
    stored module singles on qk_e9_a to 1e-6 -- that is what licenses reusing
    E26's cached pair CEs."""
    key = 'control_e26_reproduction'
    if E.SMOKE or E.loadj(JP).get(key, {}).get('pass'):
        print(f"{key}: cached/skip", flush=True)
        return
    e26 = E.loadj(E.jpath('qk_e26.json'))
    assert e26.get('control_singles_gate', {}).get('pass'), \
        'qk_e26.json singles gate not passed -- its map is unvalidated'
    m, _ = E.load_arm('qk_e9_a', E7R.make_e7m1)
    Ws = m.wte.weight.shape[1]
    base, means, rem = base_means_rem(m, Ws)
    stored = e26['module_tier']['singles_dce']
    diffs = {}
    for i in range(NS):
        v = ce_subs(m, module_subs(m, [i], means, rem), Ws) - base
        diffs[sname(i)] = abs(v - stored[sname(i)])
    mx = max(diffs.values())
    bd = abs(base - e26['singles_recomputed']['base_ce'])
    rec = {'n_sources': NS, 'base_ce_recomputed': base,
           'base_ce_stored_e26': e26['singles_recomputed']['base_ce'],
           'base_abs_diff': bd,
           'max_single_abs_diff': mx,
           'max_at': max(diffs, key=diffs.get),
           'tol': E26_TOL,
           'pass': bool(mx < E26_TOL and bd < E26_TOL),
           'note': 'licenses reusing qk_e26_partial.json pair CEs for the '
                   'reference checkpoint (same model, rows, ABL_N, '
                   'substitution convention)'}
    E.merge(JP, key, rec)
    print(f"{key}: max |diff| {mx:.2e} at {rec['max_at']}, base diff "
          f"{bd:.2e} -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'
    del m
    if E.DEV == 'cuda':
        torch.cuda.empty_cache()


def ensure_e22_probe():
    """qk_e22.json has NO wiring table: the E22 run died of OOM in the
    residual-census step BEFORE probe_e22a (see qk_e22_predbasis_run.out).
    Run that probe here (idempotent, its own module's code path) so the
    predicate-basis arm can enter the comparison at all."""
    if E.SMOKE:
        return
    j = E.loadj(E.jpath('qk_e22.json'))
    if 'light_probe_E22a_var_dims' in j and 'composed_wiring_E22a' in j:
        return
    if not os.path.exists(E.ckpath('qk_e22_a')):
        return
    print('qk_e22.json is missing its wiring table (E22 OOMed before the '
          'probe) -- running qk_e22_predbasis_run.probe_e22a now', flush=True)
    import qk_e22_predbasis_run as E22R
    E.DEV = 'cpu' if E.SMOKE else 'cuda'
    E22R.probe_e22a(15)
    if E.DEV == 'cuda':
        torch.cuda.empty_cache()


# ============================ summary =======================================
def summarize():
    j = E.loadj(JP)
    rows, targets_used = {}, ['first', 'shapley', 'shapley_approx_only',
                              'loic']
    for k, v in j.items():
        if not k.startswith('ckpt_') or v.get('skipped') or v.get('failed'):
            continue
        tag = k[5:]
        r = {'arm': v['arm'], 'seed': v['seed'], 'tier': v['tier'],
             'checkpoint': v['checkpoint']}
        for wnm in ('plain', 'cov_composed'):
            for t in targets_used:
                a = v['agreement'].get(f'{wnm}__vs__{t}')
                if a:
                    r[f'{wnm}_vs_{t}'] = a['spearman']
                    r[f'{wnm}_vs_{t}_ci95'] = a['ci95']
        r['delta_cov_shapley_minus_first'] = round(
            r.get('cov_composed_vs_shapley', 0)
            - r.get('cov_composed_vs_first', 0), 4)
        r['delta_cov_loic_minus_first'] = round(
            r.get('cov_composed_vs_loic', 0)
            - r.get('cov_composed_vs_first', 0), 4)
        r['delta_plain_shapley_minus_first'] = round(
            r.get('plain_vs_shapley', 0) - r.get('plain_vs_first', 0), 4)
        rows[tag] = r
    if not rows:
        return

    def ordering(field):
        return [t for t in sorted(rows, key=lambda t: -rows[t].get(field, -9))]
    ords = {f'by_{w}_vs_{t}': ordering(f'{w}_vs_{t}')
            for w in ('plain', 'cov_composed') for t in targets_used
            if any(f'{w}_vs_{t}' in r for r in rows.values())}
    ref = ords.get('by_cov_composed_vs_first', [])
    changes = {}
    for nm, o in ords.items():
        if nm == 'by_cov_composed_vs_first' or not ref:
            continue
        pos_ref = {t: i for i, t in enumerate(ref)}
        moves = {t: pos_ref[t] - o.index(t) for t in o if t in pos_ref}
        changes[nm] = {
            'ordering': o,
            'identical_to_first_order_ordering': bool(o == ref),
            'rank_moves_vs_cov_first_order': moves,
            'max_abs_rank_move': max(abs(x) for x in moves.values())
            if moves else 0,
            'spearman_of_the_two_orderings': round(R2.spearman(
                [pos_ref[t] for t in o], list(range(len(o)))), 4)}
    biggest = sorted(rows, key=lambda t: -abs(
        rows[t].get('delta_cov_shapley_minus_first', 0)))[:5]
    out = {
        'table': rows,
        'orderings': ords,
        'ordering_changes_vs_cov_first_order': changes,
        'arms_moving_most_cov_shapley_minus_first': [
            {'tag': t, 'delta': rows[t]['delta_cov_shapley_minus_first'],
             'cov_vs_first': rows[t].get('cov_composed_vs_first'),
             'cov_vs_shapley': rows[t].get('cov_composed_vs_shapley'),
             'ci95_first': rows[t].get('cov_composed_vs_first_ci95'),
             'ci95_shapley': rows[t].get('cov_composed_vs_shapley_ci95')}
            for t in biggest],
        'note': 'every Spearman carries a percentile bootstrap CI over the '
                '156 edges (reviewer-2 R1). Two Spearmen whose CIs overlap '
                'are a TIE; the ordering rows below are point-estimate '
                'orderings and must be read through those CIs.'}
    E.merge(JP, 'comparison_table', out)
    print(json.dumps(out['ordering_changes_vs_cov_first_order'],
                     indent=2)[:2500], flush=True)

    # ---- verdicts ----
    ov = [c['identical_to_first_order_ordering'] for c in changes.values()]
    ci_overlap = 0
    for t, r in rows.items():
        a, b = r.get('cov_composed_vs_first_ci95'), \
            r.get('cov_composed_vs_shapley_ci95')
        if a and b and not (a[1] < b[0] or b[1] < a[0]):
            ci_overlap += 1
    nulls = [j[k]['shuffled_interaction_null'] for k in j
             if k.startswith('ckpt_') and 'shuffled_interaction_null' in j[k]]
    E.merge(JP, 'verdicts', {
        'n_checkpoints_rescored': len(rows),
        'any_ordering_changes_under_an_adjusted_target':
            bool(not all(ov)) if ov else None,
        'orderings_identical_to_first_order': {nm: c[
            'identical_to_first_order_ordering'] for nm, c in changes.items()},
        'n_arms_whose_first_vs_shapley_CIs_OVERLAP': ci_overlap,
        'n_arms_total': len(rows),
        'shuffled_null_collapses_everywhere': bool(all(
            n['collapses_toward_first_order'] for n in nulls)) if nulls
        else None,
        'consequence': (
            'the arm ordering is not invariant to the causal target: '
            'readability rankings must name which causal target they use '
            'and carry the bootstrap CI' if ov and not all(ov) else
            'the arm ordering survives the interaction-adjusted target, so '
            'the mis-specification E26 found does not by itself overturn the '
            'ranking -- the seed spread (E27/E29) remains the binding '
            'problem')})


# ============================ main ==========================================
if __name__ == '__main__':
    E.setup()
    E.DEV = 'cpu' if E.SMOKE else 'cuda'
    if not E.SMOKE:
        e18 = E.loadj(E.jpath('qk_e18.json'))
        assert e18.get('gate1_uniform11_weight_support', {}).get('pass') \
            and e18.get('gate2_cov_composed_E9a', {}).get('pass'), \
            'qk_e18.json gates 1+2 not passed -- reused probes unvalidated'
    if 'design' not in E.loadj(JP):
        E.merge(JP, 'design', {
            'registered_before_computation': True,
            'motivation': 'E26 verdict causal_ground_truth_changes_materially '
                          '= TRUE: only 18% of module pairs are near-additive '
                          'and 148/300 are superadditive, so the first-order '
                          'single-ablation vector every wiring Spearman is '
                          'scored against is mis-specified',
            'estimator_A_shapley2': {
                'formula': 'adj(x) = dCE(x) + 0.5 * sum_{y != x} I(x, y)',
                'interpretation': 'importance in the INTACT model with each '
                                  'pairwise surplus split evenly between its '
                                  'two members; an EXTRAPOLATION that assumes '
                                  'the interaction expansion truncates at '
                                  'order 2 (exact only if triples and higher '
                                  'vanish)'},
            'estimator_B_loic': {
                'formula': 'loic(x) = mean over K=8 random half-contexts S of '
                           '[ CE(ablate S u {x}) - CE(ablate S \\ {x}) ]',
                'interpretation': 'importance in a HALF-DESTROYED model: the '
                                  'true marginal effect deep in the ablation '
                                  'lattice, capturing ALL interaction orders '
                                  'but at ONE coalition size and with '
                                  'context sampling noise',
                'difference': 'A and B answer different questions. A asks '
                              '"what is this module worth in the working '
                              'model, corrected for pairwise non-additivity"; '
                              'B asks "what is it worth when half the model '
                              'is already gone". Their disagreement IS the '
                              'third-and-higher-order content plus context '
                              'variance, so both are reported and neither is '
                              'promoted to THE target.'},
            'edge_extension': {
                'measured': f'true pairwise EDGE interactions at the {N_TOPC} '
                            'consumers with the largest total first-order '
                            'consumption, per checkpoint',
                'approximated': 'every other consumer: adj(li, si) = '
                                'dCE(li, si) + corr_mod(si) * share(li, si) '
                                'with share = the edge\'s relu\'d fraction of '
                                'the module\'s total first-order consumption',
                'LIMITATION': 'the share rule assumes a module\'s interaction '
                              'correction is carried by its edges in the same '
                              'proportion as its first-order consumption. '
                              'This is NOT verified for the approximated '
                              'consumers; it is checked only where the exact '
                              'edge map was measured, plus once against a '
                              'direct 156-edge LOIC on the reference '
                              'checkpoint.'},
            'bootstrap': f'B = {N_BOOT} percentile bootstrap over the 156 '
                         'edges for EVERY reported Spearman (reviewer-2 R1)',
            'controls': ['E26 module-singles reproduction to 1e-6 (licenses '
                         'reusing E26 cached pair CEs)',
                         'per-checkpoint stored-plain-Spearman gate to 1e-3',
                         'shuffled-interaction null must collapse the '
                         'adjusted target toward first order']})

    control_e26_reproduction()
    ensure_e22_probe()
    for sp in registry():
        if E.SMOKE:
            run_ckpt(sp)
            continue
        try:
            run_ckpt(sp)
        except Exception as ex:                     # one bad arm must not
            import traceback                        # kill the comparison
            traceback.print_exc()
            E.merge(JP, f"ckpt_{sp['tag']}",
                    {'failed': f'{type(ex).__name__}: {str(ex)[:300]}',
                     'checkpoint': f"{sp['stem']}.pt"})
            if E.DEV == 'cuda':
                torch.cuda.empty_cache()
    summarize()
    print('e30 interaction target run done', flush=True)
