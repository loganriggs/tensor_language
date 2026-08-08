"""ROUND-2 (independent) adversarial-review measurements for FINDING 11.

Each function here exists to attack one claim with a measurement, not an
argument.  Run:

    python tf_round2_review.py routing      # objection R1  (routing fairness)
    python tf_round2_review.py optim        # objection R2  (what else differs)
    python tf_round2_review.py lasso        # objection R3a (the lasso)
    python tf_round2_review.py codebook     # objection R3b (the codebook)
    python tf_round2_review.py shrink       # objection R3c (the shrink ranks)
    python tf_round2_review.py seeds        # objection R4  (one seed / one probe)

Results land in `tf_round2_measurements.json` (merged, one key per objection).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import tf_interp as I1
import tf_interp3 as I3
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/tf_round2_measurements.json'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

PRIMARY = ['vanilla', 'slots', 'bandwidth', 'predicate', 'codebook', 'shrink']


def stem_of(v, seed=0, suffix=''):
    return f'tf_{v}_d2_w128_b8192_s{seed}{suffix}'


def save(key, val):
    o = json.load(open(OUT)) if os.path.exists(OUT) else {}
    o[key] = val
    json.dump(o, open(OUT, 'w'), indent=2)
    print(f'-> {key} written to {OUT}')


# ===========================================================================
# R1  IS THE ROUTING MEASUREMENT ARCHITECTURE-FAIR?
# ===========================================================================
@torch.no_grad()
def routing(n_seq=32, T=256, batch=8, epss=(0.05, 0.1, 0.2)):
    """Objection: "delete layer-0 attention from layer-1's read" does not mean
    the same thing once the stream is partitioned.  In a slot model A0 owns a
    private quarter of the read, so deleting it displaces the read by ~127%; in
    the plain model the same deletion displaces it by 0.3%.  A KL ratio of 2e4
    could then be nothing but a PERTURBATION-SIZE ratio, and the comparison
    would be unfair by construction.

    The fair replacement is a MATCHED-DISPLACEMENT DIRECTIONAL PROBE, run in
    the model's own post-norm read space so no normalisation convention enters:

        hn        = layer 1's true (normalised) read
        d_X       = the part of hn that upstream module X contributes,
                    hn - norm(stream - X)            [a direction, unit-normed]
        hn(eps)   = hn + eps * ||hn|| * d_X          [the SAME 5/10/20% of the
                                                      read in every arm]
        score     = KL(model || model with layer 1 reading hn(eps))

    Every arm now receives an identical-size displacement, so whatever is left
    is sensitivity, not geometry.  Three reference directions make it scale-
    free in the other direction too: M0 (the other upstream module, which in a
    slot model owns an equally sized private slot), a global random direction,
    and a random direction confined to slot 0.  The reported statistic is the
    RATIO A0/random and A0/M0, which cannot be moved by any rescaling of
    anything.

    Also recorded: the gain of the ORIGINAL deletion measurement,
    KL / (relative read displacement)^2, which is the quadratic-response
    normalisation of the number FINDING 11 actually quotes."""
    res = {}
    g = torch.Generator(device=DEV).manual_seed(20260808)
    for v in PRIMARY:
        for seed in (0, 1, 2):
            stem = stem_of(v, seed)
            if not os.path.exists(f'{HERE}/{stem}.pt'):
                continue
            D = I3.VariantFold(stem)
            acc, ntok = {}, 0
            for x, y in I1.held_batches(D, n_seq, T, batch):
                P = D.run(x)
                lp = F.log_softmax(D.readout(P['r']).float(), -1)
                p = lp.exp()
                li = 1
                srcs = {'e': P['rem'][li], 'A0': P['A'][0], 'M0': P['M'][0]}
                pre = sum(srcs.values())
                hn = D._pre(2 * li, pre)
                hnorm = hn.norm(dim=-1, keepdim=True)
                dirs = {}
                uhn = hn / hnorm.clamp_min(1e-12)
                for nm in ('A0', 'M0', 'e'):
                    d = hn - D._pre(2 * li, pre - srcs[nm])
                    dirs[nm] = d
                    # A CONFOUND OF THE PROBE ITSELF: removing a write changes
                    # the read's MAGNITUDE as well as its direction, and a
                    # direction parallel to the read is just a rescale, which
                    # every model is sensitive to for reasons that have nothing
                    # to do with routing.  So the orthogonal component is
                    # reported beside it and the cosine is recorded.
                    dirs[nm + '_perp'] = d - (d * uhn).sum(-1, keepdim=True) * uhn
                    cs = float(((d * uhn).sum(-1)
                                / d.norm(dim=-1).clamp_min(1e-12)).abs().mean())
                    acc['cos_' + nm] = acc.get('cos_' + nm, 0.0) + cs * y.numel()
                dirs['rescale'] = uhn.expand_as(hn).clone()
                dirs['random'] = torch.randn(hn.shape, generator=g, device=DEV)
                r2 = torch.zeros_like(hn)
                r2[..., :D.s] = torch.randn(
                    (*hn.shape[:-1], D.s), generator=g, device=DEV)
                dirs['random_slot0'] = r2
                for nm, d in dirs.items():
                    u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    for eps in epss:
                        hn2 = hn + eps * hnorm * u
                        r = D.run(x, reads={li: (lambda P_, h=hn2: h)})['r']
                        q = F.log_softmax(D.readout(r).float(), -1)
                        k = f'{nm}_eps{eps}'
                        acc[k] = acc.get(k, 0.0) + float((p * (lp - q)).sum())
                # SECOND FLAVOUR, because the post-norm direction of a slot
                # model's A0 is 'slot 0 renormalised' (embedding chunk
                # included) while vanilla's is A0's bare write direction, and
                # a reviewer of this review would call that unfair.  Here the
                # injection is PRE-norm along each module's ACTUAL write
                # direction, at the same relative size, and the model's own
                # norm is then applied -- the same object in every arm.
                pnorm = pre.norm(dim=-1, keepdim=True)
                for nm in ('A0', 'M0', 'e'):
                    src = srcs[nm]          # NOT `v`: `v` is the variant name
                    u = src / src.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    for eps in epss:
                        hn2 = D._pre(2 * li, pre + eps * pnorm * u)
                        r = D.run(x, reads={li: (lambda P_, h=hn2: h)})['r']
                        q = F.log_softmax(D.readout(r).float(), -1)
                        k = f'pre_{nm}_eps{eps}'
                        acc[k] = acc.get(k, 0.0) + float((p * (lp - q)).sum())
                gr = torch.randn(pre.shape, generator=g, device=DEV)
                gr = gr / gr.norm(dim=-1, keepdim=True)
                for eps in epss:
                    hn2 = D._pre(2 * li, pre + eps * pnorm * gr)
                    r = D.run(x, reads={li: (lambda P_, h=hn2: h)})['r']
                    q = F.log_softmax(D.readout(r).float(), -1)
                    k = f'pre_random_eps{eps}'
                    acc[k] = acc.get(k, 0.0) + float((p * (lp - q)).sum())
                ntok += y.numel()
            row = {k: a / ntok for k, a in acc.items()}
            for eps in epss:
                for nm in ('A0', 'M0', 'e', 'A0_perp', 'M0_perp', 'e_perp',
                           'rescale', 'random_slot0'):
                    row[f'ratio_{nm}_over_random_eps{eps}'] = (
                        row[f'{nm}_eps{eps}'] / row[f'random_eps{eps}'])
                row[f'ratio_A0_over_M0_eps{eps}'] = (
                    row[f'A0_eps{eps}'] / row[f'M0_eps{eps}'])
                row[f'ratio_A0perp_over_M0perp_eps{eps}'] = (
                    row[f'A0_perp_eps{eps}'] / row[f'M0_perp_eps{eps}'])
                for nm in ('A0', 'M0', 'e'):
                    row[f'ratio_pre_{nm}_over_pre_random_eps{eps}'] = (
                        row[f'pre_{nm}_eps{eps}'] / row[f'pre_random_eps{eps}'])
            # the ORIGINAL number, gain-normalised
            j = json.load(open(f'{HERE}/{stem}_interp3.json'))
            ra = j['read_ablation_causal']
            for how in ('zero', 'resample'):
                for nm in ('A0', 'M0'):
                    kl = ra['kl_from_model'][f'l1_read_{how}_{nm}']
                    rr = ra['relative_change'][f'l1_read_{how}_{nm}_read_rel']
                    row[f'deletion_{how}_{nm}_kl'] = kl
                    row[f'deletion_{how}_{nm}_read_rel'] = rr
                    row[f'deletion_{how}_{nm}_gain_kl_over_readrel2'] = \
                        kl / max(rr ** 2, 1e-30)
            res[f'{v}_s{seed}'] = row
            print(f'{stem}: A0/rand {row["ratio_A0_over_random_eps0.1"]:6.3f} '
                  f'A0perp/rand {row["ratio_A0_perp_over_random_eps0.1"]:6.3f} '
                  f'A0perp/M0perp {row["ratio_A0perp_over_M0perp_eps0.1"]:6.3f} '
                  f'|cos(A0,read)| {row["cos_A0"]:.3f} '
                  f'rescale/rand {row["ratio_rescale_over_random_eps0.1"]:6.3f} '
                  f'gain(res) '
                  f'{row["deletion_resample_A0_gain_kl_over_readrel2"]:.4f}',
                  flush=True)
            del D
            torch.cuda.empty_cache()
    save('R1_routing_fairness', {
        'objection': 'the deletion measurement displaces layer-1\'s read by '
                     '0.3% in vanilla and 127% in a slot model, so the KL '
                     'ratio may be a perturbation-size ratio and not a '
                     'routing difference',
        'measurement': 'matched-displacement directional probe in the post-'
                       'norm read space (eps = 5/10/20% of ||read|| along each '
                       'upstream module\'s own direction, and along random '
                       'directions), plus the gain-normalisation KL/readrel^2 '
                       'of the original deletion number',
        'pre_norm_flavour_is_recorded_but_NOT_fair': (
            'the `pre_*` keys inject the same fraction of the PRE-norm stream '
            "norm along each module's raw write direction.  They are kept as a "
            'diagnostic and must not be quoted as the fair comparison: in a '
            'slot model the raw slot norms are wildly unequal (per-slot RMSNorm '
            'equalises them only afterwards), so a perturbation sized by the '
            'TOTAL norm is enormous relative to a small slot and negligible '
            'relative to a large one -- a random pre-norm direction costs '
            '0.0034 nats in vanilla and 0.13-0.62 in the variants.  That is the '
            'very confound this objection is about, now visible in a second '
            'place.  The post-norm probe is the fair one because it displaces '
            "the actual read by the same relative amount in every arm."),
        'cells': res})


# ===========================================================================
# R2  WHAT ELSE DIFFERS BETWEEN PLAIN AND VARIANT?
# ===========================================================================
def optim():
    """Objection: with the learning-rate confound dead, the induction gap could
    still come from something other than the architecture -- initialisation
    scale, the effective per-block learning rate, embedding capacity, or the
    number of directions training actually moves.

    Measured per parameter block, for every arm, by re-instantiating the model
    at its OWN config and seed (so the init is bit-reproducible) and diffing
    against the trained weights:
      * init RMS               -- initialisation scale
      * ||W_final - W_init||_F / ||W_init||_F  -- distance travelled, the
        observable proxy for an effective per-block learning rate
      * effective rank (participation ratio of the singular values) of both
        W_init and W_final - W_init -- how many directions training used
      * the optimizer each block was assigned."""
    import tf_train as TT
    import tf_fold
    rows = {}
    arms = [(v, s, '') for v in PRIMARY for s in (0, 1, 2)]
    arms += [('slots', 0, '_writeinit_only'), ('slots', 0, '_nolasso')]
    for v, s, suf in arms:
        stem = stem_of(v, s, suf)
        if not os.path.exists(f'{HERE}/{stem}.pt'):
            continue
        model, cfg, ck = tf_fold.load_checkpoint(stem, DEV)
        fresh = M.TinyBilin(cfg).to(DEV)           # same cfg, same seed -> same init
        fin = dict(model.named_parameters())
        ini = dict(fresh.named_parameters())
        blocks = {}
        for n, p in fin.items():
            if n not in ini or p.ndim < 2:
                continue
            W0, W1 = ini[n].detach().float(), p.detach().float()
            d = W1 - W0
            def erank(A):
                sv = torch.linalg.svdvals(A.reshape(A.shape[0], -1))
                sv = sv / sv.sum().clamp_min(1e-30)
                return float(torch.exp(-(sv * (sv + 1e-30).log()).sum()))
            blocks[n] = {
                'shape': list(p.shape),
                'init_rms': float(W0.pow(2).mean().sqrt()),
                'final_rms': float(W1.pow(2).mean().sqrt()),
                'rel_distance_travelled': float(d.norm() / W0.norm().clamp_min(1e-12))
                if float(W0.norm()) > 0 else float('inf'),
                'abs_distance_travelled': float(d.norm()),
                'erank_init': erank(W0),
                'erank_delta': erank(d),
                'erank_final': erank(W1),
                'max_erank': min(W1.shape[0], int(np.prod(W1.shape[1:]))),
            }
        rows[stem] = blocks
        print(f'{stem}: {len(blocks)} matrix blocks', flush=True)
        del model, fresh
        torch.cuda.empty_cache()
    # the optimizer assignment, read off the training code rather than assumed
    opt_note = TT.optimizer_note() if hasattr(TT, 'optimizer_note') else None
    save('R2_what_else_differs', {
        'objection': 'initialisation scale / effective per-block learning rate '
                     '/ embedding capacity / number of trainable directions '
                     'used could produce the induction gap instead of the '
                     'architecture',
        'measurement': 'per-parameter-block init RMS, relative distance '
                       'travelled from init, and effective rank (participation '
                       'ratio of singular values) of the init and of the '
                       'update, for every arm',
        'optimizer_note': opt_note,
        'blocks': rows})


# ===========================================================================
# R3a  IS "THE LASSO PRUNES NOTHING" A COEFFICIENT ARTIFACT?
# ===========================================================================
@torch.no_grad()
def lasso():
    """Objection: the group lasso is reported as failing to prune, but 3e-5 was
    inherited from another program at another scale.  If the penalty is
    numerically negligible against the cross-entropy, "the lasso prunes
    nothing" says nothing about slot architectures -- it says the coefficient
    was too small, and the finding is an artifact of our own configuration.

    Measurement 1 (arithmetic, decisive on its own): evaluate the penalty term
    at the trained checkpoint and compare `coeff * penalty` with the CE it was
    added to, and with the gradient each contributes.
    Measurement 2 (the retrain, tf_round2_chain.sh): sweep the coefficient over
    three decades and read `mean_live_slots_per_read` and CE off each arm."""
    import tf_fold
    rows = {}
    for stem in sorted(os.listdir(HERE)):
        if not stem.endswith('.pt'):
            continue
        stem = stem[:-3]
        if 'slots' not in stem or '_d2_w128' not in stem:
            continue
        model, cfg, ck = tf_fold.load_checkpoint(stem, DEV)
        if cfg.n_slots < 2:
            del model
            continue
        pen = float(model.group_penalty())
        ce = ck.get('final_held_ce') or json.load(
            open(f'{HERE}/{stem}.json'))['run']['final_held_ce']
        mech = None
        ip = f'{HERE}/{stem}_interp3.json'
        if os.path.exists(ip):
            mech = json.load(open(ip)).get('mechanism', {})
        rows[stem] = {
            'group_coeff': cfg.group_coeff,
            'group_penalty_at_checkpoint': pen,
            'coeff_times_penalty': cfg.group_coeff * pen,
            'held_ce': ce,
            'penalty_as_fraction_of_ce': cfg.group_coeff * pen / ce,
            'mean_live_slots_per_read': (mech or {}).get(
                'mean_live_slots_per_read'),
            'min_group_share': (mech or {}).get('min_group_share'),
        }
        print(f'{stem:48s} coeff {cfg.group_coeff:<8g} pen {pen:9.3f} '
              f'coeff*pen {cfg.group_coeff*pen:9.5f} '
              f'({100*cfg.group_coeff*pen/ce:.3f}% of CE)  live '
              f'{rows[stem]["mean_live_slots_per_read"]}', flush=True)
        del model
        torch.cuda.empty_cache()
    save('R3a_lasso_coefficient', {
        'objection': '"the lasso prunes nothing" may be a coefficient artifact',
        'measurement': 'penalty magnitude at the trained checkpoint against '
                       'the CE it is added to, plus a retrained sweep of the '
                       'coefficient over three decades',
        'arms': rows})


# ===========================================================================
# R3b  IS THE FLAT CODEBOOK AN ARTIFACT OF THE CODEBOOK, OR OF THE DATA?
# ===========================================================================
@torch.no_grad()
def codebook(n_seq=24, T=256, batch=8, Ks=(64, 256, 1024)):
    """Objection: "all 256 atoms used near-uniformly at 22-39% injected error"
    is reported as the codebook failing to buy legibility.  But a dictionary
    forced to spread is also exactly what you get when the thing being
    quantised has no low-dimensional discrete structure -- in which case the
    finding is about the ACTIVATIONS, not the mechanism, and reporting it as a
    failure of the mechanism is unfair to it (and, symmetrically, blaming the
    data lets the mechanism off).  Distinguish them, using the model's OWN
    quantiser so that like is compared with like.

    Measurement: take the activations the trained codebook variant actually
    quantises (post slot-norm, per consumer, per written slot), and run
    `mp_quantize` -- the same k-step matching pursuit at the same k -- against
    four dictionaries: the model's own trained codebook, a k-means dictionary
    of the same size fit on a TRAIN half of those activations, k-means at 1/4
    and 4x the size, and the random unit-norm dictionary the model started
    from.  Everything is scored on a HELD half.  Also reported: a
    same-budget CONTINUOUS alternative (PCA to k dimensions, the same number
    of scalars per token the code carries), which is the "beat a
    same-parameter-count alternative" the README requires.

    Reading: if k-means at 256 is no better than the trained codebook and 1024
    atoms barely help, the flatness and the error are properties of the
    activation distribution and the finding stands.  If k-means is much better,
    the EMA codebook is under-trained and the finding is retracted."""
    import tf_fold
    res = {}
    for seed in (0, 1):
        stem = stem_of('codebook', seed)
        if not os.path.exists(f'{HERE}/{stem}.pt'):
            continue
        D = I3.VariantFold(stem)
        model, cfg = D.model, D.cfg
        S, G = cfg.slot, cfg.n_slots
        store = {}
        for x, y in I1.held_batches(D, n_seq, T, batch):
            P = D.run(x)
            # consumer 2*li reads the stream before block li; 2*li+1 the stream
            # after that block's attention.  Both are quantised on the slots
            # already written.
            pre = {0: P['rem'][0], 1: P['pre_mlp'][0]}
            if D.L > 1:
                pre[2] = P['rem'][1] + P['A'][0] + P['M'][0]
                pre[3] = P['pre_mlp'][1]
            for k, z in pre.items():
                if k == 0:
                    continue                       # nothing written yet
                zz = model.slot_norm(z)
                for j in range(k):                 # the slots this consumer quantises
                    store.setdefault((k, j), []).append(
                        zz[..., j * S:(j + 1) * S].reshape(-1, S))
        cell = {}
        for (k, j), chunks in sorted(store.items()):
            A = torch.cat(chunks)
            n = A.shape[0]
            perm = torch.randperm(n, device=A.device)
            tr, he = A[perm[:n // 2]], A[perm[n // 2:]]
            ks = model.qz_ksteps[k]
            row = {'k_matching_pursuit_steps': ks, 'n_held_vectors': he.shape[0]}

            def score(C, tag):
                C = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                recon, idxs, alphas, _ = M.mp_quantize(he, C, ks)
                cnt = torch.bincount(torch.cat(idxs), minlength=C.shape[0]).float()
                pr = cnt / cnt.sum()
                ent = float(-(pr[pr > 0] * pr[pr > 0].log()).sum())
                srt = pr.sort(descending=True).values.cumsum(0)
                row[tag] = {
                    'n_atoms': int(C.shape[0]),
                    'rel_error_held': float((he - recon).norm() / he.norm()),
                    'usage_entropy_nats': ent,
                    'max_entropy_nats': float(np.log(C.shape[0])),
                    'entropy_fraction': ent / float(np.log(C.shape[0])),
                    'atoms_for_90pct': int((srt < 0.9).sum()) + 1}

            score(model.qz_codebook[k].float(), 'trained_codebook')
            g = torch.Generator(device=A.device).manual_seed(4242 + k)
            score(torch.randn(cfg.cb_n, S, generator=g, device=A.device),
                  'random_unit_dictionary_same_size')
            for K in Ks:
                C = tr[torch.randperm(tr.shape[0], device=A.device)[:K]].clone()
                for _ in range(20):
                    a = torch.cdist(tr, C).argmin(1)
                    for q in range(K):
                        m = a == q
                        if m.any():
                            C[q] = tr[m].mean(0)
                score(C, f'kmeans_K{K}')
            # same-budget continuous alternative: PCA to ks dimensions
            Xc = tr - tr.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            for r in (ks, 2 * ks):
                B = Vh[:r]
                rec = (he - tr.mean(0, keepdim=True)) @ B.t() @ B \
                    + tr.mean(0, keepdim=True)
                row[f'pca_r{r}_same_budget'] = {
                    'rel_error_held': float((he - rec).norm() / he.norm())}
            cell[f'consumer{k}_slot{j}'] = row
            print(f'{stem} c{k}s{j} (k={ks}): trained '
                  f'{row["trained_codebook"]["rel_error_held"]:.3f} | kmeans256 '
                  f'{row["kmeans_K256"]["rel_error_held"]:.3f} | kmeans1024 '
                  f'{row["kmeans_K1024"]["rel_error_held"]:.3f} | pca{ks} '
                  f'{row[f"pca_r{ks}_same_budget"]["rel_error_held"]:.3f}',
                  flush=True)
        res[stem] = cell
        del D
        torch.cuda.empty_cache()
    save('R3b_codebook_flatness', {
        'objection': 'the flat 256-atom usage and the 22-39% injected error '
                     'may be an artifact of the EMA codebook mechanism rather '
                     'than a fact about the activations',
        'measurement': "the model's own matching-pursuit quantiser at the same "
                       'k, run against the trained codebook, k-means '
                       'dictionaries at 64/256/1024 fit on a train half, the '
                       'random dictionary the model started from, and a '
                       'same-budget PCA alternative; all scored on a held half',
        'cells': res})


# ===========================================================================
# R3c  ARE THE SHRINK PROJECTIONS REALLY FULL RANK -- CAUSALLY?
# ===========================================================================
@torch.no_grad()
def shrink(n_seq=32, T=256, batch=8):
    """Objection: "the shrink projections are near full rank" is an ENTROPY-RANK
    statement about a spectrum.  Entropy rank is not a causal measure: a
    projection can have a flat spectrum and still be causally compressible if
    the directions with small singular values do no work.  Reporting the
    architecture as having failed on the strength of a spectral statistic is
    exactly the "arithmetic dressed as a finding" failure mode, in reverse.

    Measurement: truncate the remnant projections to their top-r singular
    directions, r = 1..full, and score the held KL from the untruncated model.
    Beside it, the same truncation to r RANDOM directions of the same
    projection (the same rank budget, no spectral information), which is the
    same-parameter-count alternative the README demands.  If the model's own
    top-r reaches a small KL well below full rank, the projections are
    causally low rank and the entropy-rank claim is retracted."""
    res = {}
    for seed in (0, 1, 2):
        stem = stem_of('shrink', seed)
        if not os.path.exists(f'{HERE}/{stem}.pt'):
            continue
        D = I3.VariantFold(stem)
        model = D.model
        # W_rem is consumed LIVE by model.remnants(), not baked into any fold,
        # so editing it in place is enough -- no refold is needed and none is
        # silently skipped.
        names = [n for n, _ in model.named_parameters()
                 if n.startswith('W_rem') and n.endswith('weight')]
        assert names, [n for n, _ in model.named_parameters()]
        base = list(I1.held_batches(D, n_seq, T, batch))
        ref, ntok = [], 0
        for x, y in base:
            ref.append(F.log_softmax(D.readout(D.run(x)['r']).float(), -1))
            ntok += y.numel()

        def kl_now():
            k = 0.0
            for (x, _), lpref in zip(base, ref):
                q = F.log_softmax(D.readout(D.run(x)['r']).float(), -1)
                k += float((lpref.exp() * (lpref - q)).sum())
            return k / ntok

        rows = {}
        for nm in names:
            p = dict(model.named_parameters())[nm]
            W = p.data.clone()
            U, S_, Vh = torch.linalg.svd(W, full_matrices=False)
            full = int(S_.numel())
            g = torch.Generator(device=W.device).manual_seed(7 + seed)
            curve, rcurve = {}, {}
            for r in sorted({1, 2, 4, 8, 12, 16, 24, 32, 48, full}):
                if r > full:
                    continue
                p.data = (U[:, :r] * S_[:r]) @ Vh[:r]
                curve[r] = kl_now()
                Q = torch.linalg.qr(torch.randn(W.shape[1], r, device=W.device,
                                                generator=g))[0]
                p.data = W @ Q @ Q.t()
                rcurve[r] = kl_now()
            p.data = W.clone()
            sv = S_ / S_.sum()
            rows[nm] = {
                'full_rank': full,
                'entropy_rank': float(torch.exp(
                    -(sv * (sv + 1e-30).log()).sum())),
                'topr_kl': curve, 'random_subspace_kl': rcurve,
                'rank_for_kl_below_0.01': min(
                    [r for r in sorted(curve) if curve[r] < 0.01],
                    default=None),
                'rank_for_kl_below_0.05': min(
                    [r for r in sorted(curve) if curve[r] < 0.05],
                    default=None)}
            print(f'{stem} {nm} (full {full}): ' + '  '.join(
                f'r{r}={curve[r]:.3f}' for r in sorted(curve)), flush=True)
        res[stem] = rows
        del D
        torch.cuda.empty_cache()
    save('R3c_shrink_causal_rank', {
        'objection': '"the shrink projections are near full rank" is a '
                     'spectral statistic, not a causal one',
        'measurement': 'top-r truncation of each remnant projection scored as '
                       'held KL from the untruncated model, against a random '
                       'subspace of the same rank',
        'cells': res})


# ===========================================================================
# R4  WHAT RESTS ON ONE SEED / ONE PROBE / ONE ABLATION
# ===========================================================================
def seeds():
    """Objection: the README's standing failure mode.  Enumerate every claim in
    FINDING 11 and record how many seeds, probes and ablation flavours actually
    stand behind it, reading the JSONs rather than the write-up."""
    res = {}
    for v in PRIMARY:
        per = {}
        for s in (0, 1, 2):
            p = f'{HERE}/{stem_of(v, s)}_interp3.json'
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            rs = j.get('induction_route_split', {})
            L = j['rung5_ladder']
            per[f's{s}'] = {
                'induction': j['rung3_induction']['induction_score_mean'],
                'induction_floor_3se': j['induction_power'][
                    'detectable_effect_floor_nats_3se'],
                'above_floor': (j['rung3_induction']['induction_score_mean']
                                > j['induction_power'][
                                    'detectable_effect_floor_nats_3se']),
                'bag_mean': j['rung3_induction']['bag_score_mean'],
                'natural_swap': j['natural_induction'][
                    'ORDER_ONLY_patch_swap']['mean'],
                'natural_swap_t': j['natural_induction'][
                    'ORDER_ONLY_patch_swap']['t'],
                'route_split': {k: rs[k] for k in rs
                                if 'fraction' in k or 'removed' in k},
                'ladder_attn_vs_mlp': [L['no_attention_at_all']['kl_from_model'],
                                       L['no_mlp']['kl_from_model']],
                'attention_beats_mlp': (L['no_attention_at_all']['kl_from_model']
                                        > L['no_mlp']['kl_from_model']),
                'mean_live_slots_per_read': j.get('mechanism', {}).get(
                    'mean_live_slots_per_read'),
            }
        res[v] = per
    save('R4_seed_support', {
        'objection': 'which FINDING 11 claims rest on one seed, one probe or '
                     'one ablation flavour',
        'measurement': 'per-seed read-out of induction (with its own power '
                       'floor), the natural-text swap, the route split and the '
                       'ladder attention-vs-MLP ordering, straight from the '
                       'cell JSONs',
        'per_variant': res})


# ===========================================================================
# R5  IS THE CONTENT-SPECTRUM NULL RESULT COMING FROM AN UNCALIBRATED DETECTOR?
# ===========================================================================
@torch.no_grad()
def content():
    """Objection (the README's own standing failure mode, applied to our own
    unflattering-looking-but-actually-flattering claim): "no variant moves
    content off its spectral null, 0.98-1.00 in all six" is a NULL RESULT, and
    a null result is worth nothing until the detector is shown to detect a
    planted effect of known size.

    Calibration: plant MLP tensors whose content genuinely IS low dimensional
    -- the bilinear factors restricted to an r-dimensional input subspace,
    r = 2/4/8/16/32 -- into the same statistic with the same shape-matched
    random-factored null, at each variant's own tensor shape.  If the ratio
    falls with r, the detector has power and the 0.98-1.00 readings mean what
    they say.  If it stays near 1, the statistic is blind and the claim is
    retracted.

    (In-sample corroboration that already exists: the SAME entropy-rank-over-
    null statistic reads 0.19-0.36 on the layer-0 selection tables, so it is
    demonstrably not pinned at 1.  The planted sweep makes that quantitative.)"""
    import tf_fold
    res = {}
    for v in PRIMARY:
        stem = stem_of(v, 0)
        if not os.path.exists(f'{HERE}/{stem}.pt'):
            continue
        D = I3.VariantFold(stem)
        T0 = D.Tl[0]
        O, Ws, hid = T0.shape[0], D.Ws, D.cfg.hidden
        k = 1                                   # layer-0 MLP consumer index
        if D.cfg.n_slots > 1 and not D.cfg.small_dec:
            live = list(range(D.s * k, D.s * (k + 1)))
        else:
            live = list(range(O))
        OL = len(live)
        gl = torch.Generator(device='cpu').manual_seed(5)
        Dnl = torch.randn(OL, hid, generator=gl).to(DEV)
        Lrl = torch.randn(hid, Ws, generator=gl).to(DEV)
        Rrl = torch.randn(hid, Ws, generator=gl).to(DEV)
        Mnl = torch.einsum('of,fi,fj->oij', Dnl, Lrl, Rrl)
        Tnl = 0.5 * (Mnl + Mnl.transpose(1, 2))
        null = tf_fold.eff_rank(torch.linalg.svdvals(
            Tnl.reshape(OL, -1).double()).cpu().numpy())['entropy_rank']
        row = {'null_entropy_rank': null, 'live_rows': OL,
               'measured_model': tf_fold.eff_rank(torch.linalg.svdvals(
                   T0[live].reshape(OL, -1).double()).cpu().numpy()
               )['entropy_rank'] / null}
        g2 = torch.Generator(device='cpu').manual_seed(11)
        for r in (2, 4, 8, 16, 32, Ws):
            if r > Ws:
                continue
            P = torch.randn(r, Ws, generator=g2).to(DEV)
            Lp = torch.randn(hid, r, generator=g2).to(DEV) @ P
            Rp = torch.randn(hid, r, generator=g2).to(DEV) @ P
            Dp = torch.randn(OL, hid, generator=g2).to(DEV)
            Mp = torch.einsum('of,fi,fj->oij', Dp, Lp, Rp)
            Tp = 0.5 * (Mp + Mp.transpose(1, 2))
            er = tf_fold.eff_rank(torch.linalg.svdvals(
                Tp.reshape(OL, -1).double()).cpu().numpy())['entropy_rank']
            row[f'planted_input_rank_{r}'] = er / null
        res[v] = row
        print(f'{v:10s} model {row["measured_model"]:.3f} | ' + '  '.join(
            f'r{r}={row[f"planted_input_rank_{r}"]:.3f}'
            for r in (2, 4, 8, 16, 32) if f'planted_input_rank_{r}' in row),
            flush=True)
        del D
        torch.cuda.empty_cache()
    save('R5_content_detector_power', {
        'objection': '"content sits on its spectral null in all six" is a null '
                     'result from a detector never shown to have power',
        'measurement': 'planted low-input-rank bilinear content tensors at each '
                       "variant's own shape, scored by the same entropy-rank-"
                       'over-null statistic',
        'cells': res})


if __name__ == '__main__':
    fn = {'routing': routing, 'optim': optim, 'lasso': lasso,
          'codebook': codebook, 'shrink': shrink, 'seeds': seeds,
          'content': content}
    for a in sys.argv[1:]:
        fn[a]()
