# ratio_docsplit: IS THE RATIO RESULT AN ARTIFACT OF DOCUMENT CLUSTERING?
#
# Codex found (15:30) that their 192 serialized rows were only 64 independent source
# documents, so intervals computed at the row unit overstated confidence, and they
# reaggregated at the true document unit. That failure mode applies to any row-based
# scoring, so it is worth checking against my own headline number rather than waiting
# to have it found for me.
#
# MEASURED FIRST: the canonical .rowcache/fineweb_n480_skip80.pt is 480 rows drawn
# from 209 unique source documents (2.3 rows/document). Clustering is real.
#
# THE ARGUMENT THAT IT DOES NOT INFLATE MY p-VALUES, which this run tests rather than
# assumes: my predictor |lam1/lam2| is computed from WEIGHTS ALONE and has no
# dependence on documents, rows or tokens whatsoever. Document-level noise enters only
# the OUTCOME (relative CE rise). Noise in an outcome that is independent of the
# predictor attenuates a correlation -- it cannot manufacture rank agreement with a
# document-independent predictor. So clustering should make my permutation p-values
# CONSERVATIVE, not anti-conservative. If that reasoning is right, splitting the
# documents into disjoint halves should give two similar rho values, both positive.
#
# DESIGN: the 209 documents are partitioned into two disjoint halves by a fixed hash of
# document_id. Each class's relative CE rise is computed separately on each half's
# rows, and rho is computed within each half. No row appears in both halves and no
# document straddles them. S1648's twelve type-spanning classes at mlp11, rank-2
# mean-ablation -- the exact configuration behind the pooled headline.
#
# Registered predictions:
#   pred_a BOTH HALVES ARE POSITIVE: rho on half A and rho on half B are both >= +.20.
#   pred_b THEY AGREE: |rho_A - rho_B| <= .35, i.e. the relationship is not carried by
#          one document subset.
#   pred_c ATTENUATION, NOT INFLATION: the mean of the two half-rhos is BELOW the
#          full-data rho of +.5105, as regression dilution on smaller samples predicts.
#          If a half-rho instead EXCEEDS the full value substantially, the clustering
#          argument above is wrong and the headline needs recomputing at document unit.
import json, time, sys, os, re, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152; T = 256
SITE = 11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ratio_docsplit_results.json'
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
NROWS = 480
FULL_RHO = 0.5105               # §1648, same twelve classes, all 480 rows
RECEIPT_JSON = PT + '.rowcache/fineweb_oracle_v2_receipt.json'

# twelve TYPE-SPANNING classes; none has ever had its CE rise measured
PATS = {'exclaim': r'^!$|^ !$', 'semicolon': r'^;$|^ ;$', 'colon': r'^:$|^ :$',
        'quote': r'^"$|^ "$', 'dash': r'^-$|^ -$', 'digit': r'^ ?[0-9]+$',
        'cap': r'^ [A-Z][a-z]+$', 'we': r'^ (we|We)$', 'you': r'^ (you|You)$',
        'it': r'^ (it|It)$', 'an': r'^ an$', 'my': r'^ my$'}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def slice_and_eigs(mask_v):
    """Top-RANK |lambda| eigenpair of the class-projected quadratic. Weights only."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:RANK]
    return V[:, o].contiguous(), [float(lam[i]) for i in o]


def mk_pre_hook(V2, mu):
    def pre(mod, args):
        f = args[0].float()
        p = f @ V2
        return ((f - (p - mu) @ V2.T).to(args[0].dtype),) + tuple(args[1:])
    return pre


@torch.no_grad()
def ce_pass(rows, mask_v, V2, mu):
    """Mean CE on the class's own positions. mu=None captures the global slice mean."""
    cap = {}
    if mu is None:
        def h(mod, args):
            f = args[0].float().reshape(-1, D)
            cap['s'] = cap.get('s', torch.zeros(RANK, device=DEV)) + (f @ V2).sum(0)
            cap['n'] = cap.get('n', 0) + f.shape[0]
            return None
        handle = H[SITE].mlp.register_forward_pre_hook(h)
    else:
        handle = H[SITE].mlp.register_forward_pre_hook(mk_pre_hook(V2, mu))
    tot, npos = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                 reduction='none').reshape(tg.shape)
            pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
            tot += float(ce[pm].sum()); npos += int(pm.sum())
    finally:
        handle.remove()
    return tot / max(npos, 1), npos, ((cap['s'] / max(cap['n'], 1)) if cap else None)


def doc_halves():
    """Partition the 480 rows into two DOCUMENT-disjoint halves by a fixed hash of
    document_id, so no document straddles the split."""
    import hashlib
    prov = json.load(open(RECEIPT_JSON))['document_provenance']['sets']['n480_skip80']
    assert len(prov) >= NROWS, f'provenance has {len(prov)} rows, need {NROWS}'
    docs = sorted({e['document_id'] for e in prov[:NROWS]})
    side = {d: (int(hashlib.sha256(d.encode()).hexdigest()[:8], 16) & 1) for d in docs}
    A = [i for i, e in enumerate(prov[:NROWS]) if side[e['document_id']] == 0]
    B = [i for i, e in enumerate(prov[:NROWS]) if side[e['document_id']] == 1]
    dA = {e['document_id'] for i, e in enumerate(prov[:NROWS]) if i in set(A)}
    dB = {e['document_id'] for i, e in enumerate(prov[:NROWS]) if i in set(B)}
    assert not (dA & dB), 'documents straddle the split'
    return A, B, len(dA), len(dB)


@torch.no_grad()
def main():
    import hashlib
    t0 = time.time()
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    rows = raw[:NROWS, :T + 1].contiguous()
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'DOCUMENT-SPLIT: {len(PATS)} classes scored on two document-disjoint halves. {NROWS} canonical '
          f'rows (receipt {rh}). Single registered hypothesis: |lam1/lam2|.', flush=True)

    A, B, nda, ndb = doc_halves()
    print(f'document-disjoint split: half A {len(A)} rows / {nda} docs | '
          f'half B {len(B)} rows / {ndb} docs | no document straddles', flush=True)
    rowsA = rows[torch.tensor(A)]
    rowsB = rows[torch.tensor(B)]

    per = {}
    for c, pat in PATS.items():
        mask_v = rx(pat)
        V2, ev = slice_and_eigs(mask_v)
        ratio = abs(ev[0]) / max(abs(ev[1]), 1e-12)
        rec = {'ratio': round(ratio, 4)}
        for lbl, rr in (('A', rowsA), ('B', rowsB)):
            base, npos, mu = ce_pass(rr, mask_v, V2, None)
            abl, _, _ = ce_pass(rr, mask_v, V2, mu)
            rec[f'rel_rise_{lbl}'] = round((abl - base) / base if base > 0 else 0.0, 5)
            rec[f'n_positions_{lbl}'] = npos
        per[c] = rec
        print(f"  {c:10s} ratio {ratio:6.3f} | A {rec['rel_rise_A']:+.5f} (n={rec['n_positions_A']:5d})"
              f" | B {rec['rel_rise_B']:+.5f} (n={rec['n_positions_B']:5d})", flush=True)

    ks = list(per); n = len(ks)

    def rk(f):
        o = sorted(ks, key=lambda z: -f(per[z])); return [o.index(z) + 1 for z in ks]

    def rho(a, b):
        return 1 - 6 * sum((a[i] - b[i]) ** 2 for i in range(n)) / (n * (n * n - 1))

    rr = rk(lambda v: v['ratio'])
    rA = rho(rr, rk(lambda v: v['rel_rise_A']))
    rB = rho(rr, rk(lambda v: v['rel_rise_B']))
    mean_half = (rA + rB) / 2

    pa = rA >= 0.20 and rB >= 0.20
    pb = abs(rA - rB) <= 0.35
    pc = mean_half < FULL_RHO

    print(f'\n  rho(|lam1/lam2|, rel CE rise)  half A = {rA:+.4f}   half B = {rB:+.4f}', flush=True)
    print(f'  |difference| = {abs(rA-rB):.4f}   mean of halves = {mean_half:+.4f}', flush=True)
    print(f'  full 480 rows (§1648) = {FULL_RHO:+.4f}', flush=True)
    print(f'  attenuation as predicted (mean of halves BELOW full): {pc}', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'classes': 'TYPE-SPANNING held out -- punctuation, digits, capitalised, function words; no CE rise ever measured for any',
                      'single_registered_hypothesis': 'abs(lam1/lam2) predicts relative CE rise',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)',
                      'full_data_rho_S1648': FULL_RHO},
           'per_class': per,
           'predictions': {'pred_a_both_halves_ge_020': bool(pa),
                           'pred_b_halves_agree_within_035': bool(pb),
                           'pred_c_attenuation_not_inflation': bool(pc)},
           'rho_half_A': round(rA, 4), 'rho_half_B': round(rB, 4),
           'mean_of_halves': round(mean_half, 4), 'full_data_rho': FULL_RHO,
           'docs_A': nda, 'docs_B': ndb,
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
