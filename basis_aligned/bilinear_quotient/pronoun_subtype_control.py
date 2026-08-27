# pronoun_subtype_control: THE ONE PENDING RETRACTION ITEM STILL STANDING, TESTED
# AGAINST AN EXACT-SUBTYPE CONTROL
#
# §1637 established that a control must match the claim on CLASS TYPE, not just cell
# and configuration, and that §1633's margin of 13/60 collapsed to 2/60 once the
# target (punctuation) was compared against punctuation instead of function words.
#
# That raises the obvious question about §1634, which is the sole surviving basis for
# the second pending item in PENDING_RETRACTION_S1612.md. §1634 compared
# pronouns@mlp17 against with/from/at/by/as. Checked rather than assumed: the target
# (he|she|they) and those controls are ALL closed-class function words, so §1634 was
# far better matched than §1633 -- but prepositions and conjunctions are a different
# SUBTYPE from personal pronouns, and §1637's lesson is precisely that subtype can
# decide the answer.
#
# This runs the exact-subtype control: four OTHER personal pronoun classes (` it`,
# ` we`, ` you`, ` I`) at pronouns' own cell -- mlp17, rank-8, TOP-6, corrected
# quantity, 20 independent bases x 3 disjoint 160-row chunks = 60 trials each.
#
# Reference from §1634 at this exact cell: pronouns (he|she|they) separates 60/60 with
# mean gap −.1320; five function-word controls gave 52-60/60.
#
# If below-null is generic among personal pronouns too, the second pending item is
# confirmed on a properly matched control and §1612's structural reading is
# unsupported. If instead he/she/they stands well clear of other pronouns, the item
# should be WITHDRAWN and §1612's reading survives -- which would be the third time
# today a correction of mine gets corrected, and is exactly why this is being run
# before Logan acts on it.
#
# Registered predictions:
#   pred_a BELOW-NULL IS GENERIC AMONG PRONOUNS: at least 3 of the 4 other pronoun
#          classes separate >= 52/60, the floor the function-word controls set.
#   pred_b NO MEANINGFUL MARGIN ON A MATCHED CONTROL: pronouns' 60/60 exceeds the BEST
#          other-pronoun class by <= 5 of 60.
#   pred_c THE DIRECTION IS SHARED: all four have a NEGATIVE mean gap, i.e. every
#          personal pronoun class writes less concentratedly than a random basis here.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
SITE_STOP = 17        # S1598 SITE: forward stops here, P = its input
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pronoun_subtype_control_results.json'
NR = 960
# per-cell site/rank set in CELLS below
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
EDIT = {'set': set(), 'V': None, 'mu': None}   # mu: {name: [2]}
FIN = {'on': False, 'V': None, 'mu': None}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def mk_cproj_hook(L):
    def hook(mod, args, output):
        nm = f'attn{L}'
        if nm not in EDIT['set']:
            return None
        o = output.float()
        pv = o @ EDIT['V']                       # [B,T,2]
        o = o - (pv - EDIT['mu'][nm]) @ EDIT['V'].T
        return o.to(output.dtype)
    return hook


def mk_mlp_hook(L):
    def hook(mod, args, output):
        o = None
        nm = f'mlp{L}'
        if nm in EDIT['set']:
            o = output.float()
            pv = o @ EDIT['V']
            o = o - (pv - EDIT['mu'][nm]) @ EDIT['V'].T
        return None if o is None else o.to(output.dtype)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    if FIN['on']:
        xf = x.float()
        pv = xf @ FIN['V']
        x = (xf - (pv - FIN['mu']) @ FIN['V'].T).to(x.dtype)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def capture_fwd(idx, V2, lam2, acc, pm):
    """Exact manual forward through layer SITE, accumulating projections of
    every component output onto V2 (global + class sums), head-grain scores,
    mean_s, and the reconstruction check. pm: [B,T] class mask."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    vmask = torch.ones(B, T, dtype=torch.bool, device=DEV)
    vmask[:, :64] = False
    vf = vmask.reshape(-1); pf = pm.reshape(-1)
    tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))

    def add(nm, o):
        pv = (o.float().reshape(-1, D) @ V2)      # [N,2]
        acc['sum'][nm] += pv[vf].sum(0)
        acc['csum'][nm] += pv[pf].sum(0)

    add('x0', x0)
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        qp = at.c_q(xin).view(B, T, 9, 128).float()
        kp = at.c_k(xin).view(B, T, 9, 128).float()
        q2p = at.c_q2(xin).view(B, T, 9, 128).float()
        k2p = at.c_k2(xin).view(B, T, 9, 128).float()
        cos, sin = at.rotary(qp)
        q = are(F.rms_norm(qp, (128,)), cos, sin)
        k = are(F.rms_norm(kp, (128,)), cos, sin)
        q2 = are(F.rms_norm(q2p, (128,)), cos, sin)
        k2 = are(F.rms_norm(k2p, (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        ao = at.c_proj(y.reshape(B, T, D))
        add(f'attn{L}', ao)
        # head grain: y_h @ Wp_h.T projected on V2
        Wp = at.c_proj.weight.float()             # [D, D]
        for hh in range(9):
            M = Wp[:, hh * 128:(hh + 1) * 128].T @ V2      # [128,2]
            pv = (y[:, :, hh].float().reshape(-1, 128) @ M)
            acc['hsum'][L][hh] += pv[vf].sum(0)
            acc['hcsum'][L][hh] += pv[pf].sum(0)
        x = xm + ao
        if L == SITE_STOP:
            P = x                      # S1597: P is the SITE's INPUT residual
            acc['n'] += int(vf.sum()); acc['cn'] += int(pf.sum())
            acc['P_proj'].append((P.float().reshape(-1, D) @ V2)[vf].sum(0))
            return                     # forward ENDS at the site (slice_writers.py:56)
        mo = blk.mlp(F.rms_norm(x, (D,)))
        add(f'mlp{L}', mo)
        x = x + mo
    P = x
    acc['n'] += int(vf.sum()); acc['cn'] += int(pf.sum())
    acc['P_proj'].append((P.float().reshape(-1, D) @ V2)[vf].sum(0))


CELLS = {
    'pronouns': {'site': 17, 'rank': 8, 'top': 6, 'certified': True, 'published': 0.482,
                 'pat': r'^ (he|she|they|He|She|They)$'},
    'it':       {'site': 17, 'rank': 8, 'top': 6, 'certified': False, 'pat': r'^ (it|It)$'},
    'we':       {'site': 17, 'rank': 8, 'top': 6, 'certified': False, 'pat': r'^ (we|We)$'},
    'you':      {'site': 17, 'rank': 8, 'top': 6, 'certified': False, 'pat': r'^ (you|You)$'},
    'I':        {'site': 17, 'rank': 8, 'top': 6, 'certified': False, 'pat': r'^ I$'},
}
CHUNKS, ROWS_PER_CHUNK = 3, 160        # three DISJOINT chunks
N_RANDOM = 20                          # seeds 1729+i; i=0 is the single seed S1625/S1627 used
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT  = PT + '.rowcache/fineweb_oracle_v2_receipt.json'


@torch.no_grad()
def abs_mass(rows, V2, lam2, mask_v):
    """Per-component ABSOLUTE attribution mass -- the S1597/S1598 statistic:
    delta_c = |coef_c * (class_mean - global_mean)| summed over slice directions."""
    SITE_UP = 17          # S1598: comps are UPSTREAM of the slice site
    comps = ['x0'] + [f'attn{L}' for L in range(SITE_UP + 1)] \
        + [f'mlp{L}' for L in range(SITE_UP)]
    rk = V2.shape[1]
    acc = {'sum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'csum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
        capture_fwd(idx, V2, lam2, acc, pm)

    lam0 = [float(b.lambdas[0]) for b in H]; lam1 = [float(b.lambdas[1]) for b in H]
    coef = {}
    for l in range(SITE_STOP + 1):
        c = 1.0
        for k in range(l + 1, SITE_STOP + 1):
            c *= lam0[k]
        coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    tx0 = 1.0
    for k in range(SITE_STOP + 1):
        tx0 = lam0[k] * tx0 + lam1[k]
    coef['x0'] = tx0

    recon = sum(coef[c] * acc['sum'][c] for c in comps)
    Pv = torch.stack(acc['P_proj']).sum(0)
    rec_err = float((recon - Pv).abs().max() / Pv.abs().max())

    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    delta = {c: (coef[c] * (cmu[c] - mu[c])).abs().sum().item() for c in comps}
    return delta, acc['cn'], rec_err


@torch.no_grad()
def main():
    import os, hashlib
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    assert raw.shape[0] >= CHUNKS * ROWS_PER_CHUNK, f'short row tensor {tuple(raw.shape)}'
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'CANONICAL .rowcache/fineweb_n480_skip80.pt: {CHUNKS} chunks x {ROWS_PER_CHUNK} '
          f'rows x {N_RANDOM} seeds (receipt sha256[:16]={rh})', flush=True)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    out_cells = {}

    for cname, cfg in CELLS.items():
        site, rank, TOP = cfg['site'], cfg['rank'], cfg['top']
        mask_v = rx(cfg['pat'])
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
        Dw = H[site].mlp.Down.weight.float()
        S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
        lam, V = torch.linalg.eigh(S)
        o = lam.abs().argsort(descending=True)[:rank]
        V2l, lam2l = V[:, o].contiguous(), lam[o].contiguous()

        lam_share = []
        for ci, ch in enumerate(chunks):
            delta, cn, rec = abs_mass(ch, V2l, lam2l, mask_v)
            ranked = sorted(delta, key=lambda c: -delta[c]); tot = sum(delta.values())
            lam_share.append(sum(delta[c] for c in ranked[:TOP]) / max(tot, 1e-9))

        per_chunk_rnd = [[] for _ in range(CHUNKS)]
        all_trials = []            # every one of the 60 random shares, for the count
        seed0_rnd = []
        for i in range(N_RANDOM):
            gen = torch.Generator(device=DEV).manual_seed(1729 + i)
            rnd, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=gen))
            for ci, ch in enumerate(chunks):
                delta, cn, rec = abs_mass(ch, rnd.contiguous(),
                                          torch.ones(rank, device=DEV), mask_v)
                ranked = sorted(delta, key=lambda c: -delta[c]); tot = sum(delta.values())
                sh = sum(delta[c] for c in ranked[:TOP]) / max(tot, 1e-9)
                per_chunk_rnd[ci].append(sh)
                all_trials.append(sh)
                if i == 0:
                    seed0_rnd.append(sh)

        mean_rnd = [sum(v) / len(v) for v in per_chunk_rnd]
        null20 = sum(mean_rnd) / len(mean_rnd)
        null1 = sum(seed0_rnd) / len(seed0_rnd)
        gaps20 = [round(l - r, 4) for l, r in zip(lam_share, mean_rnd)]
        gaps1 = [round(l - r, 4) for l, r in zip(lam_share, seed0_rnd)]
        lo = min(lam_share); hi = max(lam_share)
        mean_gap = sum(l - r for l, r in zip(lam_share, mean_rnd)) / CHUNKS
        # directional separation: how many of the 60 trials the lambda arm beats,
        # measured in whichever direction the class actually points
        if mean_gap >= 0:
            sep = sum(1 for t in all_trials if lo > t)
        else:
            sep = sum(1 for t in all_trials if hi < t)
        out_cells[cname] = {'separation': sep, 'trials': len(all_trials),
                            'mean_gap': round(mean_gap, 4),
                            'certified': bool(cfg.get('certified', False)),
                            'random_range': [round(min(all_trials), 4), round(max(all_trials), 4)],
                            'lambda': [round(x, 4) for x in lam_share],
                            'null_20seed_per_chunk': [round(x, 4) for x in mean_rnd],
                            'null_20seed': round(null20, 4),
                            'null_seed1729': round(null1, 4),
                            'gaps_20seed': gaps20, 'gaps_seed1729': gaps1}
        print(f"  {cname:9s} {'CERTIFIED' if cfg.get('certified') else 'fresh    '} "
              f"sep {sep:2d}/{len(all_trials)} | mean gap {mean_gap:+.4f} | "
              f"lambda {[round(x,4) for x in lam_share]} | null20 {null20:.4f}", flush=True)

    fresh = {k: v for k, v in out_cells.items() if not v['certified']}
    cert = out_cells['pronouns']
    NT = cert['trials']
    best_fresh = max(v['separation'] for v in fresh.values())

    n_high = sum(1 for v in fresh.values() if v['separation'] >= 52)
    n_neg = sum(1 for v in fresh.values() if v['mean_gap'] < 0)
    pa = n_high >= 3
    pb = cert['separation'] - best_fresh <= 5
    pc = n_neg == 4

    print(f"\n  SEPARATION at mlp11 r2 TOP-4, site held FIXED, {N_RANDOM*CHUNKS} trials each:",
          flush=True)
    for k in sorted(out_cells, key=lambda x: -out_cells[x]['separation']):
        v = out_cells[k]
        print(f"    {k:9s} {'CERTIFIED' if v['certified'] else 'fresh    '} "
              f"separation {v['separation']:2d}/{v['trials']}  mean gap {v['mean_gap']:+.4f}  "
              f"lambda {v['lambda']}  random {v['random_range']}", flush=True)
    print(f"  pronouns(he/she/they) {cert['separation']}/{NT}  vs  best other pronoun "
          f"{best_fresh}/{NT}  -> MARGIN {cert['separation']-best_fresh}", flush=True)
    print(f"  other-pronoun classes at >= 52/60: {n_high}/4 | with NEGATIVE gap: {n_neg}/4",
          flush=True)
    print(f"  (§1634 same cell, FUNCTION-WORD controls: 52-60/60, margin 0)", flush=True)
    print(f"  (§1637: a type-MISmatched control gave §1633 a margin of 13 that was really 2)",
          flush=True)

    out = {'config': {'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'pat'}
                                for k, v in CELLS.items()},
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'n_random': N_RANDOM, 'seeds': f'1729..{1729+N_RANDOM-1}',
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'statistic': 'ABSOLUTE attribution mass, corrected quantity; null is a 60-trial mean'},
           'cells': out_cells,
           'predictions': {'pred_a_3of4_other_pronouns_ge_52': bool(pa),
                           'pred_b_margin_le_5_on_matched_control': bool(pb),
                           'pred_c_all_4_negative_gap': bool(pc)},
           'n_other_pronouns_ge_52': n_high, 'n_negative_gap': n_neg,
           'certified_separation': cert['separation'], 'best_fresh_separation': best_fresh,
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
