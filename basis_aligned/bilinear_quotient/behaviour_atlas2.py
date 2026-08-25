# behaviour_atlas2: GENERATOR #1 — refill the candidate pool. Seven unscreened behavior
# classes through a per-attention-layer ablation sweep (18 layers + base = 19 passes,
# every class scored from the same passes). Denominator note (§513's bug corrected): the
# elsewhere reference is the global non-target complement per class; all seven classes
# are SMALL (<2% of positions), so complement ~ global and the §513 large-class bias
# does not arise — stated, not assumed. This is a GENERATOR: flags feed the registry
# pool; every promoted candidate still gets its own §1302-standard screen with controls.
#
# Classes (masks from the token stream, boundary phenomena separated per §1339's rule):
#   possessive   next tok strips to "'s"
#   ellipsis     next tok contains '...' or the unicode ellipsis
#   ordinal      next tok strips to st/nd/rd/th AND current tok is pure digits
#   year         next tok is a 4-digit 19xx/20xx number
#   unit         next tok in a small unit lexicon (km, kg, mm, cm, mph, GB, MB, lbs, ft)
#   hyphen       next tok strips to '-' AND both neighbors are alphabetic (word join)
#   quote_close  next tok contains '"' AND an odd number of '"' tokens precede (parity)
#
# Registered predictions:
#   pred_a THE POOL REFILLS: >= 2 classes show a top layer with target damage >= 0.05
#          nats AND target/else ratio >= 3.
#   pred_b NEW CAPABILITIES, NOT ECHOES: no class's top layer is a8 or a13.
#   pred_c GRAMMAR-BAND PRIOR: possessive-'s peaks in the front band (L0-2) — syntactic
#          attachment, not content.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'behaviour_atlas2_results.json'
NMEAN = 24; NR = 960
H = m.transformer.h
CUR = {'abl': None, 'mean': None}


def mk_hook(L):
    def hook(mod, args, out):
        if CUR['abl'] == L:
            y = out[0] if isinstance(out, tuple) else out
            rep = CUR['mean'][L].to(y.dtype).expand_as(y)
            return (rep,) + tuple(out[1:]) if isinstance(out, tuple) else rep
        return out
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    UNITS = {'km', 'kg', 'mm', 'cm', 'mph', 'gb', 'mb', 'lbs', 'ft', 'mi', 'oz', 'ml'}
    sets = {k: set() for k in ('poss', 'ell', 'ordn', 'year', 'unit', 'hyph', 'q')}
    digits = set(); alpha = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        ds = d.strip()
        if ds == "'s":
            sets['poss'].add(tok)
        if '...' in d or '…' in d:
            sets['ell'].add(tok)
        if ds in ('st', 'nd', 'rd', 'th'):
            sets['ordn'].add(tok)
        if ds.isdigit() and len(ds) == 4 and ds[:2] in ('19', '20'):
            sets['year'].add(tok)
        if ds.lower() in UNITS:
            sets['unit'].add(tok)
        if ds == '-':
            sets['hyph'].add(tok)
        if '"' in d:
            sets['q'].add(tok)
        if ds.isdigit():
            digits.add(tok)
        if ds.isalpha():
            alpha.add(tok)
    tt = lambda s: torch.tensor(sorted(s)) if s else torch.tensor([-1])
    ids = {k: tt(v) for k, v in sets.items()}
    dig_t, alp_t = tt(digits), tt(alpha)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]

    MASKS = {}
    MASKS['possessive'] = torch.isin(tgt, ids['poss'])
    MASKS['ellipsis'] = torch.isin(tgt, ids['ell'])
    MASKS['ordinal'] = torch.isin(tgt, ids['ordn']) & torch.isin(toks, dig_t)
    MASKS['year'] = torch.isin(tgt, ids['year'])
    MASKS['unit'] = torch.isin(tgt, ids['unit'])
    nxt = torch.cat([tgt[:, 1:], torch.zeros_like(tgt[:, :1])], 1)
    MASKS['hyphen'] = torch.isin(tgt, ids['hyph']) & torch.isin(toks, alp_t) \
        & torch.isin(nxt, alp_t)
    isq = torch.isin(toks, ids['q'])
    parity = (isq.long().cumsum(1) % 2) == 1
    MASKS['quote_close'] = torch.isin(tgt, ids['q']) & parity
    for k in MASKS:
        MASKS[k][:, :64] = False
        print(f"{k}: n {int(MASKS[k].sum())}", flush=True)
    ANY = torch.zeros_like(tgt, dtype=torch.bool)
    for k in MASKS:
        ANY |= MASKS[k]
    ELSE = ~ANY; ELSE[:, :64] = False

    # per-layer means
    caps = {L: [] for L in range(18)}
    hs = []
    for L in range(18):
        def mk(L):
            def h(mod, args, out):
                y = out[0] if isinstance(out, tuple) else out
                caps[L].append(y.detach().float().mean((0, 1)))
                return out
            return h
        hs.append(H[L].attn.register_forward_hook(mk(L)))
    CUR['abl'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hs:
        h.remove()
    CUR['mean'] = {L: torch.stack(caps[L]).mean(0) for L in range(18)}
    hooks = [H[L].attn.register_forward_hook(mk_hook(L)) for L in range(18)]

    def ce_all(abl):
        CUR['abl'] = abl
        sums = {k: 0.0 for k in MASKS}; ns = {k: 0 for k in MASKS}
        se = 0.0; ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg2 = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg2.reshape(-1),
                                 reduction='none').view(tg2.shape)
            for k in MASKS:
                mm = MASKS[k][i:i + 8].to(DEV)
                sums[k] += float(ce[mm].sum()); ns[k] += int(mm.sum())
            me = ELSE[i:i + 8].to(DEV)
            se += float(ce[me].sum()); ne += int(me.sum())
        return {k: sums[k] / max(ns[k], 1) for k in MASKS}, se / max(ne, 1), ns

    base, base_e, ns = ce_all(None)
    print('base:', {k: round(v, 3) for k, v in base.items()}, flush=True)
    prof = {k: {} for k in MASKS}
    else_prof = {}
    for L in range(18):
        r, re_, _ = ce_all(L)
        else_prof[L] = re_ - base_e
        for k in MASKS:
            prof[k][L] = round(r[k] - base[k], 4)
        print(f"a{L}: " + " ".join(f"{k} {prof[k][L]:+.3f}" for k in MASKS)
              + f" | else {else_prof[L]:+.3f}", flush=True)
        json.dump({'partial': True, 'profiles': prof}, open(OUT, 'w'), indent=1)
    for h in hooks:
        h.remove()

    summary = {}
    for k in MASKS:
        top = max(prof[k], key=prof[k].get)
        dmg = prof[k][top]
        ratio = dmg / max(abs(else_prof[top]), 1e-4)
        summary[k] = {'n': ns[k], 'base_ce': round(base[k], 3), 'top': f'a{top}',
                      'dmg': round(dmg, 4), 'else_dmg_of_top': round(else_prof[top], 4),
                      'ratio': round(ratio, 2),
                      'qualifies': bool(dmg >= 0.05 and ratio >= 3.0)}
        print(f"{k}: top a{top} dmg {dmg:+.3f} ratio {ratio:.1f} "
              f"{'QUALIFIES' if summary[k]['qualifies'] else 'no'}", flush=True)
    qual = [k for k in summary if summary[k]['qualifies']]
    pa = len(qual) >= 2
    pb = all(summary[k]['top'] not in ('a8', 'a13') for k in qual) if qual else False
    poss_top = int(summary['possessive']['top'][1:])
    pc = poss_top <= 2
    out = {'n_rows': NR, 'classes': summary, 'profiles': prof,
           'else_profile': {str(L): round(v, 4) for L, v in else_prof.items()},
           'qualifying': qual, 'pred_a_pool_refills': bool(pa),
           'pred_b_new_not_echo': bool(pb), 'pred_c_possessive_front': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nqualifying: {qual}")
    print(f"pred_a refills {pa} | pred_b new {pb} | pred_c poss-front {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
