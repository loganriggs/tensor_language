# NOVEL-HARM SITE SWEEP -- is "ablating this site IMPROVES loss on targets absent from context" a
# property of three late attention sites, or a broader pattern nobody has looked for?
#
# S1729 certified, on two-set sign replication and with no interval, that attn14/15/16 have NEGATIVE
# per-token damage on `novel` targets -- replacing them with their optimal constant IMPROVES
# cross-entropy on tokens that never appear in the context -- while their TOTAL removal stays
# positive, so they are net useful sites that are actively wrong on half the tokens. That result
# came out of looking at three sites flagged by a broken ratio in S1728. Nobody has asked the same
# question of the other thirty-three.
#
# This sweep asks it of all 36, both eval sets, and puts a bootstrap interval on the `novel`
# component so S1729's sign-only certification can be upgraded or withdrawn. Bootstrap draws are
# SHARED across sites within an eval set, so the per-site intervals are comparable to each other.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE EFFECT IS CONFINED TO LATE ATTENTION: every site with negative held-out `novel`
#          damage is an attention site at layer >= 12. If FALSE the effect is broader than S1729
#          suggests and the entry's scope is wrong -- which is a bigger result, not a smaller one.
#   pred_b S1729 UPGRADES FROM SIGN TO INTERVAL: at attn14, attn15 and attn16 the 95% interval on
#          held-out `novel` damage lies entirely below zero. If FALSE the sign replicates on two
#          sets but is not resolved at this row count, and the registry entry stays sign-only.
#   pred_c CONTROLS: the nine held-out per-class damages S1729 published reproduce within 0.002, and
#          both baselines reproduce 3.29205 / 3.09711.
#   pred_d IT IS AN ATTENTION PROPERTY: no MLP site has negative `novel` damage on either eval set.
#          If FALSE, "the weights are the only source for novel targets" (S1729) is too simple --
#          some MLP is also actively wrong there.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV
from ops.target_token_classes import target_token_classes

D = 1152; T = 256; NB = 2000
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/novel_harm_site_sweep_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205, 1e-3),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711, 1e-2)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
# the nine held-out per-token damages S1729 certified on sign alone; reproduced here as a control
S1729_HO = {'attn14': {'induction': 0.0353, 'repeat': 0.0833, 'novel': -0.0100},
            'attn15': {'induction': 0.0075, 'repeat': 0.0197, 'novel': -0.0022},
            'attn16': {'induction': 0.0191, 'repeat': 0.0567, 'novel': -0.0148}}
LATE = ['attn14', 'attn15', 'attn16']
CLASSES = ('induction', 'repeat', 'novel')
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


@torch.no_grad()
def per_row(rows, hooks=()):
    n = rows.shape[0]
    s = {c: torch.zeros(n) for c in CLASSES}
    k = {c: torch.zeros(n) for c in CLASSES}
    hs = list(hooks)
    try:
        for i in range(0, n, 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            tg = bb[:, 1:].to(DEV)
            e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                reduction='none').reshape(tg.shape)[:, 64:]
            cls = target_token_classes(idx, tg)
            cov = COV['seen'][idx[:, 64:]]
            for c in CLASSES:
                msk = cls[c][:, 64:] & cov
                s[c][i:i + bb.shape[0]] = (e * msk).sum(1).cpu()
                k[c][i:i + bb.shape[0]] = msk.sum(1).float().cpu()
    finally:
        for h in hs:
            h.remove()
    return s, k


def dmg(cs, ck, ls, c, sel=None):
    """Per-token damage in class c: (constant CE minus live CE) over that class's tokens."""
    a = cs[c] if sel is None else cs[c][sel]
    b = ls[c] if sel is None else ls[c][sel]
    nn = ck[c] if sel is None else ck[c][sel]
    n = float(nn.sum())
    return ((float(a.sum()) - float(b.sum())) / n) if n else float('nan')


def ratio(cs, ck, ls, sel=None):
    den = dmg(cs, ck, ls, 'novel', sel)
    return dmg(cs, ck, ls, 'induction', sel) / den if abs(den) > 1e-9 else float('nan')


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    fit = load(FIT_ROWS)
    seen = torch.zeros(50257, dtype=torch.bool)
    seen[fit[:, :T].reshape(-1).long()] = True
    COV['seen'] = seen.to(DEV)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    out = {}
    print(f'NOVEL-HARM SITE SWEEP | all 36 sites | per-class per-token damage with {NB} row-level '
          f'bootstrap draws on the `novel` component', flush=True)

    for ename, epath, ce_ref, tol in EVAL_SETS:
        ev = load(epath)
        ls, lk = per_row(ev)
        ntok = int(sum(float(lk[c].sum()) for c in CLASSES))
        base = sum(float(ls[c].sum()) for c in CLASSES) / ntok
        assert abs(base - ce_ref) <= tol, f'{ename} baseline CE {base:.5f} != {ce_ref} (+/-{tol})'
        print(f'\n  {ename}: baseline CE {base:.5f} (ref {ce_ref}) | scored {ntok}', flush=True)
        g = torch.Generator().manual_seed(1729)
        nrow = ev.shape[0]
        sels = [torch.randint(0, nrow, (nrow,), generator=g) for _ in range(NB)]
        rows = {}
        neg = []
        for st in sites:
            cs, ck = per_row(ev, hooks=[mod_of(*st).register_forward_hook(
                const_hook(K[f'{st[0]}{st[1]}'].to(DEV).float()))])
            d = {c: dmg(cs, ck, ls, c) for c in CLASSES}
            tot = sum(float(cs[c].sum()) - float(ls[c].sum()) for c in CLASSES) / ntok
            dr = sorted(dmg(cs, ck, ls, 'novel', sl) for sl in sels)
            ci = (round(dr[int(0.025 * NB)], 5), round(dr[int(0.975 * NB)], 5))
            nm = f'{st[0]}{st[1]}'
            rows[nm] = {'per_token_damage': {c: round(d[c], 5) for c in CLASSES},
                        'total_removal_nats': round(tot, 5),
                        'novel_ci95': ci, 'novel_negative': d['novel'] < 0,
                        'novel_negative_resolved': ci[1] < 0}
            if d['novel'] < 0:
                neg.append(nm)
        print(f'    sites with NEGATIVE novel damage (ablation improves CE on targets absent from '
              f'context): {neg if neg else "none"}', flush=True)
        for nm in neg:
            r = rows[nm]
            print(f'      {nm:7s} novel {r["per_token_damage"]["novel"]:+8.5f} 95% CI '
                  f'{r["novel_ci95"]}  resolved {r["novel_negative_resolved"]}  | induction '
                  f'{r["per_token_damage"]["induction"]:+8.5f}  total {r["total_removal_nats"]:+8.5f}',
                  flush=True)
        out[ename] = {'baseline_ce': round(base, 5), 'scored_tokens': ntok, 'sites': rows,
                      'negative_novel_sites': neg}
        del ev
        torch.cuda.empty_cache()

    ho = out['skip11000']['sites']
    ref = out['skip7000']['sites']
    neg_ho = out['skip11000']['negative_novel_sites']
    pa = bool(neg_ho) and all(n.startswith('attn') and int(n[4:]) >= 12 for n in neg_ho)
    pb = all(ho[n]['novel_negative_resolved'] for n in LATE)
    pc = all(abs(ho[n]['per_token_damage'][c] - v) <= 0.002
             for n, kv in S1729_HO.items() for c, v in kv.items())
    pd = not any(ho[f'mlp{L}']['per_token_damage']['novel'] < 0
                 or ref[f'mlp{L}']['per_token_damage']['novel'] < 0 for L in range(18))

    print(f'\n  held-out negative-novel sites {neg_ho} -> confined to late attention {pa}', flush=True)
    print(f'  attn14/15/16 novel CI excludes zero held out -> resolved {pb}', flush=True)
    print(f'  S1729 nine damages reproduce within 0.002 -> control {pc}', flush=True)
    print(f'  no MLP site has negative novel damage on either set -> {pd}', flush=True)

    res = {'config': {'eval_sets': [e[0] for e in EVAL_SETS], 'bootstrap_draws': NB,
                      'bootstrap': 'ROW-level clusters over the 192 eval rows; NOT document-clustered '
                                   '(S1701). Draws are SHARED across sites within an eval set, so the '
                                   'per-site intervals are comparable to each other.',
                      'measure': 'per-token constant-ablation damage by target class; NEGATIVE means '
                                 'replacing the site with its optimal constant IMPROVES CE there'},
           'results': out,
           'predictions': {'pred_a_confined_to_late_attention': bool(pa),
                           'pred_b_late_attn_novel_resolved': bool(pb),
                           'pred_c_S1729_reproduces': bool(pc),
                           'pred_d_no_mlp_negative': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
