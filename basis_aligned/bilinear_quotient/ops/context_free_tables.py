# WHERE ARE THE 0.594 NATS LOST? -- the table is a context AVERAGE, and the ceiling is the
# context-FREE value.
#
# §1768: the position-wise class caps at 5.97902 covered CE held out (the model's own output given
# only the current token, from a length-1 sequence). The best compiled program reaches 6.57289. The
# gap is 0.59387 nats -- 43% of everything the class can deliver -- and §1768 left it unattributed.
#
# There is an obvious candidate. Every program in §1747-§1758 uses a table built as the per-token MEAN
# of a site's output over the fit rows, i.e. averaged over the CONTEXTS that token appeared in. The
# ceiling is achieved by the site's output with NO context at all. Those are different summaries of
# the same token, and by §1765's induction the context-free one composes to exactly the ceiling.
#
# So: build each site's table from its output on a LENGTH-1 sequence, install all 36, and measure.
# If that reproduces 5.97902, the entire gap is attributable to one named choice -- using a context
# average where the class's own optimum is the context-free value -- and the fix costs nothing extra
# to store.
#
# ROLES. Both eval roles, covered positions from 64, hybrid coverage rule (§1661) exactly as in
# §1747-§1758 so the comparison is like-for-like. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, each read back against its own sentence per
# LESSON 39:
#   pred_a THE INDUCTION HOLDS EXACTLY: the context-free-table program's covered CE is within 0.02
#          nats of §1768's ceiling, 5.97902 on skip11000. A pass means the 36 substituted sites
#          compose to the length-1 forward, as §1765 says they must. A FAIL means something breaks
#          the induction -- RMSNorm length dependence or the `v1` bus threaded from block 0 are the
#          two candidates -- and that gap is then itself the finding.
#   pred_b IT BEATS THE FIT-MEAN PROGRAM BY AT LEAST 0.4 NATS: context-free CE is at least 0.4 BELOW
#          6.57289 (lower is better). If FALSE the context average is nearly as good as the
#          context-free value and the 0.594 nats are somewhere else entirely.
#   pred_c THE CORRECTION BECOMES UNNECESSARY: the context-free table with NO linear correction
#          already scores below 6.57289, the fit-mean table WITH a rank-128 correction. If TRUE, the
#          0.664M-25.8M reals of correction machinery in §1748-§1758 were buying back an error the
#          table itself introduced.
#   pred_d CONTROLS: the length-1 logit lookup recomputed inside this script reproduces §1768's
#          5.97902 and 6.03465 within 0.001; live CE reproduces 3.09711 and 3.29205; coverage is
#          exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/context_free_tables_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1768_CEILING = {'skip7000': 6.03465, 'skip11000': 5.97902}
BEST_FITMEAN_PROGRAM = {'skip7000': 6.57512, 'skip11000': 6.57289}
ALL_TABLED = {'skip7000': 7.35114, 'skip11000': 7.35825}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl, seen):
    """The hybrid rule of §1661, unchanged: table where covered, live module elsewhere."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


@torch.no_grad()
def forward_logits(idx, hooks=()):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    STATE['idx'] = idx
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def ce(rows, hooks=()):
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward_logits(idx, hooks)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        c = COV['seen'][idx[:, 64:]]
        tot += float(e[c].sum()); cnt += int(c.sum())
    return tot / cnt


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen_cpu.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    seen = seen_cpu.to(DEV)
    COV['seen'] = seen
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'CONTEXT-FREE TABLES | {ncov} covered tokens as length-1 sequences, capturing all 36 site '
          f'outputs | DISCOVERY ONLY', flush=True)

    # one pass: capture every site's output AND the final logits, for each covered token alone
    tables = {st: torch.zeros(V, D, device=DEV) for st in sites}
    lp = torch.zeros(ncov, W, dtype=torch.float16, device=DEV)
    idmap = torch.full((V,), -1, dtype=torch.long)
    idmap[toks] = torch.arange(ncov)
    idmap = idmap.to(DEV)
    cap = {}

    def mk(st):
        def hook(mod, args, out):
            cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
            return None
        return hook

    for i in range(0, ncov, 256):
        t = toks[i:i + 256].to(DEV).unsqueeze(1)
        lg = forward_logits(t, [(st, mk(st)) for st in sites])
        lp[i:i + t.shape[0]] = torch.log_softmax(lg[:, 0].float(), -1).half()
        for st in sites:
            tables[st][t.squeeze(1)] = cap[st]
    # uncovered rows keep the mean over covered rows, exactly as the fit-mean tables did
    for st in sites:
        mu = tables[st][toks.to(DEV)].mean(0)
        mask = ~seen
        tables[st][mask] = mu
    print(f'  built 36 context-free tables and the logit lookup ({time.time() - t0:.0f}s)',
          flush=True)

    hooks = [(st, table_hook(tables[st], seen)) for st in sites]
    out = {}
    for ename, epath, ref in EVAL_SETS:
        ev = load(epath)
        lv = ce(ev)
        assert abs(lv - ref) <= 1e-3, f'{ename} live CE {lv:.5f} != {ref}'
        cf = ce(ev, hooks)
        # the direct lookup, recomputed here, as a control on §1768
        tot, cnt = 0.0, 0
        for i in range(0, ev.shape[0], 8):
            bb = ev[i:i + 8]
            idx = bb[:, :-1].to(DEV)[:, 64:]
            tg = bb[:, 1:].to(DEV)[:, 64:]
            c = seen[idx]
            r = idmap[idx].clamp(min=0)
            v = lp[r].float().gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            tot += float((-v.double())[c].sum()); cnt += int(c.sum())
        look = tot / cnt
        out[ename] = {'live_ce': round(lv, 5), 'context_free_table_ce': round(cf, 5),
                      'lookup_ceiling_recomputed': round(look, 5),
                      'S1768_ceiling': S1768_CEILING[ename],
                      'best_fitmean_program_ce': BEST_FITMEAN_PROGRAM[ename],
                      'all_tabled_ce': ALL_TABLED[ename],
                      'cf_minus_ceiling': round(cf - look, 5),
                      'fitmean_minus_cf': round(BEST_FITMEAN_PROGRAM[ename] - cf, 5)}
        o = out[ename]
        print(f'\n  {ename}: live {lv:.5f}', flush=True)
        print(f'    context-free TABLE program    {o["context_free_table_ce"]:.5f}', flush=True)
        print(f'    ceiling (lookup, recomputed)  {o["lookup_ceiling_recomputed"]:.5f}  '
              f'(§1768 {o["S1768_ceiling"]})', flush=True)
        print(f'    best FIT-MEAN program         {o["best_fitmean_program_ce"]:.5f}', flush=True)
        print(f'    all-tabled (fit-mean) base    {o["all_tabled_ce"]:.5f}', flush=True)
        print(f'    context-free is {o["cf_minus_ceiling"]:+.5f} from the ceiling and beats the '
              f'fit-mean program by {o["fitmean_minus_cf"]:+.5f}', flush=True)
        del ev

    ho = 'skip11000'
    pa = abs(out[ho]['cf_minus_ceiling']) <= 0.02
    pb = out[ho]['fitmean_minus_cf'] >= 0.4
    pc = out[ho]['context_free_table_ce'] < BEST_FITMEAN_PROGRAM[ho]
    pd = (all(abs(out[e]['lookup_ceiling_recomputed'] - v) <= 0.001
              for e, v in S1768_CEILING.items())
          and abs(out['skip7000']['live_ce'] - 3.29205) <= 1e-3
          and abs(out[ho]['live_ce'] - 3.09711) <= 1e-3 and ncov == NCOV)

    print(f'\n  the 36 context-free tables compose to the ceiling within 0.02 '
          f'({out[ho]["cf_minus_ceiling"]:+.5f}) -> induction holds {pa}', flush=True)
    print(f'  they beat the fit-mean program by >=0.4 ({out[ho]["fitmean_minus_cf"]:+.5f}) -> {pb}',
          flush=True)
    print(f'  the context-free table alone, no correction, beats the corrected fit-mean program '
          f'({out[ho]["context_free_table_ce"]:.5f} < {BEST_FITMEAN_PROGRAM[ho]}) -> {pc}',
          flush=True)
    print(f'  §1768 ceilings reproduce + live CEs + coverage {ncov} -> control {pd}', flush=True)

    r = {'config': {'tables': "each site's output on a LENGTH-1 sequence for each covered token; "
                              'uncovered rows keep the mean over covered rows, as the fit-mean '
                              'tables did',
                    'coverage_rule': 'hybrid (§1661), unchanged, so the comparison against '
                                     '§1747-§1758 is like-for-like',
                    'WHY': '§1768 left 0.59387 nats between the best program and the ceiling '
                           'unattributed. The tables in §1747-§1758 are per-token MEANS over fit '
                           'contexts; the ceiling is the context-FREE value.',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'results': out,
         'predictions': {'pred_a_induction_holds': bool(pa),
                         'pred_b_beats_fitmean_by_0p4': bool(pb),
                         'pred_c_correction_unnecessary': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
