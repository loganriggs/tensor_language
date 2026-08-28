# TIGHTENING THE BOUND ON THE POSITION-WISE OPTIMUM
#
# §1771b, after Codex's red team: the model's length-1 output is ONE FEASIBLE member of the
# position-wise class, not its optimum. Because it is achievable, its CE (5.97902 held out) is an
# UPPER bound on the class optimum H(T_{j+1}|T_j) -- so the class reaches AT LEAST 32.4% of the stake
# and context is AT MOST 67.6%. The open question §1771b names is how to tighten that.
#
# §1767 showed these row budgets cannot ESTIMATE H(T_{j+1}|T_j): a leave-one-out bigram on 27k eval
# tokens scores 7.334, worse than the program. But tightening the bound does not require estimating
# the optimum. It requires exhibiting a BETTER MEMBER of the class, and any better member tightens
# the upper bound immediately and with no statistical claim at all.
#
# Three cheap constructions, all position-wise by construction:
#   TEMPERATURE  p proportional to p_model^(1/tau). One parameter. A length-1 forward is a regime the
#                model was never trained in, so its confidence calibration has no reason to be right.
#   BLEND        a convex mixture of the length-1 model, the fit-row bigram (§1766) and the fit-row
#                unigram. A mixture of position-wise predictors is position-wise.
#   BOTH         temperature applied to the model component inside the blend.
# Every free parameter is fitted on skip7000 ALONE and the bound is read off skip11000, so the
# reported bound is achievable on rows that chose nothing.
#
# ROLES. skip7000 selects the parameters; skip11000 carries the bound. Covered positions from 64.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, each read back against its own sentence per
# LESSON 39:
#   pred_a THE BOUND TIGHTENS: the best selected member scores BELOW 5.97902 on skip11000 (lower CE
#          is better). If FALSE the length-1 output is already the best position-wise predictor these
#          three constructions can reach, which would make 5.979 a far more interesting reference
#          than it currently is.
#   pred_b BY AT LEAST 0.05 NATS. Scored independently of pred_a, since a tiny improvement would
#          tighten the bound without changing any conclusion drawn from it.
#   pred_c TEMPERATURE ALONE MOVES IT: the best temperature-only member also scores below 5.97902. If
#          FALSE the gain comes entirely from mixing in corpus statistics rather than from the model
#          being miscalibrated at length 1, which is a different and more informative reason.
#   pred_d CONTROLS: the untouched length-1 arm reproduces §1768's 5.97902 and 6.03465 within 0.001;
#          the pure fit-row bigram arm reproduces §1766's 7.90729 and 7.88804 within 0.001; coverage
#          is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
ALPHA = 0.01
TAUS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
STEP = 0.05
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/tighten_position_wise_bound_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1768_REF = {'skip7000': 6.03465, 'skip11000': 5.97902}
S1766_BIGRAM = {'skip7000': 7.88804, 'skip11000': 7.90729}
SELECT, HOLD = 'skip7000', 'skip11000'


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


@torch.no_grad()
def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen_cpu.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    seen = seen_cpu.to(DEV)
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    idmap = torch.full((V,), -1, dtype=torch.long)
    idmap[toks] = torch.arange(ncov)
    idmap = idmap.to(DEV)
    print(f'TIGHTEN THE POSITION-WISE BOUND | length-1 model, fit-row bigram, unigram | parameters '
          f'selected on {SELECT}, bound read on {HOLD} | DISCOVERY ONLY', flush=True)

    lgm = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = toks[i:i + 256].to(DEV).unsqueeze(1)
        lgm[i:i + t.shape[0]] = forward_logits(t)[:, 0].float()

    cur, nxt = fit[:, :T].reshape(-1).long(), fit[:, 1:T + 1].reshape(-1).long()
    rows_i = idmap[cur.to(DEV)]
    keep = rows_i >= 0
    counts = torch.zeros(ncov, W, device=DEV)
    counts.index_put_((rows_i[keep], nxt.to(DEV)[keep]),
                      torch.ones(int(keep.sum()), device=DEV), accumulate=True)
    uni = counts.sum(0)
    p_uni = ((uni + 1.0) / (uni.sum() + W)).unsqueeze(0)
    p_big = (counts + ALPHA * W * p_uni) / (counts.sum(1, keepdim=True) + ALPHA * W)
    del counts
    torch.cuda.empty_cache()
    print(f'  built the three position-wise components ({time.time() - t0:.0f}s)', flush=True)

    ev = {}
    for ename, epath, ref in EVAL_SETS:
        e = load(epath)
        ids, tgs = [], []
        for i in range(0, e.shape[0], 8):
            bb = e[i:i + 8]
            idx = bb[:, :-1].to(DEV)[:, 64:]
            tg = bb[:, 1:].to(DEV)[:, 64:]
            c = seen[idx]
            ids.append(idmap[idx][c]); tgs.append(tg[c])
        ev[ename] = (torch.cat(ids), torch.cat(tgs))
        print(f'  {ename}: {ev[ename][0].numel()} covered scored positions', flush=True)

    def ce_of(pm_rows, w_m, w_b, w_u, ename):
        r, tg = ev[ename]
        tot, n, B = 0.0, 0, 8192
        for i in range(0, r.numel(), B):
            rr, tt = r[i:i + B], tg[i:i + B]
            p = w_m * pm_rows[rr] + w_b * p_big[rr] + w_u * p_uni
            v = p.gather(-1, tt.unsqueeze(-1)).squeeze(-1)
            tot += float(-torch.log(v.clamp_min(1e-30)).double().sum()); n += rr.numel()
        return tot / n

    temp = {}
    for tau in TAUS:
        pm = torch.softmax(lgm / tau, -1)
        temp[tau] = {e: round(ce_of(pm, 1.0, 0.0, 0.0, e), 5) for e in ev}
        del pm
    best_tau = min(TAUS, key=lambda t: temp[t][SELECT])
    print(f'\n  temperature (selected on {SELECT}):', flush=True)
    for tau in TAUS:
        mark = '  <-' if tau == best_tau else ''
        print(f'    tau {tau:<5} {SELECT} {temp[tau][SELECT]:.5f}  {HOLD} {temp[tau][HOLD]:.5f}{mark}',
              flush=True)

    pm = torch.softmax(lgm / best_tau, -1)
    best = None
    wm = 0.0
    while wm <= 1.0 + 1e-9:
        wb = 0.0
        while wm + wb <= 1.0 + 1e-9:
            wu = max(0.0, 1.0 - wm - wb)
            s = ce_of(pm, wm, wb, wu, SELECT)
            if best is None or s < best[3]:
                best = (round(wm, 3), round(wb, 3), round(wu, 3), round(s, 5))
            wb += STEP
        wm += STEP
    hold = ce_of(pm, best[0], best[1], best[2], HOLD)
    pm1 = torch.softmax(lgm, -1)
    pure_model = {e: round(ce_of(pm1, 1.0, 0.0, 0.0, e), 5) for e in ev}
    pure_big = {e: round(ce_of(pm1, 0.0, 1.0, 0.0, e), 5) for e in ev}
    print(f'\n  best blend on {SELECT}: w_model {best[0]} w_bigram {best[1]} w_unigram {best[2]} '
          f'at tau {best_tau} -> {SELECT} {best[3]:.5f}  {HOLD} {hold:.5f}', flush=True)
    print(f'  pure length-1 model: {pure_model}   (§1768 {S1768_REF})', flush=True)
    print(f'  pure fit-row bigram: {pure_big}   (§1766 {S1766_BIGRAM})', flush=True)

    gain = S1768_REF[HOLD] - hold
    tau_hold = temp[best_tau][HOLD]
    pa = hold < S1768_REF[HOLD]
    pb = gain >= 0.05
    pc = tau_hold < S1768_REF[HOLD]
    pd = (all(abs(pure_model[e] - v) <= 0.001 for e, v in S1768_REF.items())
          and all(abs(pure_big[e] - v) <= 0.001 for e, v in S1766_BIGRAM.items())
          and ncov == NCOV)

    print(f'\n  the bound tightens: {hold:.5f} < {S1768_REF[HOLD]} -> {pa}  (by {gain:+.5f})',
          flush=True)
    print(f'  by at least 0.05 -> {pb}', flush=True)
    print(f'  temperature alone moves it ({tau_hold:.5f} at tau {best_tau}) -> {pc}', flush=True)
    print(f'  pure arms reproduce §1768 and §1766 + coverage {ncov} -> control {pd}', flush=True)

    r = {'config': {'components': 'length-1 model logits; fit-row add-alpha bigram with unigram '
                                  'backoff; fit-row unigram. A convex mixture of position-wise '
                                  'predictors is position-wise.',
                    'selection': f'every free parameter fitted on {SELECT} alone; the bound is read '
                                 f'on {HOLD}, which chose nothing',
                    'WHY': '§1771b: the length-1 output is a feasible member, not the optimum, so its '
                           'CE upper-bounds the class optimum. Exhibiting a BETTER MEMBER tightens '
                           'that bound with no statistical claim, which §1767 showed estimation '
                           'cannot supply at these row budgets.',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'temperature_sweep': {str(k): v for k, v in temp.items()}, 'best_tau': best_tau,
         'best_blend': {'w_model': best[0], 'w_bigram': best[1], 'w_unigram': best[2],
                        'select_ce': best[3], 'hold_ce': round(hold, 5)},
         'pure_length1_model': pure_model, 'pure_fit_row_bigram': pure_big,
         'new_upper_bound_on_class_optimum': round(hold, 5),
         'previous_bound_S1768': S1768_REF[HOLD], 'tightened_by': round(gain, 5),
         'predictions': {'pred_a_bound_tightens': bool(pa),
                         'pred_b_by_at_least_0p05': bool(pb),
                         'pred_c_temperature_alone_moves_it': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
