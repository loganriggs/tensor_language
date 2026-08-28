# THE NULL MODEL THIS THREAD NEVER HAD
#
# §1765: with the 36-site table+linear program installed, a perturbation at any position reaches NO
# later position -- exactly zero -- because every site's substitute is a function of the current
# position's token and residual, so by induction the whole forward at a substituted position is a
# function of that token alone. The program deletes attention. Every figure in §1747-§1758 is
# therefore the recovery of a PURE PER-TOKEN FUNCTION, computed through 36 layers of position-wise
# arithmetic.
#
# Which raises the question the thread never asked: how much of that is the token, and how much is
# the 36 layers? The null model is a direct map from the current token to the next-token
# distribution -- a bigram -- fitted on the SAME fit rows the programs use, with no layers at all.
#
# If the bigram beats the best program, then 36 layers of position-wise arithmetic are worth less
# than a lookup table, and the compilation thread's fidelity figures have been measured against the
# wrong floor since §1662.
#
# DATA LIMITATION, STATED UP FRONT: the fit rows are 96 x 256 = 24,576 tokens, which is very little
# for a bigram over 5,419 covered types. Add-alpha smoothing with unigram backoff is used and alpha
# is swept, so the comparison is not decided by one arbitrary smoothing choice. This is the same
# corpus the programs were fitted on, which is what makes it the fair floor rather than a strong
# language model.
#
# ROLES. Fitting uses the fit rows; both eval roles reported, scored on covered positions from 64 to
# match every published figure. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a THE BIGRAM BEATS THE BEST PROGRAM: its covered CE on skip11000 is below 6.57289, which is
#          the best 36-site program (table 64 + correction 128, +0.78536 off the 7.35825 baseline).
#          If FALSE, the 36 layers of position-wise arithmetic earn their place over a raw lookup and
#          the thread's figures survive as more than a bigram in disguise.
#   pred_b THE BIGRAM BEATS THE ALL-TABLED BASELINE of 7.35825. Scored independently of pred_a, since
#          it can clear the baseline without clearing the best program. If FALSE the baseline itself
#          is stronger than a bigram and the null is not informative.
#   pred_c THE BIGRAM DOES NOT BEAT THE LIVE MODEL (3.09711). A positive control on the whole setup:
#          if a bigram fitted on 24,576 tokens beats a 546M-parameter model, the scoring is wrong.
#   pred_d CONTROLS: live CE reproduces 3.29205 and 3.09711 on covered positions, coverage is 5419 of
#          50257, and the UNIGRAM is worse than the bigram at the selected alpha -- so the bigram is
#          demonstrably using the conditioning it claims to use.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257
ALPHAS = (0.01, 0.1, 1.0)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/per_token_null_model_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
BEST_PROGRAM_CE = {'skip7000': 7.35114 - 0.77602, 'skip11000': 7.35825 - 0.78536}
ALL_TABLED_CE = {'skip7000': 7.35114, 'skip11000': 7.35825}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


@torch.no_grad()
def live_ce(rows, seen):
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        cov = seen[idx[:, 64:]]
        tot += float(e[cov].sum()); cnt += int(cov.sum())
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
    print(f'PER-TOKEN NULL MODEL | bigram on the SAME {fit.shape[0]}x{T} fit rows | '
          f'coverage {ncov} of {V} | DISCOVERY ONLY', flush=True)

    # bigram counts from the fit rows, on GPU as a sparse-ish dense row bank over covered types only
    cur = fit[:, :T].reshape(-1).long()
    nxt = fit[:, 1:T + 1].reshape(-1).long()
    idmap = torch.full((V,), -1, dtype=torch.long)
    idmap[seen_cpu.nonzero(as_tuple=True)[0]] = torch.arange(ncov)
    rows_i = idmap[cur]
    keep = rows_i >= 0
    counts = torch.zeros(ncov, V, dtype=torch.float32, device=DEV)
    counts.index_put_((rows_i[keep].to(DEV), nxt[keep].to(DEV)),
                      torch.ones(int(keep.sum()), device=DEV), accumulate=True)
    uni = counts.sum(0)
    print(f'  {int(keep.sum())} bigram observations over {ncov} covered current-token types; '
          f'{int((counts.sum(1) > 0).sum())} types actually observed', flush=True)
    idmap_dev = idmap.to(DEV)

    ev = {n: load(p) for n, p, _ in EVAL_SETS}
    live = {}
    for ename, _, ref in EVAL_SETS:
        live[ename] = live_ce(ev[ename], seen)
        assert abs(live[ename] - ref) <= 1e-3, f'{ename} live CE {live[ename]:.5f} != {ref}'
    print(f'  live CE covered: ' + '  '.join(f'{e} {live[e]:.5f}' for e in live), flush=True)

    def score(table_logprob, ename):
        tot, cnt = 0.0, 0
        rows = ev[ename]
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV)[:, 64:]
            tg = bb[:, 1:].to(DEV)[:, 64:]
            cov = seen[idx]
            r = idmap_dev[idx]
            lp = table_logprob[r.clamp(min=0)].gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            tot += float((-lp.double())[cov].sum()); cnt += int(cov.sum())
        return tot / cnt

    out = {}
    uni_lp = torch.log((uni + 1.0) / (uni.sum() + V)).unsqueeze(0).expand(ncov, V)
    out['unigram'] = {e: round(score(uni_lp, e), 5) for e in ev}
    print(f'  unigram: ' + '  '.join(f'{e} {out["unigram"][e]:.5f}' for e in out['unigram']),
          flush=True)
    for a in ALPHAS:
        back = torch.softmax(uni_lp[0], 0).unsqueeze(0)
        p = (counts + a * V * back) / (counts.sum(1, keepdim=True) + a * V)
        lp = torch.log(p.clamp_min(1e-30))
        out[f'bigram_alpha{a}'] = {e: round(score(lp, e), 5) for e in ev}
        print(f'  bigram alpha={a}: ' + '  '.join(
            f'{e} {out[f"bigram_alpha{a}"][e]:.5f}' for e in out[f'bigram_alpha{a}']), flush=True)
        del p, lp

    ho = 'skip11000'
    best_key = min((k for k in out if k.startswith('bigram')), key=lambda k: out[k][ho])
    best = out[best_key][ho]
    pa = best < BEST_PROGRAM_CE[ho]
    pb = best < ALL_TABLED_CE[ho]
    pc = best > live[ho]
    pd = (abs(live['skip7000'] - 3.29205) <= 1e-3 and abs(live[ho] - 3.09711) <= 1e-3
          and ncov == NCOV and best < out['unigram'][ho])

    print(f'\n  best bigram {best_key} at {best:.5f}', flush=True)
    print(f'  beats the best 36-site program ({BEST_PROGRAM_CE[ho]:.5f}) -> {pa}', flush=True)
    print(f'  beats the all-tabled baseline ({ALL_TABLED_CE[ho]:.5f}) -> {pb}', flush=True)
    print(f'  does NOT beat the live model ({live[ho]:.5f}) -> {pc}', flush=True)
    print(f'  live CEs + coverage {ncov} + bigram beats unigram ({out["unigram"][ho]:.5f}) -> '
          f'control {pd}', flush=True)

    r = {'config': {'fit_rows': f'{fit.shape[0]}x{T} = {fit.shape[0] * T} tokens',
                    'smoothing': 'add-alpha with unigram backoff, alpha swept over ' + str(ALPHAS),
                    'scoring': 'covered positions from 64, identical to every published figure',
                    'WHY': '§1765 showed the installed 36-site program is a pure function of the '
                           'current token, so a bigram on the same fit rows is the floor it should '
                           'have been measured against since §1662.',
                    'DATA_LIMITATION': '24,576 fit tokens is very little for a bigram over 5,419 '
                                       'covered types; alpha is swept so one smoothing choice does '
                                       'not decide the comparison.',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'live_ce_covered': {e: round(v, 5) for e, v in live.items()},
         'reference': {'best_36_site_program_ce': BEST_PROGRAM_CE, 'all_tabled_ce': ALL_TABLED_CE},
         'null_models': out, 'best_bigram': best_key,
         'predictions': {'pred_a_bigram_beats_best_program': bool(pa),
                         'pred_b_bigram_beats_all_tabled': bool(pb),
                         'pred_c_bigram_loses_to_live': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
