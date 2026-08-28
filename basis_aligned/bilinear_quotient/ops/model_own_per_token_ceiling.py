# THE EXACT CEILING OF THE POSITION-WISE CLASS, WITH NO ESTIMATION
#
# §1765: a fully-installed table+linear program is a pure function of the current token -- every
# cross-position Jacobian is zero, measured at exactly 0.000e+00. Codex pruned the class on that
# derivation. §1767 then tried to bound how much fidelity the class still holds, using bigrams
# estimated on the eval rows, and FAILED: the program at 6.57289 beats the leave-one-out bigram at
# 7.33406, because 27k tokens cannot estimate P(next | current) as well as the model already knows
# it. Three estimators spanned 4.76 nats and none bracketed the answer.
#
# The ceiling needs no estimator. The best per-token function THIS MODEL can express is the model's
# own output when it is given only the current token: run each covered token as a length-1 sequence
# and read its logits. Exact, no smoothing, no corpus.
#
# WHY POSITION 0 IS THE RIGHT PLACE TO READ IT, and the caveat. There is no additive positional
# embedding; rotary acts inside attention, and for a position attending only to ITSELF the relative
# rotation is zero, so a self-only forward is position-independent. A length-1 sequence is exactly a
# self-only forward. The caveat that remains is RMSNorm and any length-dependent normalisation
# statistics, which a length-1 batch cannot rule out, so this is stated as "the model's per-token
# function evaluated at position 0" and not as a proven position-invariant.
#
# ROLES. Both eval roles, scored on covered positions from 64 to match every published figure.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a IT IS ACTUALLY A CEILING: the model's own per-token CE is WORSE than (above) the best
#          36-site program's 6.57289 on skip11000. If FALSE the compiled program beats the model at
#          its own per-token game, which would mean the tables carry something the position-0 forward
#          does not -- and this construction would not be the ceiling I am calling it. That is the
#          same failure §1767 had, and it is the first thing to check rather than the last.
#   pred_b THE CLASS IS NEARLY EXHAUSTED: the gap from the program to the ceiling is at most 0.5
#          nats. This is the number Codex's prune needs. If FALSE there is real fidelity left inside
#          the position-wise grammar, and the prune -- right about context -- is leaving it behind.
#   pred_c CONTEXT DOMINATES: the ceiling sits at least 2.0 nats above the live model's 3.09711.
#          Scored independently of pred_a and pred_b. This is the quantity §1767's pred_b tried to
#          measure with a data-limited estimator and could not.
#   pred_d CONTROLS: coverage is exactly 5419 of 50257; EVERY scored covered position finds its token
#          in the lookup, so no fallback is used and the ceiling is not diluted by one; and the live
#          CE reproduces 3.29205 and 3.09711.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257
# the checkpoint's head is 50,304 wide; the log-softmax MUST be taken over all of it, because
# slicing to the tokenizer's 50,257 would change the normalisation and therefore the CE.
W = 50304
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/model_own_per_token_ceiling_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
BEST_PROGRAM_CE = {'skip7000': 7.35114 - 0.77602, 'skip11000': 7.35825 - 0.78536}
ALL_TABLED_CE = {'skip7000': 7.35114, 'skip11000': 7.35825}
S1767_LOO = {'skip7000': 7.29459, 'skip11000': 7.33406}


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
def live_ce(rows, seen):
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward_logits(idx)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        c = seen[idx[:, 64:]]
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
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    print(f'MODEL-OWN PER-TOKEN CEILING | {ncov} covered tokens as length-1 sequences | '
          f'DISCOVERY ONLY', flush=True)

    # the model's own per-token log-probabilities: one length-1 forward per covered token.
    # indexed by COMPACT row id (5419 x 50257 fp16 = 0.54 GB) rather than by token id (5.05 GB),
    # because only covered tokens are ever looked up and the GPU is shared.
    idmap = torch.full((V,), -1, dtype=torch.long)
    idmap[toks] = torch.arange(ncov)
    idmap = idmap.to(DEV)
    lp = torch.zeros(ncov, W, dtype=torch.float16, device=DEV)
    assert m.lm_head.weight.shape[0] == W, f'head width {m.lm_head.weight.shape[0]} != {W}'
    for i in range(0, ncov, 256):
        t = toks[i:i + 256].to(DEV).unsqueeze(1)          # [b, 1]
        lg = forward_logits(t)[:, 0].float()              # [b, V]
        lp[i:i + t.shape[0]] = torch.log_softmax(lg, -1).half()
    print(f'  built the per-token lookup ({time.time() - t0:.0f}s)', flush=True)

    out = {}
    for ename, epath, ref in EVAL_SETS:
        ev = load(epath)
        lv = live_ce(ev, seen)
        assert abs(lv - ref) <= 1e-3, f'{ename} live CE {lv:.5f} != {ref}'
        tot, cnt, miss = 0.0, 0, 0
        for i in range(0, ev.shape[0], 8):
            bb = ev[i:i + 8]
            idx = bb[:, :-1].to(DEV)[:, 64:]
            tg = bb[:, 1:].to(DEV)[:, 64:]
            c = seen[idx]
            r = idmap[idx]
            miss += int((r[c] < 0).sum())
            v = lp[r.clamp(min=0)].float().gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            tot += float((-v.double())[c].sum()); cnt += int(c.sum())
        ceil = tot / cnt
        out[ename] = {'live_ce': round(lv, 5), 'model_own_per_token_ce': round(ceil, 5),
                      'best_program_ce': round(BEST_PROGRAM_CE[ename], 5),
                      'all_tabled_ce': ALL_TABLED_CE[ename],
                      'loo_bigram_S1767': S1767_LOO[ename],
                      'scored': cnt, 'lookup_misses': miss,
                      'gap_program_to_ceiling': round(BEST_PROGRAM_CE[ename] - ceil, 5),
                      'gap_ceiling_to_live': round(ceil - lv, 5)}
        o = out[ename]
        print(f'\n  {ename}: {cnt} scored, {miss} lookup misses', flush=True)
        print(f'    live model                {o["live_ce"]:.5f}', flush=True)
        print(f'    MODEL-OWN per-token       {o["model_own_per_token_ce"]:.5f}   <- the ceiling',
              flush=True)
        print(f'    best 36-site program      {o["best_program_ce"]:.5f}', flush=True)
        print(f'    all-tabled baseline       {o["all_tabled_ce"]:.5f}', flush=True)
        print(f'    LOO bigram (§1767)        {o["loo_bigram_S1767"]:.5f}', flush=True)
        print(f'    program is {o["gap_program_to_ceiling"]:+.5f} from the ceiling; the ceiling is '
              f'{o["gap_ceiling_to_live"]:+.5f} above live', flush=True)
        del ev

    ho = 'skip11000'
    pa = out[ho]['model_own_per_token_ce'] > out[ho]['best_program_ce']
    pb = out[ho]['gap_program_to_ceiling'] <= 0.5 and out[ho]['gap_program_to_ceiling'] >= 0.0
    pc = out[ho]['gap_ceiling_to_live'] >= 2.0
    pd = (ncov == NCOV and all(out[e]['lookup_misses'] == 0 for e in out)
          and abs(out['skip7000']['live_ce'] - 3.29205) <= 1e-3
          and abs(out[ho]['live_ce'] - 3.09711) <= 1e-3)

    print(f'\n  the ceiling is WORSE than the program, so it is a ceiling '
          f'({out[ho]["model_own_per_token_ce"]:.5f} vs {out[ho]["best_program_ce"]:.5f}) -> {pa}',
          flush=True)
    print(f'  the program is within 0.5 nats of it ({out[ho]["gap_program_to_ceiling"]:+.5f}) -> '
          f'class nearly exhausted {pb}', flush=True)
    print(f'  the ceiling is >=2.0 above live ({out[ho]["gap_ceiling_to_live"]:+.5f}) -> context '
          f'dominates {pc}', flush=True)
    print(f'  coverage {ncov}, zero lookup misses, live CEs reproduce -> control {pd}', flush=True)

    r = {'config': {'construction': 'each covered token run as a LENGTH-1 sequence; its logits are '
                                    'the model\'s own prediction given only that token',
                    'position_caveat': 'no additive positional embedding, and rotary cancels for a '
                                       'position attending only to itself, so a self-only forward is '
                                       'position-independent. RMSNorm/length-dependent statistics are '
                                       'NOT ruled out by a length-1 batch, so this is the model\'s '
                                       'per-token function evaluated at position 0.',
                    'scoring': 'covered positions from 64, matching every published figure',
                    'logit_width': f'log_softmax over the full {W}-wide head; slicing to 50257 '
                                   'would change the normalisation and the CE',
                    'WHY': '§1767 tried to bound the position-wise class with bigrams estimated on '
                           'the eval rows and the program BEAT them, because 27k tokens cannot '
                           'estimate P(next|current) as well as the model already knows it. This '
                           'construction needs no estimator.',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'results': out,
         'predictions': {'pred_a_it_is_a_ceiling': bool(pa),
                         'pred_b_class_nearly_exhausted': bool(pb),
                         'pred_c_context_dominates': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
