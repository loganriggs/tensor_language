# A BETTER MEMBER OF THE POSITION-WISE CLASS: give the model a neutral prefix.
#
# §1772: temperature and blending with corpus statistics both fail to improve on the length-1 model,
# which stands as the tightest upper bound on the position-wise optimum (5.97902 held out). §1771b's
# open question is what a fundamentally better member looks like.
#
# Here is one. A length-1 forward is far off the distribution the model was trained on -- 256-token
# contexts. Run instead on `[pad] * k + [token]` and read the LAST position. For any FIXED k and
# fixed pad, that is still a function of the current token alone, so it is a member of the class; but
# the model sees a sequence of realistic length and its own normalisation statistics are closer to
# what it was trained with. If a neutral prefix calibrates it, the bound tightens.
#
# TWO PAD CHOICES, because the pad is not neutral and pretending otherwise would be the error:
#   token 0 -- an arbitrary fixed id
#   the most frequent token in the fit rows -- the most "ordinary" context filler available
# Whether the choice matters is itself a registered question.
#
# ROLES. k and the pad are selected on skip7000; the bound is read on skip11000. Covered positions
# from 64. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, with MARGINS per LESSON 40 (the run's own control
# tolerance is 0.001, so no arm is decided below that), each read back against its sentence per
# LESSON 39:
#   pred_a A PREFIX TIGHTENS THE BOUND BY AT LEAST 0.01 NATS: the selected (k, pad) scores below
#          5.97902 - 0.01 on skip11000. If FALSE, a neutral prefix does not help and the length-1
#          model's calibration was not the limitation -- which, after §1772's temperature null, would
#          make the length-1 point look genuinely hard to beat inside this class.
#   pred_b THE OPTIMUM IS INTERIOR IN k: the best k on skip7000 is neither 0 nor the largest tested.
#          If FALSE and the best is the largest k, the sweep is under-budgeted -- the defect §1756,
#          §1770 and §1771 each recorded -- and I report that rather than a design point.
#   pred_c THE PAD CHOICE MATTERS by at least 0.01 nats at the best k: the two pads differ on
#          skip11000. If FALSE the prefix acts as generic length rather than as content, which is the
#          more interesting reading and means any filler would do.
#   pred_d CONTROLS: the k=0 arm reproduces §1768's 5.97902 and 6.03465 within 0.001 -- it is the
#          length-1 model by construction -- and coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
KS = (0, 1, 2, 4, 8, 16, 32, 64)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/prefix_padded_position_wise_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1768_REF = {'skip7000': 6.03465, 'skip11000': 5.97902}
SELECT, HOLD = 'skip7000', 'skip11000'
MARGIN = 0.01


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
    flat = fit[:, :T].reshape(-1).long()
    common = int(torch.bincount(flat, minlength=V).argmax())
    pads = {'token0': 0, 'most_common': common}
    print(f'PREFIX-PADDED POSITION-WISE | k in {KS} | pads {pads} | selected on {SELECT}, bound on '
          f'{HOLD} | DISCOVERY ONLY', flush=True)

    ev = {}
    for ename, epath in EVAL_SETS:
        e = load(epath)
        ids, tgs = [], []
        for i in range(0, e.shape[0], 8):
            bb = e[i:i + 8]
            idx = bb[:, :-1].to(DEV)[:, 64:]
            tg = bb[:, 1:].to(DEV)[:, 64:]
            c = seen[idx]
            ids.append(idmap[idx][c]); tgs.append(tg[c])
        ev[ename] = (torch.cat(ids), torch.cat(tgs))

    def score(lut):
        out = {}
        for ename, (r, tg) in ev.items():
            tot, n, B = 0.0, 0, 8192
            for i in range(0, r.numel(), B):
                rr, tt = r[i:i + B], tg[i:i + B]
                v = lut[rr].gather(-1, tt.unsqueeze(-1)).squeeze(-1)
                tot += float(-v.double().sum()); n += rr.numel()
            out[ename] = tot / n
        return out

    res = {}
    for pname, pid in pads.items():
        for k in KS:
            lut = torch.zeros(ncov, W, device=DEV)
            for i in range(0, ncov, 256):
                t = toks[i:i + 256].to(DEV).unsqueeze(1)
                seq = torch.cat([torch.full((t.shape[0], k), pid, device=DEV,
                                            dtype=t.dtype), t], 1) if k else t
                lut[i:i + t.shape[0]] = torch.log_softmax(forward_logits(seq)[:, -1].float(), -1)
            s = score(lut)
            res[f'{pname}_k{k}'] = {'pad': pname, 'pad_id': pid, 'k': k,
                                    **{e: round(v, 5) for e, v in s.items()}}
            print(f'  {pname:12s} k {k:3d}: {SELECT} {s[SELECT]:.5f}  {HOLD} {s[HOLD]:.5f}   '
                  f'[{time.time() - t0:.0f}s]', flush=True)
            del lut
            torch.cuda.empty_cache()

    best = min(res, key=lambda kk: res[kk][SELECT])
    b = res[best]
    other_pad = 'most_common' if b['pad'] == 'token0' else 'token0'
    twin = res[f'{other_pad}_k{b["k"]}']
    k0 = res['token0_k0']

    pa = b[HOLD] < S1768_REF[HOLD] - MARGIN
    pb = b['k'] not in (KS[0], KS[-1])
    pc = abs(b[HOLD] - twin[HOLD]) >= MARGIN
    pd = (all(abs(k0[e] - v) <= 0.001 for e, v in S1768_REF.items()) and ncov == NCOV)

    print(f'\n  selected on {SELECT}: {best} -> {HOLD} {b[HOLD]:.5f}', flush=True)
    print(f'  tightens 5.97902 by at least {MARGIN} -> {pa} '
          f'(by {S1768_REF[HOLD] - b[HOLD]:+.5f})', flush=True)
    print(f'  best k={b["k"]} is interior in {KS} -> {pb}', flush=True)
    print(f'  the two pads differ at k={b["k"]} by {abs(b[HOLD] - twin[HOLD]):.5f} '
          f'(>= {MARGIN}) -> {pc}', flush=True)
    print(f'  k=0 reproduces §1768 ({k0[SELECT]:.5f} / {k0[HOLD]:.5f}) + coverage {ncov} -> '
          f'control {pd}', flush=True)

    r = {'config': {'k_values': list(KS), 'pads': pads, 'margin': MARGIN,
                    'construction': 'the model run on [pad]*k + [token], reading the LAST position. '
                                    'For fixed k and fixed pad this is a function of the current '
                                    'token alone, hence a member of the position-wise class.',
                    'selection': f'k and pad selected on {SELECT}; the bound is read on {HOLD}',
                    'WHY': '§1772 found temperature and corpus blending both fail to beat the '
                           'length-1 model. §1771b asks what a fundamentally better member is. A '
                           'length-1 forward is far off the training distribution; a neutral prefix '
                           'of realistic length is the cheapest way to move it back on.',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'arms': res, 'selected': best,
         'new_upper_bound_on_class_optimum': round(b[HOLD], 5),
         'previous_bound_S1768': S1768_REF[HOLD],
         'tightened_by': round(S1768_REF[HOLD] - b[HOLD], 5),
         'predictions': {'pred_a_prefix_tightens_by_0p01': bool(pa),
                         'pred_b_interior_k': bool(pb),
                         'pred_c_pad_choice_matters': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
