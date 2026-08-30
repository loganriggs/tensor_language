# THE FULL CAUSAL BATTERY ON EVERY CURATED CIRCUIT, NOT JUST THE TWELVE.
#
# TASK CONTEXT (Logan, 2026-08-30): "get as much useful info on as many circuits as you can, hopefully
# having like 35 candidate circuits. Use ablations and ... interchange patching and DAS ... to isolate
# specifically where it is located." Codex works the same folder on ten behaviorally-named circuits
# (induction, brackets, quotes, ...); this lane is the 70-circuit census, so the two do not overlap.
#
# WHY THIS RUN. §2054 mean-ablated all 36 components and §2055 interchange-ablated all 36, but BOTH
# scripts scored only the twelve circuits that lacked a `components` field. The 36 dCE vectors do not
# depend on which circuit is scored -- they were already computed over the whole grid. Scoring all 70
# curated circuits against the same vectors therefore costs ZERO extra sweeps and yields 5.8x the
# localisation. That is the cheapest large gain available in this folder.
#
# WHAT IT DOES. Two methods, same 36 components, every curated circuit present in the census state:
#   1. MEAN ablation  -- replace a component's output with its grid mean (removes the signal).
#   2. INTERCHANGE    -- replace it with its output at a random other grid position, fixed seed
#                        (resample ablation: substitutes a DIFFERENT valid signal, the classic
#                        interchange intervention; damage means the position-specific value matters).
# Concentration = mean|dCE| on the circuit's members / mean|dCE| off its slice.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  At least 35 curated circuits have a best component with mean-ablation concentration >= 2.0 --
#           i.e. Logan's target of 35 well-localised candidates is reachable from the census alone. If
#           FALSE the census does not contain 35 localisable circuits and the target needs new circuits,
#           not better scoring of old ones.
#   pred_b  The two methods pick the SAME best component for at least 60% of circuits. They are different
#           interventions (mean removes signal, interchange substitutes signal) and agreement is evidence
#           the localisation is a property of the circuit rather than of the intervention. If FALSE, every
#           single-method localisation in this folder -- including §2054's twelve -- is method-dependent
#           and must be reported as such.
#   pred_c  Interchange concentration EXCEEDS mean-ablation concentration for the majority of circuits.
#           Interchange injects an off-distribution activation for that position while the mean is a
#           bland in-distribution value, so it should hit position-specific circuits harder. Concentration
#           is a ratio, so this is a genuine question and not an arithmetic certainty.
#
# Writes circuits/BATTERY.json (read-only artifact; modifies no circuit file, because Codex is working
# the same folder and a shared read-only artifact cannot collide with them).
import json
import os
import time

import torch

import census_lib as C

KEYS = [f'{k}{L}' for k in ('a', 'm') for L in range(18)]
SEED = 20260830

CUR = []
for fn in sorted(os.listdir('circuits')):
    if not fn.endswith('.json') or fn.split('.')[0].isupper():
        continue
    try:
        d = json.load(open('circuits/' + fn))
    except Exception:
        continue
    if isinstance(d, dict) and 'tag' in d:
        CUR.append(d['tag'])

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()

TAGS, missing = [], []
for t in CUR:
    try:
        lf = C.leaf(t)
        if len(lf['member']) > 0 and len(lf['slice']) > len(lf['member']):
            TAGS.append(t)
        else:
            missing.append((t, 'empty members or slice==members'))
    except Exception:
        missing.append((t, 'absent from census state'))

masks = {}
for t in TAGS:
    lf = C.leaf(t)
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    masks[t] = (mm, sl)
print(f'grid {nflat} positions, base CE {base.mean():.4f}', flush=True)
print(f'curated {len(CUR)} -> scorable {len(TAGS)}; unusable {len(missing)}', flush=True)


@torch.no_grad()
def interchange_dce(key):
    """dCE when `key`'s output at every position is replaced by its output at a random other position."""
    R = C.rows()
    cap = []
    h = C.MODS[key].register_forward_hook(
        lambda mo, i_, o_: cap.append(((o_[0] if isinstance(o_, tuple) else o_)
                                       .detach().float().reshape(-1, C.D).cpu())))
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    acts = torch.cat(cap)
    g = torch.Generator().manual_seed(SEED)
    donor = acts[torch.randperm(acts.shape[0], generator=g)]

    ces, off = [], 0
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        n = bb.shape[0] * 256
        rep = donor[off:off + n].to(C.DEV); off += n

        def fh(mo, i_, o_, rep=rep):
            if isinstance(o_, tuple):
                y, v1 = o_
                return (rep.view_as(y).to(y.dtype), v1)
            return rep.view_as(o_).to(o_.dtype)

        hh = C.MODS[key].register_forward_hook(fh)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = torch.nn.functional.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
        for blkm in C.m.transformer.h:
            x, v1 = blkm(x, v1, x0)
        lg = (30 * torch.tanh(C.m.lm_head(torch.nn.functional.rms_norm(x, (C.D,))) / 30)).float()
        ces.append(torch.nn.functional.cross_entropy(
            lg.view(-1, lg.size(-1)), tg, reduction='none').cpu())
        hh.remove()
    return torch.cat(ces).float() - base


def score(d):
    o = {}
    for t in TAGS:
        mm, sl = masks[t]
        am = float(d[mm].abs().mean()); ag = float(d[~sl].abs().mean())
        o[t] = {'abs_dce_members': round(am, 4), 'abs_dce_offslice': round(ag, 4),
                'concentration': round(am / ag, 3) if ag > 0 else None,
                'signed_dce_members': round(float(d[mm].mean()), 4)}
    return o


t0 = time.time()
MEAN, INTER = {}, {}
for i, key in enumerate(KEYS):
    for t, v in score(C.ce_sweep(C.mean_hooks([key])) - base).items():
        MEAN.setdefault(t, {})[key] = v
    print(f'  mean [{i+1:2d}/36] {key}  ({time.time()-t0:.0f}s)', flush=True)
for i, key in enumerate(KEYS):
    for t, v in score(interchange_dce(key)).items():
        INTER.setdefault(t, {})[key] = v
    print(f'  intr [{i+1:2d}/36] {key}  ({time.time()-t0:.0f}s)', flush=True)


def top(per, n=6):
    r = sorted(((v['concentration'], k) for k, v in per.items() if v['concentration'] is not None),
               reverse=True)
    return [{'component': k, **per[k]} for _c, k in r[:n]]


rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude, circuit task (Logan)',
       'method': 'mean-ablation and interchange (resample, seed 20260830) of each of the 36 components '
                 'over the census grid; concentration = mean|dCE| on members / mean|dCE| off slice',
       'state': 'census_state_diverse.pt', 'seed': SEED, 'grid_positions': nflat,
       'note': 'read-only artifact; no circuit file was modified',
       'unusable': missing, 'by_tag': {}}
for t in TAGS:
    tm, ti = top(MEAN[t]), top(INTER[t])
    rep['by_tag'][t] = {'mean_ablation': {'top': tm, 'all': MEAN[t]},
                        'interchange': {'top': ti, 'all': INTER[t]},
                        'best_mean': tm[0]['component'] if tm else None,
                        'best_interchange': ti[0]['component'] if ti else None,
                        'methods_agree': bool(tm and ti and tm[0]['component'] == ti[0]['component'])}
json.dump(rep, open('circuits/BATTERY.json', 'w'), indent=1)

loc = [t for t in TAGS if MEAN[t] and (top(MEAN[t])[0]['concentration'] or 0) >= 2.0]
agree = [t for t in TAGS if rep['by_tag'][t]['methods_agree']]
hi = [t for t in TAGS if (top(INTER[t])[0]['concentration'] or 0) > (top(MEAN[t])[0]['concentration'] or 0)]
print(f'\nwrote circuits/BATTERY.json  ({time.time()-t0:.0f}s)', flush=True)
print(f'pred_a  circuits with best mean-ablation concentration >=2.0: {len(loc)}/{len(TAGS)}  '
      f'(bar >=35) : {len(loc) >= 35}')
print(f'pred_b  methods pick the same best component: {len(agree)}/{len(TAGS)} '
      f'= {100*len(agree)/max(1,len(TAGS)):.0f}%  (bar >=60%) : {len(agree) >= 0.6*len(TAGS)}')
print(f'pred_c  interchange concentration > mean-ablation: {len(hi)}/{len(TAGS)}  '
      f'(bar majority) : {len(hi) > len(TAGS)/2}')
print('\ntop 40 by mean-ablation concentration:')
for c, t in sorted(((top(MEAN[t])[0]['concentration'] or 0, t) for t in TAGS), reverse=True)[:40]:
    b = rep['by_tag'][t]
    print(f"  {t:12s} {b['best_mean']:4s} conc {c:6.2f} | interchange {b['best_interchange']:4s} "
          f"{top(INTER[t])[0]['concentration']:6.2f} | agree {b['methods_agree']}")
