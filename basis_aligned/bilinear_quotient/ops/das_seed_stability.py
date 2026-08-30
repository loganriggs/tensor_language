# ARE §2060'S PUBLISHED DAS NUMBERS SEED-STABLE? A RUNG-2 CHECK ON MY OWN REGISTRY ENTRY.
#
# RUNG 2: second-class confirmation of a certified result -- and the result being checked is mine.
#
# §2070 measured, for the first time, the seed-to-seed spread of these DAS fits: the sd of r.1.1.1's
# held-out margin at m16 is 0.0461, forty-six times the 0.0010 that §2069's single fit cleared its bar by,
# with spreads across six circuits running 0.029 to 0.319. That was measured at m16, on MARGINS.
#
# §2060 is a published section and a registry entry (DAS_CIRCUITS_ARE_ENRICHED_BUT_NOT_LOW_DIMENSIONAL),
# and every number in it comes from ONE initialisation: the overlaps with the closed-form direction
# (0.006-0.336, quoted as "up to 390x random but never near identity"), the concentrations, and the
# recovery fractions. Whether those carry comparable spread is untested, and §2070 makes it irresponsible
# to keep assuming they do not. If they do, the sentence "DAS and the mean-difference probe find
# overlapping but genuinely different directions" -- which settled §2058's caveat and which I told Codex
# twice -- rests on unreplicated point estimates.
#
# Three seeds per circuit, on §2060's own ten circuits, reporting spread for the three quantities §2060
# published.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  The overlaps carry real seed noise: the sd of overlap_with_closed_form across seeds is >= 0.05
#           for at least half the circuits. Registered in the direction §2070 makes likely. If FALSE these
#           fits are more reproducible on overlap than on margin, which would be worth knowing and would
#           leave §2060 stronger than I currently think it is.
#   pred_b  §2060's CONCLUSION survives regardless of pred_a: the MEAN overlap across seeds stays below
#           0.50 for every circuit, so "DAS and the closed form find overlapping but genuinely different
#           directions" holds on averaged evidence rather than on one draw. If FALSE, a circuit whose mean
#           overlap exceeds 0.5 would mean the two methods DO agree there and §2060's headline needs
#           narrowing -- which I would report as a correction to a published claim.
#   pred_c  CONTROL: every fit passes the optimiser health gate. LESSON 108; a spread measured over fits
#           that did not train is a spread of nothing.
#
# Writes circuits/DAS_SEED_STABILITY.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

# PLAN PRE-FLIGHT (LESSON 109): census_lib builds MODS at import, so the gate must be answered first.
if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json', 'circuits/DAS.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: state, BATTERY.json and S2060 DAS.json present')
    raise SystemExit(0)

import torch
import torch.nn.functional as F

import census_lib as C

RANKS = (1,)
STEPS = 400
LR = 5e-2
BATCH = 4
SEED = 20260830
SEEDS = (20260830, 20260831, 20260832)
TRAIN_ROWS = (0, 600)
EVAL_ROWS = (600, 1000)
A8 = ['r.11.1.1', 'r.11.1.2', 'r.11.3.1', 'r.23.2.1', 'r.23.2.3']
N_EXTRA = 5

C.use_state('census_state_diverse.pt')
for p in C.m.parameters():
    p.requires_grad_(False)
R = C.rows()
NP = C.T                                                    # scored positions per row

BAT = json.load(open('circuits/BATTERY.json'))
rank_all = sorted(((v['mean_ablation']['top'][0]['concentration'] or 0, t)
                   for t, v in BAT['by_tag'].items() if v['mean_ablation']['top']), reverse=True)
TARGETS = [t for t in A8 if t in BAT['by_tag']]
for _c, t in rank_all:
    if len(TARGETS) >= len(A8) + N_EXTRA:
        break
    if t not in TARGETS:
        TARGETS.append(t)
print(f'DAS targets ({len(TARGETS)}): {TARGETS}', flush=True)

# NOTE. circuits/SUBSPACE.json stores only cosines and concentrations, not the direction vectors, so
# the closed-form comparison arm is recomputed here -- and recomputed AT THE SAME COMPONENT DAS runs at,
# which makes the comparison internally valid even where a circuit's best component is not §2056's a8.


def leaf_masks(tag):
    lf = C.leaf(tag)
    mm = torch.zeros(len(R) * NP, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(len(R) * NP, dtype=torch.bool); sl[lf['slice']] = True
    return mm.view(len(R), NP), sl.view(len(R), NP)


def fwd(idx, tg, key=None, Q=None, donor=None, full_donor=None):
    """forward with an optional interchange at `key`; along Q if given, on the whole output if not."""
    h = None
    if key is not None:
        def fh(mo, i_, o_):
            y = o_[0] if isinstance(o_, tuple) else o_
            f = y.float()
            if full_donor is not None:
                f = full_donor.view_as(f)
            else:
                co = f.reshape(-1, C.D) @ Q
                cd = donor.reshape(-1, C.D) @ Q
                f = (f.reshape(-1, C.D) + (cd - co) @ Q.T).view_as(f)
            f = f.to(y.dtype)
            return (f, o_[1]) if isinstance(o_, tuple) else f
        h = C.MODS[key].register_forward_hook(fh)
    x = F.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
    for blk in C.m.transformer.h:
        x, v1 = blk(x, v1, x0)
    lg = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
    if h is not None:
        h.remove()
    return F.cross_entropy(lg.view(-1, lg.size(-1)), tg.reshape(-1), reduction='none').view(idx.shape)


@torch.no_grad()
def capture(key, lo, hi):
    cap = []
    h = C.MODS[key].register_forward_hook(
        lambda mo, i_, o_: cap.append(((o_[0] if isinstance(o_, tuple) else o_)
                                       .detach().float().cpu())))
    for i in range(lo, hi, BATCH):
        bb = R[i:i + BATCH, :NP + 1].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    return torch.cat(cap)


def batches(lo, hi, mm=None):
    """row batches; when mm is given, only batches containing at least one circuit member.

    A circuit's members are ~0.3% of the grid, so a blind batch of 4 rows carries about three of them and
    the member half of the objective is estimated from three positions. Skipping member-free batches puts
    every gradient step on a batch that can actually see the circuit.
    """
    for i in range(lo, hi, BATCH):
        if mm is not None and mm[i:i + BATCH].sum() == 0:
            continue
        bb = R[i:i + BATCH, :NP + 1].to(C.DEV)
        yield i, bb[:, :-1].contiguous(), bb[:, 1:].contiguous()


def closed_form_dir(acts, mm, sl, lo, hi):
    """§2056's probe: unit(mean over members - mean off slice), at this component, on the train rows."""
    f = acts.reshape(-1, C.D)
    m_ = mm[lo:hi].reshape(-1); s_ = sl[lo:hi].reshape(-1)
    if m_.sum() == 0 or (~s_).sum() == 0:
        return None
    u = f[m_].mean(0) - f[~s_].mean(0)
    return (u / u.norm()).to(C.DEV)


def evaluate(tag, key, mm, sl, Q, acts, lo, hi):
    """held-out mean dCE on members and off slice, for a Q-interchange and for the full-output one."""
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(acts.shape[0] * NP, generator=g)
    flat = acts.reshape(-1, C.D)
    dq_m, dq_o, df_m, df_o = [], [], [], []
    with torch.no_grad():
        for i, idx, tg in batches(lo, hi):
            k = (i - lo) // BATCH
            n = idx.shape[0] * NP
            dn = flat[perm[k * BATCH * NP:k * BATCH * NP + n]].to(C.DEV).view(idx.shape[0], NP, C.D)
            b0 = fwd(idx, tg)
            m_, s_ = mm[i:i + idx.shape[0]].to(C.DEV), sl[i:i + idx.shape[0]].to(C.DEV)
            if Q is not None:
                d = fwd(idx, tg, key, Q, dn) - b0
                dq_m.append(d[m_]); dq_o.append(d[~s_])
            d = fwd(idx, tg, key, full_donor=dn) - b0
            df_m.append(d[m_]); df_o.append(d[~s_])
    f = lambda L: float(torch.cat(L).abs().mean()) if L and sum(x.numel() for x in L) else float('nan')
    return f(dq_m), f(dq_o), f(df_m), f(df_o)


PER_SEED = {}
t0 = time.time()
for ACTIVE_SEED in SEEDS:
  out = {}
  for tag in TARGETS:
      key = BAT['by_tag'][tag]['best_mean']
      mm, sl = leaf_masks(tag)
      acts_tr = capture(key, *TRAIN_ROWS)
      acts_ev = capture(key, *EVAL_ROWS)
      rec = {'component': key, 'members': int(mm.sum()), 'ranks': {}}
      for r in RANKS:
          g = torch.Generator(device='cpu').manual_seed(ACTIVE_SEED + r)
          # unit-scale init: |P| ~ 1 so an lr 5e-2 Adam step is a real rotation, not a 1e-4 nudge
          P0 = (torch.randn(C.D, r, generator=g) / C.D ** 0.5).to(C.DEV)
          Q_init = torch.linalg.qr(P0)[0].detach().clone()
          P = P0.clone().requires_grad_(True)
          opt = torch.optim.Adam([P], lr=LR)
          gg = torch.Generator().manual_seed(ACTIVE_SEED)
          perm = torch.randperm(acts_tr.shape[0] * NP, generator=gg)
          flat = acts_tr.reshape(-1, C.D)
          step = 0
          losses = []
          while step < STEPS:
              for i, idx, tg in batches(*TRAIN_ROWS, mm=mm):
                  if step >= STEPS:
                      break
                  m_, s_ = mm[i:i + idx.shape[0]].to(C.DEV), sl[i:i + idx.shape[0]].to(C.DEV)
                  if m_.sum() == 0:
                      continue
                  k = (i - TRAIN_ROWS[0]) // BATCH
                  n = idx.shape[0] * NP
                  dn = flat[perm[k * BATCH * NP:k * BATCH * NP + n]].to(C.DEV).view(
                      idx.shape[0], NP, C.D)
                  Q = torch.linalg.qr(P)[0]
                  with torch.no_grad():
                      b0 = fwd(idx, tg)
                  d = fwd(idx, tg, key, Q, dn) - b0
                  loss = -d[m_].mean() + d[~s_].abs().mean()
                  opt.zero_grad(); loss.backward(); opt.step()
                  losses.append(float(loss))
                  step += 1
          with torch.no_grad():
              Q = torch.linalg.qr(P)[0]
              moved = 1.0 - float((Q.T @ Q_init).pow(2).sum() / r)      # 0 = never moved, 1 = orthogonal
          first, last = (sum(losses[:20]) / 20, sum(losses[-20:]) / 20) if len(losses) >= 40 else (0., 0.)
          healthy = moved > 0.02 and last < first
          qm, qo, fm, fo = evaluate(tag, key, mm, sl, Q, acts_ev, *EVAL_ROWS)
          ent = {'das_dce_members': round(qm, 4), 'das_dce_offslice': round(qo, 4),
                 'das_concentration': round(qm / qo, 3) if qo > 0 else None,
                 'full_dce_members': round(fm, 4), 'full_dce_offslice': round(fo, 4),
                 'full_concentration': round(fm / fo, 3) if fo > 0 else None,
                 'fraction_of_full_recovered': round(qm / fm, 3) if fm > 0 else None,
                 'subspace_moved_from_init': round(moved, 4),
                 'loss_first20': round(first, 6), 'loss_last20': round(last, 6),
                 'optimiser_healthy': bool(healthy)}
          if r == 1:
              u = closed_form_dir(acts_tr, mm, sl, *TRAIN_ROWS)
              if u is not None:
                  ent['overlap_with_closed_form'] = round(float((Q[:, 0] @ u) ** 2), 3)
                  cm, co, _fm, _fo = evaluate(tag, key, mm, sl, u.unsqueeze(1), acts_ev, *EVAL_ROWS)
                  ent['closed_form_dce_members'] = round(cm, 4)
                  ent['closed_form_concentration'] = round(cm / co, 3) if co > 0 else None
                  ent['das_beats_closed_form'] = bool(qm > cm)
          rec['ranks'][r] = ent
          print(f'  {tag:12s} {key:4s} rank {r}: members {qm:.4f} off {qo:.4f} conc '
                f'{ent["das_concentration"]} recovered {ent["fraction_of_full_recovered"]} | '
                f'moved {moved:.3f} loss {first:+.5f}->{last:+.5f} healthy {healthy} '
                f'({time.time()-t0:.0f}s)', flush=True)
      out[tag] = rec


  PER_SEED[ACTIVE_SEED] = out
  print(f'  --- seed {ACTIVE_SEED} done ({time.time()-t0:.0f}s)', flush=True)

def agg(field):
    o = {}
    for tag in TARGETS:
        vals = [PER_SEED[sd][tag]['ranks'][1].get(field) for sd in SEEDS
                if tag in PER_SEED[sd] and PER_SEED[sd][tag]['ranks'][1].get(field) is not None]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        sdv = (sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
        o[tag] = {'values': [round(v, 4) for v in vals], 'mean': round(m, 4), 'sd': round(sdv, 4)}
    return o


OV = agg('overlap_with_closed_form')
CN = agg('das_concentration')
FR = agg('fraction_of_full_recovered')
unhealthy = [(sd, t, r) for sd in PER_SEED for t, v in PER_SEED[sd].items()
             for r, e in v['ranks'].items() if not e.get('optimiser_healthy')]
noisy = [t for t, v in OV.items() if v['sd'] >= 0.05]
over_half = [t for t, v in OV.items() if v['mean'] >= 0.50]

rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude', 'seeds': list(SEEDS),
       'checks': 'S2060 / registry DAS_CIRCUITS_ARE_ENRICHED_BUT_NOT_LOW_DIMENSIONAL_S2060',
       'method': 'S2060 rank-1 DAS re-run at three seeds per circuit; spread reported for the three '
                 'quantities S2060 published',
       'overlap_with_closed_form': OV, 'das_concentration': CN, 'fraction_of_full_recovered': FR,
       'circuits_with_overlap_sd_at_least_0.05': noisy,
       'circuits_with_mean_overlap_at_least_0.5': over_half,
       'unhealthy_fits': unhealthy,
       'pred_a_overlaps_carry_seed_noise': bool(len(noisy) >= len(OV) / 2),
       'pred_b_S2060_conclusion_survives': bool(len(over_half) == 0),
       'pred_c_optimiser_healthy': bool(not unhealthy),
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/DAS_SEED_STABILITY.json', 'w'), indent=1)

print(f'\nwrote circuits/DAS_SEED_STABILITY.json ({time.time()-t0:.0f}s)')
if unhealthy:
    print(f'  UNHEALTHY FITS {unhealthy} -- reporting nothing (LESSON 108)')
else:
    print(f'pred_a  overlap sd >=0.05 for {len(noisy)}/{len(OV)} circuits (bar >= half) : '
          f'{rep["pred_a_overlaps_carry_seed_noise"]}   {noisy}')
    print(f'pred_b  mean overlap stays <0.50 for every circuit (bar none over) : '
          f'{rep["pred_b_S2060_conclusion_survives"]}   over: {over_half}')
    print(f'pred_c  every fit healthy : {rep["pred_c_optimiser_healthy"]}')
    print('\n  overlap with closed form (S2060 single-seed -> mean +- sd over 3 seeds):')
    S2060 = json.load(open('circuits/DAS.json'))['by_tag']
    for t, v in OV.items():
        was = S2060.get(t, {}).get('ranks', {}).get('1', {}).get('overlap_with_closed_form')
        print(f'    {t:12s} S2060 {was}  ->  {v["mean"]:.4f} +- {v["sd"]:.4f}   {v["values"]}')
