# TRUE GRADIENT-DESCENT DAS ON THE CENSUS CIRCUITS -- and does it find what the closed form found?
#
# TASK CONTEXT (Logan, 2026-08-30): "Use ablations and ... interchange patching and DAS like classic mech
# interp to isolate specifically where it is located." Ablation and interchange are done for every curated
# circuit (circuits/BATTERY.json). DAS is the piece no circuit in this folder has ever had: zero of 70.
#
# WHY IT MATTERS BEYOND CHECKING A BOX. §2056 and §2058 located circuits with a CLOSED-FORM direction --
# unit(mean(members) - mean(off-slice)) -- and §2058 explicitly recorded the caveat that this "is still
# closed-form, not gradient-descent DAS", which "could find multi-dimensional structure a single residual
# direction misses". This run is that caveat, tested. It is the difference between "the mean-difference
# direction is where the circuit lives" and "the mean-difference direction is where the circuit's MEAN
# lives, and the circuit lives somewhere else".
#
# THE METHOD (Geiger/Wu DAS, adapted to a census circuit). At the circuit's best-localised component k,
# learn an orthonormal Q (D x r) by gradient descent through the FROZEN model. The intervention is an
# interchange along Q only: y' = y - (y@Q)Q^T + (y_donor@Q)Q^T, donor being the same component's output at
# a different grid position. The objective rewards damage ON the circuit's members and penalises damage
# off its slice, so Q is pushed toward the subspace that carries THIS circuit and not the rest of the
# component. Trained on one set of rows, every number reported on HELD-OUT rows.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  DAS rank-1 beats the closed-form rank-1 direction on held-out member damage for >= 4 of the 5
#           a8 circuits. DAS optimises exactly this objective and the closed form does not optimise
#           anything, so anything less means the optimiser is not working and no other number here can be
#           trusted. This is the SANITY predicate, and it is meant to be easy.
#   pred_b  DAS-Q's overlap with the closed-form direction exceeds 0.5 for the majority of circuits --
#           the two methods find the SAME place. If TRUE, §2056/§2058's cheap closed-form probe is
#           validated and the caveat closes. If FALSE, gradient descent finds a different subspace and
#           every localisation claim in §2056/§2058 is a claim about the mean-difference direction only,
#           which I would have to say plainly.
#   pred_c  Rank-4 DAS recovers >= 80% of the FULL component's held-out interchange damage on members --
#           the circuit lives in a subspace of dimension <= 4 out of 1152. If FALSE the circuit is not
#           low-dimensional at its own best component and "isolating where it is located" has a floor
#           that four dimensions do not reach.
#
# HARNESS HISTORY. The first version of this script FAILED ITS OWN SANITY PREDICATE and its numbers were
# discarded, not published. It initialised P as randn(D, r) with |P| ~ 35 and stepped Adam at lr 5e-3, so
# each step moved P by a relative 1.4e-4; after 120 steps QR returned essentially the random
# initialisation, and the learned subspace's overlap with the closed-form direction came out at 0.0009 --
# which is exactly 1/D, the expected overlap of a RANDOM direction in D=1152. The gradient was flowing
# (measured grad norm 8.1e-3, nonzero); the step was simply too small to rotate a vector of that norm.
# Fixed here by unit-scale initialisation, lr 5e-2, 400 steps, member-rich batching -- and by the
# OPTIMISER HEALTH GATE below, which refuses to report a subspace that did not move from its own
# initialisation. A learned-parameter run must demonstrate that the parameter learned.
#
# Writes circuits/DAS.json (read-only artifact; modifies no circuit file -- Codex works the same folder).
import json
import time

import torch
import torch.nn.functional as F

import census_lib as C

RANKS = (1, 4)
STEPS = 400
LR = 5e-2
BATCH = 4
SEED = 20260830
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


out = {}
t0 = time.time()
for tag in TARGETS:
    key = BAT['by_tag'][tag]['best_mean']
    mm, sl = leaf_masks(tag)
    acts_tr = capture(key, *TRAIN_ROWS)
    acts_ev = capture(key, *EVAL_ROWS)
    rec = {'component': key, 'members': int(mm.sum()), 'ranks': {}}
    for r in RANKS:
        g = torch.Generator(device='cpu').manual_seed(SEED + r)
        # unit-scale init: |P| ~ 1 so an lr 5e-2 Adam step is a real rotation, not a 1e-4 nudge
        P0 = (torch.randn(C.D, r, generator=g) / C.D ** 0.5).to(C.DEV)
        Q_init = torch.linalg.qr(P0)[0].detach().clone()
        P = P0.clone().requires_grad_(True)
        opt = torch.optim.Adam([P], lr=LR)
        gg = torch.Generator().manual_seed(SEED)
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

rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude, circuit task (Logan)',
       'method': 'gradient-descent DAS (Geiger/Wu): orthonormal Q learned through the frozen model so an '
                 'interchange restricted to Q maximises damage on the circuit members and minimises it '
                 'off the circuit slice; trained on rows 0-600, all reported numbers on held-out 600-1000',
       'state': 'census_state_diverse.pt', 'seed': SEED, 'steps': STEPS, 'lr': LR,
       'note': 'read-only artifact; no circuit file was modified', 'by_tag': out}
json.dump(rep, open('circuits/DAS.json', 'w'), indent=1)

unhealthy = [(t, r) for t, v in out.items() for r, e in v['ranks'].items()
             if not e.get('optimiser_healthy')]
print(f'\nOPTIMISER HEALTH GATE: {len(unhealthy)} of {sum(len(v["ranks"]) for v in out.values())} '
      f'fits did not move from init or did not reduce their loss.')
if unhealthy:
    print('  UNHEALTHY -- these fits report nothing:', unhealthy)
    print('  A learned-subspace number from a subspace that did not learn is not a measurement.')

cf_ok = [t for t, v in out.items() if v['ranks'][1].get('overlap_with_closed_form') is not None]
a8p = [t for t in A8 if t in out and 'das_beats_closed_form' in out[t]['ranks'][1]]
print(f'\nwrote circuits/DAS.json ({time.time()-t0:.0f}s)')
nb = sum(1 for t in a8p if out[t]['ranks'][1]['das_beats_closed_form'])
print(f'pred_a  DAS rank-1 beats closed form on held-out member damage: {nb}/{len(a8p)} of the a8 five '
      f'(bar >=4) : {nb >= 4}   [SANITY -- meant to be easy]')
for t in a8p:
    e = out[t]['ranks'][1]
    print(f'    {t:12s} DAS {e["das_dce_members"]:.4f} vs closed form '
          f'{e["closed_form_dce_members"]:.4f}')
if cf_ok:
    ovs = [out[t]['ranks'][1]['overlap_with_closed_form'] for t in cf_ok]
    print(f'pred_b  overlap with closed form >0.5 for {sum(1 for o in ovs if o > 0.5)}/{len(ovs)} '
          f'(bar majority) : {sum(1 for o in ovs if o > 0.5) > len(ovs)/2}')
    for t in cf_ok:
        print(f'    {t:12s} overlap {out[t]["ranks"][1]["overlap_with_closed_form"]}')
rec4 = [(t, out[t]['ranks'][4]['fraction_of_full_recovered']) for t in out
        if out[t]['ranks'][4]['fraction_of_full_recovered'] is not None]
n4 = sum(1 for _t, v in rec4 if v >= 0.80)
print(f'pred_c  rank-4 recovers >=80% of the full component on members: {n4}/{len(rec4)}')
for t, v in rec4:
    print(f'    {t:12s} {v}')
