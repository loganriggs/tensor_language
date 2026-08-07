"""E25 LEARNED BROADCAST GATES on the commons (fresh single-epoch batch-16,
recipe conventions; results -> qk_e25.json).

MOTIVATION: the scale box's S2 soft write-lasso (MAILBOX 'S2 soft
write-lasso') recovered the ENTIRE partition cost (-0.218 vs the recipe)
but collapsed wiring Spearman to 0.31, and its learned permission matrix
was BIMODAL: ~16 modules broadcasting everywhere, 7 slot-confined, top
broadcasters = the commons ledger's top writers (mlp11, mlp0, attn9,
mlp10). The mailbox's own architecture-idea: 'designated broadcast
modules' as first-class structure -- halfway between the commons (one
shared subspace, E14c: -0.156 vs E9a at Spearman 0.69) and S2
(free-for-all: -0.218 at 0.31). E25 tests exactly that with a LEARNED,
PRICED permission bit per module.

BASE: the E14c commons layout verbatim (readable recipe at 24 x 9-dim
slots = 216 + one shared 48-dim commons all modules may write; per-slot
RMSNorm over the 25 segments, Muon 0.02 / embedding AdamW, in-loss
group-lasso 3e-5 with the commons as the 25th read column group --
qk_e14_slotcap_run.VarSlotRoute reused as the superclass).

CHANGE: each module m's COMMONS write is multiplied by a hard-concrete
(L0) learnable binary gate g_m in [0,1] (Louizos-style stretched concrete:
temperature 2/3, stretch [-0.1, 1.1]; stochastic during training,
deterministic clamp(sigmoid(log_alpha)*(zeta-gamma)+gamma, 0, 1) at eval).
Slot writes are UNGATED. The loss gains lambda_gate * sum_m E[gate_m open]
with E[open] = sigmoid(log_alpha - beta*log(-gamma/zeta)) -- a price per
PERMISSION (per module allowed to broadcast), not per magnitude.

LAMBDA CALCULATION (documented, pre-registered): the measured commons gain
is E14c minus E9a = -0.1557 nats (qk_e14.json E14c_minus_e9a_fresh). If
~8 open gates suffice to keep the full gain, the spend at break-even is
8 * lambda_gate = 0.1557 => lambda_gate = 0.1557 / 8 = 0.019463 nats per
open gate. Cheaper configurations (fewer permissions at most of the gain)
are profitable; a diffuse commons (all 24 needed) is priced out.
ANNEAL-IN (standing decision, BRAINSTORM_STATE queue item 6: strong-early
sparsity kills channels -- the zero-init dead-gradient trap): lambda_gate
ramps linearly 0 -> full over training steps 0..2000, constant after.
Gate parameters (1-D log_alpha, init +2.5 => start effectively open ~0.99)
sit in the AdamW no-decay group (Muon is matrices-only, harness rule).

CONTROLS BEFORE TRAINING (hard gates):
  1. ALL-GATES-HARD-OPEN reproduces the E14c commons forward EXACTLY at
     matched init (parameter identity + bit-exact logits: the gated mask
     slot_mask + 1*commons_mask equals E14c's write mask elementwise);
  2. ALL-GATES-HARD-CLOSED reproduces slots-only exactly (an E14c twin
     whose write masks have the commons columns zeroed);
  3. EXPECTED-L0 penalty: the vectorized sum matches a naive per-gate
     loop of the Louizos formula (rel < 1e-6), and the eval gate value
     matches the per-gate clamp formula.

REGISTERED PREDICTIONS (merged before training):
  (i)   gates POLARIZE: final expected-open values bimodal (<= 4 of 24 in
        the [0.05, 0.95] middle band), with <= 10 effectively open
        (eval gate value > 0.5);
  (ii)  the open set overlaps the S2 write-lasso broadcast cast at >= 3 of
        the top-4 gates in {mlp11, mlp0, attn9, mlp10};
  (iii) CE recovery in [-0.20, -0.05] paired vs the E9a slots-base with
        wiring readability (var-slot Spearman all) >= 0.70 -- between the
        write-lasso's 0.31 and the typed-quarters 0.82.

MEASURE: paired fresh CE vs E0a/E0b (harness) + E9a slots-base + E14c
ungated commons; oldheld record; per-seq heldloss; var-slot light probe +
commons ledger (gated write norms); gate report (eval values, expected
open, trajectory via traj_extra into qk_e25_a_traj.npz every 200 steps);
verdicts vs the three predictions. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, C, DEPTH, F, nn, torch
import qk_e7_evenout_run as E7R
import qk_e14_slotcap_run as E14R

JP = E.jpath('qk_e25.json')
GC25 = 3e-5                                # the E14c lasso (unchanged)
BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1
COMMONS_GAIN = 0.1557                      # E14c minus E9a (qk_e14.json)
N_BREAK_EVEN = 8
LAM_GATE = COMMONS_GAIN / N_BREAK_EVEN     # 0.019463 nats per open gate
LAM_RAMP_STEPS = 100 if E.SMOKE else 2000
GATE_INIT = 2.5                            # log_alpha init: ~open at start
S2_CAST = ('mlp11', 'mlp0', 'attn9', 'mlp10')


def e25_layout():
    if E.SMOKE:
        return [1] * 20 + [0] * 4, 4       # the E14c smoke layout, verbatim
    return [9] * 24, 48


class E25Gated(E14R.VarSlotRoute):
    """VarSlotRoute (E14c layout) + per-module hard-concrete gates on the
    COMMONS portion of each write. gate_mode: 'learned' (stochastic in
    training, deterministic at eval), 'open' (hard 1), 'closed' (hard 0).
    The gate parameter is created AFTER the superclass init with a
    constant (RNG-free) initializer, so every shared parameter consumes
    the identical global RNG stream as make_e14c (asserted by control 1)."""

    def __init__(self, variant, depth, sizes, commons):
        super().__init__(variant, depth, sizes, commons=commons)
        Dm = self.wte.weight.shape[1]
        wm_slot = self.wmask.clone()
        wm_slot[:, Dm - commons:] = 0.0
        self.register_buffer('wmask_slot', wm_slot)
        cm = torch.zeros(Dm)
        cm[Dm - commons:] = 1.0
        self.register_buffer('cmask', cm)
        self.gate_log_alpha = nn.Parameter(
            torch.full((2 * depth,), GATE_INIT))
        self.gate_mode = 'learned'
        self.register_buffer('gate_step', torch.zeros((), dtype=torch.long))
        self._aux = None

    # ---- hard-concrete machinery ----
    def gate_values(self):
        la = self.gate_log_alpha
        if self.gate_mode == 'open':
            return torch.ones_like(la)
        if self.gate_mode == 'closed':
            return torch.zeros_like(la)
        if self.training and torch.is_grad_enabled():
            u = torch.rand_like(la).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((u.log() - (-u).log1p() + la) / BETA)
        else:
            s = torch.sigmoid(la)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)

    def expected_gate_open(self):
        return torch.sigmoid(self.gate_log_alpha
                             - BETA * math.log(-GAMMA / ZETA))

    def gate_lambda_now(self):
        return LAM_GATE * min(1.0, float(self.gate_step) / LAM_RAMP_STEPS)

    def pop_aux_loss(self):
        a = self._aux
        self._aux = None
        return a

    def traj_extra(self):
        with torch.no_grad():
            ev = torch.sigmoid(self.gate_log_alpha.detach().float())
            zdet = ((ev * (ZETA - GAMMA)) + GAMMA).clamp(0, 1)
        return {'gate_expected_open':
                self.expected_gate_open().detach().float().cpu().numpy(),
                'gate_eval_value': zdet.cpu().numpy()}

    # ---- forward: E1Route.forward with the commons-gated write masks ----
    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        if self.training and torch.is_grad_enabled():
            self.gate_step += 1
        z = self.gate_values()
        e = F.rms_norm(self.wte(idx), (Dm,))
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        def wmask_gated(k):
            return (self.wmask_slot[k] + z[k] * self.cmask)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)

            def qk(lin):
                zq = lin(hn).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(zq, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * wmask_gated(2 * l).to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = self.slot_norm(x)
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * wmask_gated(2 * l + 1).to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        if self.training and torch.is_grad_enabled() \
                and self.gate_mode == 'learned':
            self._aux = self.gate_lambda_now() \
                * self.expected_gate_open().sum()
        return 30 * torch.tanh(logits / 30)


def make_e25():
    sizes, commons = e25_layout()
    C.register('E25a')
    torch.manual_seed(Q.SEED)
    return E25Gated('E25a', DEPTH, sizes, commons).to(E.DEV)


def sname(k):
    return ('attn' if k % 2 == 0 else 'mlp') + str(k // 2)


# ---------------- controls ----------------
@torch.no_grad()
def controls():
    j = E.loadj(JP)
    if j.get('controls_E25', {}).get('pass'):
        print("controls_E25: already passed -- skip", flush=True)
        return
    idx = Q.HELD[:2, :Q.T]
    # 1: hard-open == E14c commons forward at matched init
    m14 = E14R.make_e14c().eval().float()
    m25 = make_e25().eval().float()
    p14 = dict(m14.named_parameters())
    pdiff = max(float((p - p14[nm]).abs().max())
                for nm, p in m25.named_parameters()
                if nm in p14)
    extra = [nm for nm, _ in m25.named_parameters() if nm not in p14]
    m25.gate_mode = 'open'
    d_open = float((m25(idx) - m14(idx)).abs().max())
    # 2: hard-closed == slots-only (E14c twin with commons write cols zeroed)
    cdim = m14.commons
    Dm = m14.wte.weight.shape[1]
    m14.wmask[:, Dm - cdim:] = 0.0
    m25.gate_mode = 'closed'
    d_closed = float((m25(idx) - m14(idx)).abs().max())
    m25.gate_mode = 'learned'
    # 3: expected-L0 penalty vs a naive per-gate loop + eval-value formula
    fast = float(m25.expected_gate_open().sum())
    naive = 0.0
    for la in m25.gate_log_alpha.detach().tolist():
        naive += 1.0 / (1.0 + math.exp(-(la - BETA * math.log(-GAMMA / ZETA))))
    rel = abs(fast - naive) / max(abs(naive), 1e-12)
    zv = m25.gate_values()                     # eval mode (no_grad)
    z_naive = [min(1.0, max(0.0, (1.0 / (1.0 + math.exp(-la)))
                            * (ZETA - GAMMA) + GAMMA))
               for la in m25.gate_log_alpha.detach().tolist()]
    z_diff = max(abs(float(a) - b) for a, b in zip(zv, z_naive))
    rec = {'param_identity_max_abs_diff_shared': pdiff,
           'params_only_in_E25': extra,
           'forward_hard_open_vs_e14c_max_logit_diff': d_open,
           'forward_hard_closed_vs_slotsonly_max_logit_diff': d_closed,
           'expected_l0_fast_vs_naive_rel': rel,
           'eval_gate_value_vs_formula_max_diff': z_diff,
           'pass': bool(pdiff == 0.0 and d_open == 0.0 and d_closed == 0.0
                        and rel < 1e-6 and z_diff < 1e-6
                        and extra == ['gate_log_alpha'])}
    E.merge(JP, 'controls_E25', rec)
    print(f"controls_E25: params {pdiff:.1e}, open {d_open:.1e}, closed "
          f"{d_closed:.1e}, penalty rel {rel:.1e}, eval-z {z_diff:.1e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], 'controls_E25 FAILED'
    del m14, m25
    torch.cuda.empty_cache()


# ---------------- gate report + verdicts ----------------
def gate_report():
    key = 'gate_report_E25'
    if E.SMOKE or key in E.loadj(JP):
        return
    stem = 'qk_e25_a'
    if not os.path.exists(E.ckpath(stem)):
        return
    m, _ = E.load_arm(stem, make_e25)
    with torch.no_grad():
        eo = m.expected_gate_open().float().cpu().numpy()
        zdet = ((torch.sigmoid(m.gate_log_alpha.detach().float())
                 * (ZETA - GAMMA)) + GAMMA).clamp(0, 1).cpu().numpy()
    order = np.argsort(-zdet)
    open_set = [sname(k) for k in range(len(zdet)) if zdet[k] > 0.5]
    top4 = [sname(int(k)) for k in order[:4]]
    overlap = len(set(top4) & set(S2_CAST))
    middle = int(((eo > 0.05) & (eo < 0.95)).sum())
    rec = {'gate_eval_values': {sname(k): round(float(zdet[k]), 4)
                                for k in range(len(zdet))},
           'expected_open': {sname(k): round(float(eo[k]), 4)
                             for k in range(len(eo))},
           'n_effectively_open_evalz_gt0.5': len(open_set),
           'open_set': open_set,
           'top4_by_eval_value': top4,
           's2_broadcast_cast': list(S2_CAST),
           'top4_overlap_with_s2_cast': overlap,
           'n_expected_open_in_middle_band_0.05_0.95': middle,
           'lambda_gate': round(LAM_GATE, 6),
           'lambda_calc': f'E14c commons gain {COMMONS_GAIN} nats / '
                          f'{N_BREAK_EVEN} break-even permissions',
           'ramp': f'lambda 0 -> full over steps 0..{LAM_RAMP_STEPS}'}
    E.merge(JP, key, rec)
    print(f"gates: {len(open_set)} open ({', '.join(open_set) or 'none'}); "
          f"top4 {top4} overlap with S2 cast {overlap}/4; middle band "
          f"{middle}", flush=True)
    del m
    torch.cuda.empty_cache()


def summarize():
    j = E.loadj(JP)
    if 'E25a' not in j:
        return
    summary = {'bars': {
        'E14c_commons_vs_e9a': -0.1557, 'S2_write_lasso_vs_recipe': -0.218,
        'S2_spearman': 0.31, 'E14c_spearman': 0.6875,
        'typed_quarters_spearman': 0.82}}
    row = {'final_held_ce_fresh_bf16': j['E25a']['final_held_ce_fresh_bf16'],
           'diverged': j['E25a']['diverged']}
    d9 = j.get('E25a_minus_e9a_fresh', {}).get('minus_e9a')
    d14 = j.get('E25a_minus_e14c_fresh', {}).get('minus_e14c')
    if d9 is not None:
        row['minus_e9a_paired'] = round(d9, 4)
    if d14 is not None:
        row['minus_e14c_commons_paired'] = round(d14, 4)
    lp = j.get('light_probe_E25a', {})
    sp = lp.get('wiring_spearman_all')
    if sp is not None:
        row['wiring_spearman_all'] = sp
    gr = j.get('gate_report_E25', {})
    summary['E25a'] = row
    if gr:
        n_open = gr['n_effectively_open_evalz_gt0.5']
        middle = gr['n_expected_open_in_middle_band_0.05_0.95']
        summary['prediction_i_verdict'] = (
            f"{'CONFIRMED' if n_open <= 10 and middle <= 4 else 'REFUTED'}"
            f": {n_open} gates effectively open (registered <= 10), "
            f"{middle} in the 0.05-0.95 middle band (registered <= 4)")
        ov = gr['top4_overlap_with_s2_cast']
        summary['prediction_ii_verdict'] = (
            f"{'CONFIRMED' if ov >= 3 else 'REFUTED'}: top-4 gates "
            f"{gr['top4_by_eval_value']} overlap the S2 broadcast cast "
            f"{list(S2_CAST)} at {ov}/4 (registered >= 3)")
    if d9 is not None and sp is not None:
        ok = (-0.20 <= d9 <= -0.05) and sp >= 0.70
        summary['prediction_iii_verdict'] = (
            f"{'CONFIRMED' if ok else 'REFUTED'}: CE vs E9a {round(d9, 4)} "
            f"(registered [-0.20, -0.05]) at Spearman {sp} (registered "
            f">= 0.70)")
    E.merge(JP, 'summary_E25', summary)
    print(json.dumps({'summary_E25': summary}, indent=2), flush=True)


def save_seq_heldloss(stem):
    if E.SMOKE:
        return
    p = f'{E.QK}/{stem}_heldloss.npy'
    q = f'{E.QK}/{stem}_heldloss_seq.npy'
    if os.path.exists(p) and not os.path.exists(q):
        pt = np.load(p)
        np.save(q, pt.reshape(len(Q.HELD), Q.T).mean(1))
        print(f"saved {stem}_heldloss_seq.npy", flush=True)


if __name__ == '__main__':
    E.setup()
    controls()

    if 'E25_prediction' not in E.loadj(JP):
        E.merge(JP, 'E25_prediction', {
            'registered_before_training': True,
            'design': 'E14c commons layout (24x9 slots + 48-dim commons, '
                      'lasso 3e-5, readable recipe) + hard-concrete gate '
                      'per module on its COMMONS write (temperature 2/3, '
                      'stretch [-0.1, 1.1]); slot writes ungated; penalty '
                      'lambda_gate * sum(expected open) -- price per '
                      'permission, not magnitude; gates on AdamW no-decay '
                      '(1-D params, Muon is matrices-only)',
            'lambda_gate': round(LAM_GATE, 6),
            'lambda_calc': f'measured commons gain = E14c minus E9a = '
                           f'-{COMMONS_GAIN} nats (qk_e14.json); break-even '
                           f'at {N_BREAK_EVEN} open gates: lambda = '
                           f'{COMMONS_GAIN}/{N_BREAK_EVEN} = '
                           f'{COMMONS_GAIN / N_BREAK_EVEN:.6f}',
            'anneal': f'lambda ramps 0 -> full over steps 0..'
                      f'{LAM_RAMP_STEPS} (standing decision: strong-early '
                      'sparsity kills channels)',
            'i_polarization': 'final expected-open bimodal: <= 4 of 24 in '
                              '[0.05, 0.95], <= 10 effectively open '
                              '(eval value > 0.5)',
            'ii_open_set': 'top-4 gates overlap the S2 write-lasso '
                           f'broadcast cast {list(S2_CAST)} at >= 3 of 4',
            'iii_ce_and_readability': 'paired CE vs E9a in [-0.20, -0.05] '
                                      'with var-slot wiring Spearman >= '
                                      '0.70 (between write-lasso 0.31 and '
                                      'typed 0.82)'})

    E14R.train_var_arm('qk_e25_a', 'E25a', make_e25,
                       'E14c commons + per-module hard-concrete broadcast '
                       f'gates, lambda_gate {LAM_GATE:.5f} '
                       f'(= {COMMONS_GAIN}/{N_BREAK_EVEN}), ramp '
                       f'{LAM_RAMP_STEPS} steps, gates on AdamW')
    E14R.pair_extra('qk_e25_a', 'E25a', (('qk_e14_c', 'e14c'),))
    save_seq_heldloss('qk_e25_a')

    if not E.SMOKE and os.path.exists(E.ckpath('qk_e25_a')) \
            and 'E25a_commons_ledger' not in E.loadj(JP):
        m, _ = E.load_arm('qk_e25_a', make_e25)
        E.merge(JP, 'E25a_commons_ledger', E14R.commons_ledger(m))
        del m
        torch.cuda.empty_cache()
    gate_report()
    summarize()
    print('e25 gates run done', flush=True)
