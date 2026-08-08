"""Codebook slots at w1152: does discrete content survive 65-dim slots?

At w264 (qk_e20.json) vector-quantizing the frontier arm's inter-module
messages was VIABLE: 256 codes x 2-step matching pursuit per slot cost
+0.1344 CE over its continuous parent (inside the registered <= +0.15
PROMISING band), produced ZERO dead codes, and -- the surprise -- IMPROVED
wiring readability, plain Spearman 0.8936 against the parent's 0.7911 with
readout-interface top-10 precision 0.8. Content came to 342 bits/token.

The transfer question is whether both halves survive the width jump, which
at w264 was asked of 15-dim slots and here is asked of 65-dim slots. There
is a concrete reason to doubt it: a fixed 256-code budget has to cover a
slot whose ambient dimension is 4.3x larger, so 2-code pursuit spans a much
smaller fraction of the content space. The registered predictions below make
that quantitative rather than rhetorical.

PARENT (argv[1], default bw3e5): the branch-point arm whose architecture and
lasso coefficient this inherits verbatim -- bw1e4 (lasso 1e-4) or bw3e5
(lasso 3e-5). The codebook arm is its parent plus quantization and nothing
else, so the CE delta is attributable to quantization alone.

WHAT IS QUANTIZED (unchanged from E20, including its exemptions): every
module-written slot's post-per-slot-RMSNorm content, at every block-level
read, is replaced by a 2-code matching-pursuit message from that slot's own
256-code unit-norm codebook (scales are the inner products and stay
continuous), straight-through on the backward, EMA 0.99 + commitment 0.25 +
dead-code reinit. Documented exemptions: slots not yet written (pure
bottom-injected embedding) and the readout, which reads at the GLOBAL norm,
so slot 23 (mlp11's write) never passes a codebook.

ACCUMULATION SEMANTICS (the one real adaptation, and it matters). At w264
the protocol was batch 16 with a single forward per optimizer step, so the
codebook's internal step counter advanced once per optimizer step and
QZ_DEAD = 200 meant "unused for 200 optimizer steps". Here the effective
batch of 32 is reached by micro-batch accumulation (4 chunks of 8), so the
counter advances 4x per optimizer step. QZ_DEAD is therefore scaled by the
accumulation factor to preserve the ORIGINAL semantics; the EMA decay is
left at 0.99 per update, which does track ~4x faster in optimizer-step
terms, recorded here as a known deviation. It is not load-bearing for the
headline: E20's dead-code count was zero, i.e. the threshold never fired.

GATES (all three ported, all before training):
  1. BIT-EXACT BYPASS: with quantization off the model must be parameter-
     identical to the parent bandwidth arm, reproduce its forward exactly,
     and reproduce 3 steps of its training exactly through the SCALE
     trainer -- which also proves the new pop_aux_loss hook is inert.
  2. EXACT CAPACITY RECOVERY: with n >= the number of distinct vectors and
     k = s pursuit steps, quantization error must vanish.
  3. PLANTED-TOY EMA RECOVERY: the EMA + dead-reinit machinery, driven
     through the exact functions the model calls, must recover 10 planted
     cluster centers.

Outputs qk_s_w1152_cb_<parent>.{json,pt} + heldloss/f34kloss npys, codebook
snapshots, and a dead-code event log. Idempotent on the JSON keys.
"""
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import functools
import inspect
import json
import math

import numpy as np

import qk_tokenline_train as Q

Q.gpu_guard = lambda *a, **k: None

import qk_s_gate_run as G

G.MEM_BUDGET_MIB = 24000       # quantization adds fp32 pursuit temporaries

import qk_w1152_train as W2
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_e_common as E
from qk_e_common import DEPTH, torch
import qk_e15_reinvest_run as E15R
import qk_e20_codebook_run as E20R          # E20Route + mp_quantize + ema_update
import qk_s_bw1152_run as BW                # the parent architecture + controls
import qk_s_muon_run as M

PARENT = sys.argv[1] if len(sys.argv) > 1 else 'bw3e5'
PARENT_COEFF = {'bw1e4': 1e-4, 'bw3e5': 3e-5}[PARENT]
STEM = f'qk_s_w1152_cb_{PARENT}'
JP = os.path.join(G.OUT_DIR, f'{STEM}.json')
QZ_N, QZ_K = E20R.QZ_N, E20R.QZ_K
SNAPSHOTS = {}
_EMA_UPDATE_ORIG = E20R.ema_update           # captured before any rebinding


def slot_dim():
    """The parent's solved slot dim, read from its controls block."""
    pj = os.path.join(G.OUT_DIR, f'qk_s_w1152_{PARENT}.json')
    s = G.loadj(pj).get('controls', {}).get('slot')
    assert s, f'parent {PARENT} has no solved slot dim yet ({pj})'
    return s


def make_cb(s=None, qz_on=True, std=None):
    """E20Route at the scale width with the parent's decoder init.

    Mirrors BW.make_bw exactly (same global-RNG discipline, same generator-888
    decoder re-draw at the SCALE write-init std); E20Route draws its codebook
    from its own fixed generator, so all shared parameters stay bit-identical
    to the parent -- which gate 1 asserts rather than assumes."""
    if s is None:
        s = SLOT
    if std is None:
        std = BW.write_std()
    hidden = 4 * Q.D
    torch.manual_seed(Q.SEED)
    m = E20R.E20Route(f'CB1152_{PARENT}', DEPTH, s, Q.D, Q.NH, Q.HD, hidden,
                      n_codes=QZ_N).to(E.DEV)
    gw = torch.Generator().manual_seed(888)
    with torch.no_grad():
        for blk in m.h:
            for p in (blk.c_proj.weight, blk.Down.weight):
                p.copy_(torch.randn(p.shape, generator=gw) * std)
    m.qz_on = qz_on
    return m


def step_cb(step, model):
    """Read-only logging: codebook snapshots + dead-code event flushes."""
    if not getattr(model, 'qz_on', False):
        return
    ev = getattr(model, 'qz_dead_events', None)
    if ev:
        with open(os.path.join(G.OUT_DIR, f'{STEM}_deadcode_events.jsonl'),
                  'a') as f:
            for e_ in ev:
                f.write(json.dumps(e_) + '\n')
        model.qz_dead_events = []
    if step % 1000 == 0:
        SNAPSHOTS[f'step{step:05d}'] = model.qz_codebook.detach().cpu().numpy()
        np.savez(os.path.join(G.OUT_DIR, f'{STEM}_codebook_snapshots.npz'),
                 **SNAPSHOTS)


step_cb.every = 250


def three_step_scale(fac, micro, lr_adamw, steps=3):
    """3 steps through the SCALE trainer with a given factory."""
    prev = M.factory
    M.factory = fac
    try:
        log = M.train_muon_run(0.02, lr_adamw, steps, micro, save_stem=None,
                               log_every=1)
    finally:
        M.factory = prev
    return {'per_step_ce': [x[1] for x in log['train_loss']],
            'held100_ce': log['final_held_ce']}


# ---------------- gates ----------------
def gate_bypass(micro, lr_adamw):
    key = 'gate1_bypass'
    out = G.loadj(JP)
    if out.get(key, {}).get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    mp_ = BW.make_bw(s=SLOT, std=BW.write_std()).eval().float()
    mq = make_cb(s=SLOT, qz_on=False).eval().float()
    pp = dict(mp_.named_parameters())
    pdiff = max(float((p - pp[nm]).abs().max())
                for nm, p in mq.named_parameters())
    idx = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        d_fwd = float((mq(idx) - mp_(idx)).abs().max())
        mq.qz_on = True
        d_qz = float((mq(idx) - mp_(idx)).abs().max())
        mq.qz_on = False
    del mp_, mq
    torch.cuda.empty_cache()
    parent = three_step_scale(lambda: BW.make_bw(s=SLOT, std=BW.write_std()),
                              micro, lr_adamw)
    mine = three_step_scale(lambda: make_cb(s=SLOT, qz_on=False),
                            micro, lr_adamw)
    sd = max(abs(a - b) for a, b in zip(parent['per_step_ce'],
                                        mine['per_step_ce']))
    hd = abs(parent['held100_ce'] - mine['held100_ce'])
    rec = {'param_identity_max_abs_diff': pdiff,
           'forward_bypass_max_logit_diff': d_fwd,
           'forward_quantized_at_init_logit_shift_informational': d_qz,
           'train3_parent_per_step_ce': parent['per_step_ce'],
           'train3_bypass_per_step_ce': mine['per_step_ce'],
           'train3_max_per_step_abs_diff': sd,
           'train3_held100_abs_diff': hd,
           'note': '3-step run goes through the SCALE trainer, so it also '
                   'proves the new pop_aux_loss + step_cb hooks in '
                   'qk_s_muon_run are inert with quantization off',
           'pass': bool(pdiff == 0.0 and d_fwd == 0.0 and sd == 0.0
                        and hd < 1e-6)}
    G.savej(JP, {**G.loadj(JP), key: rec})
    print(f"{key}: params {pdiff:.1e}, forward {d_fwd:.1e}, 3-step "
          f"{sd:.1e}/{hd:.1e} (quantized-at-init logit shift {d_qz:.3f}) -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def gate_capacity():
    key = 'gate2_capacity'
    out = G.loadj(JP)
    if out.get(key, {}).get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    m = make_cb(qz_on=True).eval().float()
    b = E.OLD_HELD[:2, :Q.T]
    with torch.no_grad():
        e = torch.nn.functional.rms_norm(m.wte(b), (m.Ws,))
        v = torch.nn.functional.rms_norm(
            e[..., :m.s], (m.s,)).reshape(-1, m.s).float()[:128]
        uniq = torch.unique(v, dim=0)
        Cb = uniq / uniq.norm(dim=1, keepdim=True).clamp_min(1e-8)
        recon, _, _, _ = E20R.mp_quantize(v, Cb, m.s)
        rel = float((v - recon).norm() / v.norm().clamp_min(1e-12))
    del m
    torch.cuda.empty_cache()
    rec = {'n_vectors': int(v.shape[0]), 'n_distinct': int(uniq.shape[0]),
           'codebook_n': int(Cb.shape[0]), 'k_steps': SLOT,
           'rel_error': rel, 'pass': bool(rel < 1e-4)}
    G.savej(JP, {**G.loadj(JP), key: rec})
    print(f"{key}: {rec['n_distinct']} distinct vectors, k={SLOT}, rel error "
          f"{rel:.2e} -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def gate_toy():
    """Planted-toy EMA recovery through the exact ema_update the model uses."""
    key = 'gate3_toy_ema'
    out = G.loadj(JP)
    if out.get(key, {}).get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    g = torch.Generator(device='cpu').manual_seed(7)
    d, ncent, n = SLOT, 10, 4096
    cent = torch.randn(ncent, d, generator=g)
    cent = cent / cent.norm(dim=1, keepdim=True)
    Cb = torch.randn(ncent, d, generator=g)
    Cb = (Cb / Cb.norm(dim=1, keepdim=True)).to(E.DEV)
    Mema = Cb.clone()
    last = torch.zeros(ncent, dtype=torch.long, device=E.DEV)
    usage = torch.zeros(ncent, dtype=torch.long, device=E.DEV)
    for it in range(400):
        lab = torch.randint(0, ncent, (n,), generator=g)
        x = (cent[lab] + 0.02 * torch.randn(n, d, generator=g)).to(E.DEV)
        _, idxs, alphas, resids = E20R.mp_quantize(x, Cb, 1)
        # the ORIGINAL threshold (200), not the accumulation-scaled one, so
        # this reproduces exactly the gate E20 passed at w264 and still
        # exercises the dead-code reinit path within 400 iterations
        _EMA_UPDATE_ORIG(Cb, Mema, last, usage, it, x, idxs, alphas, resids)
    cos = (Cb @ cent.to(E.DEV).t()).abs().max(1).values
    nrec = int((cos > 0.99).sum())
    rec = {'planted_centers': ncent, 'recovered_at_cos_gt_0.99': nrec,
           'min_cos': float(cos.min()), 'iters': 400, 'slot_dim': d,
           'pass': bool(nrec >= 9)}
    G.savej(JP, {**G.loadj(JP), key: rec})
    print(f"{key}: recovered {nrec}/{ncent} centers (min cos "
          f"{rec['min_cos']:.4f}) -> {'PASS' if rec['pass'] else 'FAIL'}",
          flush=True)
    assert rec['pass'], f'{key} FAILED'


def register_predictions(accum):
    out = G.loadj(JP)
    if 'registered_predictions' in out:
        return
    pj = G.loadj(os.path.join(G.OUT_DIR, f'qk_s_w1152_{PARENT}.json'))
    parent_ce = pj.get('run', {}).get('held_ce_scale_bf16')
    out['registered_predictions'] = {
        'registered': 'before training',
        'parent': PARENT, 'parent_scale_held_ce': parent_ce,
        'parent_group_coeff': PARENT_COEFF,
        'w264_basis': {'E20a_cost_vs_parent': 0.1344, 'dead_code_frac': 0.0,
                       'plain_spearman': 0.8936, 'parent_plain': 0.7911,
                       'content_bits_per_token': 342, 'slot_dim': 15},
        'a_cost': (
            'CE cost vs the parent <= +0.15 keeps discrete content in the '
            'PROMISING band at 65-dim slots and confirms transfer; +0.15 to '
            '+0.30 triggers the distillation control per the E20 decision '
            'tree; > +0.30 REFUTES n=256/k=2 at this slot width'),
        'a_rationale': (
            'the honest prior is that the cost GROWS with slot dim: 2 codes '
            'out of 256 span a far smaller fraction of a 65-dim slot than of '
            'a 15-dim one, so a cost above the w264 +0.1344 is expected and '
            'the interesting question is whether it stays under +0.30'),
        'b_dead_codes': 'dead-code fraction < 30% (was exactly 0 at w264)',
        'c_readability': (
            'the w264 surprise was that quantization IMPROVED plain wiring '
            'Spearman (0.8936 vs the parent 0.7911); if that mechanism is '
            'real -- quantization killing the low-variance content that the '
            'covariance correction existed to discount -- it should repeat '
            'here, so predict this arm >= its parent on plain Spearman'),
        'accumulation_deviation': {
            'micro_chunks_per_step': accum,
            'qz_dead_scaled_to': E20R.QZ_DEAD * accum,
            'note': 'QZ_DEAD scaled by the accumulation factor so "unused '
                    'for 200 optimizer steps" keeps its w264 meaning; EMA '
                    'decay left at 0.99 per update, which tracks ~accum x '
                    'faster in optimizer-step terms (recorded, not '
                    'corrected -- E20 saw zero dead codes, so the threshold '
                    'is not load-bearing)'}}
    G.savej(JP, out)
    print("registered predictions written before training", flush=True)


SLOT = None


def main():
    global SLOT
    W2.patch_width(G.WIDTH)
    SLOT = slot_dim()
    print(f"codebook arm on parent {PARENT}: slot {SLOT}, stream {24 * SLOT}, "
          f"n_codes {QZ_N}, k {QZ_K}, lasso {PARENT_COEFF}", flush=True)

    M.ARM = f'cb_{PARENT}'
    M.CFG = dict(stem=STEM, coeff=PARENT_COEFF, prox=None, sweep=False)
    M.COEFF = PARENT_COEFF
    M.PROX = None
    M.STEM = STEM
    M.JP = JP
    M.factory = lambda: make_cb(qz_on=True)

    total_steps, spec, f34k_held = G.setup_data()
    lr_adamw, lr_src = M.resolve_lr_adamw()

    out = G.loadj(JP)
    out['data'] = spec
    out['env'] = {'gpu': torch.cuda.get_device_name(0),
                  'torch': torch.__version__, 'cooc_substitute': True}
    G.savej(JP, out)

    # measured 2026-08-08: quantization on top of the 1560-wide stream OOMs
    # at micro 8, so extend the ladder downward rather than shrink the
    # effective batch (which would break comparability with the parent)
    M.MICRO_LADDER = [16, 8, 4, 2]
    micro = M.preflight(G.loadj(JP), lr_adamw)
    accum = G.EFF_BATCH // micro
    # Preserve the w264 meaning of QZ_DEAD ("unused for 200 OPTIMIZER steps")
    # under micro-accumulation. Rebinding the module global is NOT enough:
    # ema_update takes dead_t as a DEFAULT ARGUMENT, bound at def time, and
    # _qz_slot calls it without passing one. Rebind the name E20R._qz_slot
    # resolves at call time instead, so the new threshold actually takes.
    # (ema_update is wrapped by @torch.no_grad(), so its __defaults__ is None
    # and the real default lives on the wrapped function -- read it through
    # inspect, which follows __wrapped__. The partial still overrides it
    # correctly either way, because the keyword passes through the wrapper.)
    orig_dead = inspect.signature(_EMA_UPDATE_ORIG).parameters['dead_t'].default
    E20R.QZ_DEAD = E20R.QZ_DEAD * accum
    E20R.ema_update = functools.partial(_EMA_UPDATE_ORIG,
                                        dead_t=E20R.QZ_DEAD)
    assert orig_dead * accum == E20R.QZ_DEAD, (orig_dead, accum)
    print(f"micro {micro} (accum {accum}); QZ_DEAD {orig_dead} -> "
          f"{E20R.QZ_DEAD} forward passes (= {orig_dead} optimizer steps, "
          f"the w264 semantics), rebound via partial so _qz_slot picks it up",
          flush=True)

    gate_bypass(micro, lr_adamw)
    gate_capacity()
    gate_toy()
    register_predictions(accum)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if os.path.exists(os.path.join(G.OUT_DIR, f'{STEM}.pt')) \
            and 'run' in G.loadj(JP):
        print(f"{STEM}.pt exists -- done", flush=True)
        return
    print(f"==== training {STEM} ({total_steps} steps) ====", flush=True)
    log = M.train_muon_run(0.02, lr_adamw, total_steps, micro,
                           save_stem=STEM, f34k_held=f34k_held,
                           step_cb=step_cb)
    out = G.loadj(JP)
    out['run'] = {'lr_muon': 0.02, 'lr_adamw': lr_adamw,
                  'held_ce_scale_bf16': log.get('final_held_ce'),
                  'held_ce_f34k_bf16': log.get('final_f34k_ce'),
                  'spikes': log['spikes'], 'diverged': log['diverged'],
                  'peak_mem_mib': log.get('peak_mem_mib'),
                  'sec_per_step': log.get('sec_per_step_measured'),
                  'train_curve_every200': log['train_loss'],
                  'held100_scale_curve': log['held_ce']}
    G.savej(JP, out)
    print(json.dumps(out['run'], indent=2)[:600], flush=True)


if __name__ == '__main__':
    main()
