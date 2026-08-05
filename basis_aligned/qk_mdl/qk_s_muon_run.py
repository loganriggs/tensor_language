"""Width-1152 optimizer gate, Muon side (scale session; AdamW side is the
gc1e4 arm of qk_s_gate_run.py -- same architecture, same data order, so the
comparison isolates the optimizer).

Architecture: slots + group-lasso 1e-4 (the unconditional-recipe base).
Muon exactly as qk_e0m at small scale: from-scratch Muon per the
nanoGPT-speedrun convention (nesterov momentum 0.95, 5-step Newton-Schulz
orthogonalization in bf16, sqrt(max/min) aspect scale) on the 2D hidden
matrices; tied embedding + sub-2D params on AdamW at the family lr. The
group-lasso penalty stays IN THE LOSS (matching qk_e0m so scale and small
numbers compare; NOTE Muon's orthogonalization distorts loss-lasso shrinkage
-- Muon LOST by +0.076 at width 264, E7a's proximal fix was not yet pushed
when this launched).

Muon lr sweep {0.01, 0.02, 0.04} x 400 steps, widened one octave (<=2x) on
a grid-edge winner. The AdamW lr for the embedding/sub-2D split is read from
qk_s_w1152_gc1e4.json (falls back to the slots arm's winner, then 0.002);
the source is recorded in the JSON.

Everything else (data, held sets, micro-accumulation, curve conventions,
eval, outputs) reuses qk_s_gate_run. Outputs: qk_s_w1152_muonbase.{json,pt},
qk_s_w1152_muonbase_heldloss.npy / _f34kloss.npy. Run under
CUDA_VISIBLE_DEVICES=<free gpu>; TEST mode via QK_S_TEST=1 as in the gate.
"""
import os
import sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import qk_s_gate_run as G
import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_v9_common as C
import qk_w1152_train as W2
from qk_deeproute_train import DEPTH

TEST = G.TEST
OUT_DIR = G.OUT_DIR
MUON_LRS = [0.01, 0.02, 0.04] if not TEST else [0.02]
WIDEN_CAP_HI, WIDEN_CAP_LO = 0.08, 0.0025

# ---- round-4 variants (E7 landed: proximal lasso verified; combo = E1
# per-slot norm + proximal Muon is the recipe candidate). 'base' keeps the
# original loss-lasso optimizer-gate arm. prox arms put NO lasso in the loss;
# the penalty is applied as the exact decoupled proximal soft-threshold after
# each step with tau = lr_muon * schedule_factor * prox_coeff (E7a rule).
# prox/combo skip the lr sweep: 0.02 was the interior winner both at w264
# (qk_e0m, and E7 used it) and at w1152 (muonbase sweep) -- recorded in JSON.
ARM = (sys.argv[1] if len(sys.argv) > 1 else 'base')
CFG = {'base':  dict(stem='qk_s_w1152_muonbase', coeff=1e-4, prox=None,
                     sweep=True),
       'prox':  dict(stem='qk_s_w1152_muonprox', coeff=0.0, prox=1e-4,
                     sweep=False),
       'combo': dict(stem='qk_s_w1152_combo', coeff=0.0, prox=1e-4,
                     sweep=False),
       # THE recipe candidate after the probe verdicts: proximal kills
       # readability (combo/muonprox Spearman ~0 or negative) but in-loss
       # keeps it under Muon (muonbase 0.88) -- so: per-slot norm + Muon +
       # IN-LOSS lasso at the 3e-5 readability point (gc3e5 Spearman 0.76)
       'combo3e5loss': dict(stem='qk_s_w1152_combo3e5loss', coeff=3e-5,
                            prox=None, sweep=False),
       # Muon vanilla control: prices the combo against the best optimizer's
       # vanilla, not just AdamW's (Muon won vanilla -0.094 at w264, qk_e0m)
       'vanilla': dict(stem='qk_s_w1152_muonvanilla', coeff=0.0, prox=None,
                       sweep=False)}[ARM]
COEFF = CFG['coeff']
PROX = CFG['prox']
STEM = CFG['stem']
JP = os.path.join(OUT_DIR, f'{STEM}.json')

if ARM in ('combo', 'combo3e5loss'):
    import qk_s_e1_run as E1R           # guard already neutered via G
    factory = E1R.make_e1               # per-slot RMSNorm slots model
elif ARM == 'vanilla':
    factory = W2.make_vanilla1152       # zero-init-write MiniBilin A
else:
    factory = lambda: C.make_variant('W1152', None)

READ_NAMES = ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v', 'Left', 'Right')
NGROUP = 2 * DEPTH


@torch.no_grad()
def prox_group_lasso(model, tau):
    """Exact proximal operator of tau * sum_groups ||g||_F on the read-matrix
    slot column groups (verbatim from qk_e_common; E7's permanent known-answer
    control verified this implementation tracks lasso-free Muon within 1e-4
    nats at tau->0 and produces no spurious zeros)."""
    for blk in model.h:
        for nm in READ_NAMES:
            Mw = getattr(blk, nm).weight
            S = Mw.shape[1] // NGROUP
            norms = Mw.detach().float().pow(2) \
                      .view(Mw.shape[0], NGROUP, S).sum((0, 2)).sqrt()
            scale = (1.0 - tau / norms.clamp_min(1e-12)).clamp_min(0.0)
            Mw.mul_(scale.repeat_interleave(S).to(Mw.dtype)[None, :])


# ---- from-scratch Muon, verbatim from qk_e_common (kept import-light) ----
def ns_orth(Gm, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = Gm.to(torch.bfloat16)
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                assert p.dim() == 2, "Muon is for 2D matrices only"
                g = p.grad
                st = self.state[p]
                if 'buf' not in st:
                    st['buf'] = torch.zeros_like(g)
                buf = st['buf']
                buf.mul_(group['momentum']).add_(g)
                u = g.add(buf, alpha=group['momentum']) \
                    if group['nesterov'] else buf
                u = ns_orth(u, group['ns_steps']).to(p.dtype)
                r, cdim = p.shape
                scale = math.sqrt(max(r, cdim) / min(r, cdim))
                p.add_(u, alpha=-group['lr'] * scale)


def muon_params_split(model):
    mu, adamw_decay, adamw_nodecay = [], [], []
    for nm, p in model.named_parameters():
        if p.dim() >= 2 and not nm.startswith('wte'):
            mu.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)
    return mu, adamw_decay, adamw_nodecay


def resolve_lr_adamw():
    for src in ('qk_s_w1152_gc1e4.json', 'qk_s_w1152_slots.json'):
        p = os.path.join(OUT_DIR, src)
        if os.path.exists(p):
            d = json.load(open(p))
            if 'lrsweep' in d:
                return d['lrsweep']['chosen'], src
    return 0.002, 'fallback'


def train_muon_run(lr_muon, lr_adamw, total_steps, micro, save_stem=None,
                   log_every=200, held_every=2000, f34k_held=None):
    Q.gpu_guard(min_free=7000)
    model = factory()
    mu, dec, nod = muon_params_split(model)
    opt_m = Muon(mu, lr=lr_muon)
    opt_a = torch.optim.AdamW([{'params': dec, 'weight_decay': Q.WD},
                               {'params': nod, 'weight_decay': 0.0}],
                              lr=lr_adamw, betas=(0.9, 0.95))

    def fac(step):
        if step < Q.WARMUP:
            return (step + 1) / Q.WARMUP
        p = (step - Q.WARMUP) / max(1, total_steps - Q.WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))

    log = {'arm': f'muon_{ARM}', 'lr': lr_muon, 'lr_adamw': lr_adamw,
           'group_coeff': COEFF, 'prox_coeff': PROX, 'steps': total_steps,
           'micro_batch': micro, 'eff_batch': G.EFF_BATCH, 'train_loss': [],
           'held_ce': [], 'spikes': 0, 'diverged': False}
    order = Q.epoch_order(0)                # identical to every gate arm
    t0, run = time.time(), None
    step_times = []
    model.train()
    for step in range(total_steps):
        ts = time.time()
        f = fac(step)
        for gpg in opt_m.param_groups:
            gpg['lr'] = lr_muon * f
        for gpg in opt_a.param_groups:
            gpg['lr'] = lr_adamw * f
        seqs = Q.TRAIN[order[step * G.EFF_BATCH:(step + 1) * G.EFF_BATCH]]
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        ce_step = 0.0
        for j in range(0, G.EFF_BATCH, micro):
            chunk = seqs[j:j + micro]
            frac = chunk.shape[0] / G.EFF_BATCH
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(chunk[:, :Q.T])
            ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                 chunk[:, 1:Q.T + 1].reshape(-1))
            loss = ce * frac
            if j == 0 and COEFF > 0:
                loss = loss + COEFF * V8T.group_penalty(model)
            loss.backward()
            ce_step += ce.item() * frac
        if not math.isfinite(ce_step) or ce_step > 30:
            log['diverged'] = True
            log['diverged_at'] = step
            print(f"  MUON DIVERGED at step {step} (ce {ce_step})", flush=True)
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
        opt_m.step()
        opt_a.step()
        if PROX is not None:
            prox_group_lasso(model, lr_muon * f * PROX)
        if 20 <= step < 100:
            torch.cuda.synchronize()
            step_times.append(time.time() - ts)
        run = ce_step if run is None else 0.98 * run + 0.02 * ce_step
        if ce_step > run + 1.0:
            log['spikes'] += 1
        if step % log_every == 0:
            log['train_loss'].append([step, round(ce_step, 4), round(run, 4)])
            print(f"  MUON[lr {lr_muon}/{lr_adamw}] step {step}/{total_steps} "
                  f"ce {ce_step:.4f} (ema {run:.4f}) {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB",
                  flush=True)
        if step == 100 and step_times:
            spt = sum(step_times) / len(step_times)
            log['sec_per_step_measured'] = round(spt, 3)
            print(f"  MUON: measured {spt:.3f} s/step", flush=True)
        if save_stem and step > 0 and step % held_every == 0:
            hce, _ = G.eval_data(model, Q.HELD, n_seq=100)
            log['held_ce'].append([step, round(hce, 4)])
            print(f"  MUON step {step} held100 scale CE {hce:.4f}", flush=True)
    log['peak_mem_mib'] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    if log['diverged']:
        log['final_held_ce'] = float('inf')
        del model, opt_m, opt_a
        torch.cuda.empty_cache()
        return log
    log['final_penalty'] = float(V8T.group_penalty(model).detach())
    if save_stem is None:
        ce100, _ = G.eval_data(model, Q.HELD, n_seq=100)
        log['final_held_ce'] = ce100
        print(f"  muon sweep lr {lr_muon}: held100 scale CE {ce100:.4f} "
              f"({log['spikes']} spikes, {time.time() - t0:.0f}s)", flush=True)
    else:
        hce, pt = G.eval_data(model, Q.HELD, per_token=True)
        log['final_held_ce'] = hce
        fce, fpt = G.eval_data(model, f34k_held, per_token=True)
        log['final_f34k_ce'] = fce
        print(f"== MUON FINAL held scale CE {hce:.5f}  f34k CE {fce:.5f}  "
              f"({time.time() - t0:.0f}s, {log['spikes']} spikes)", flush=True)
        torch.save({'state_dict': model.state_dict(), 'variant': 'W1152',
                    'config': dict(NL=DEPTH, D=G.WIDTH, NH=G.WIDTH // 64,
                                   HD=64, V=Q.V, EXP=Q.EXP, T=Q.T,
                                   BATCH=G.EFF_BATCH, micro_batch=micro,
                                   EPOCHS=1, SEED=Q.SEED,
                                   DATA_SEED=Q.DATA_SEED, lr=lr_muon,
                                   lr_adamw=lr_adamw, group_coeff=COEFF,
                                   optimizer='muon', WARMUP=Q.WARMUP,
                                   steps=total_steps,
                                   write_init_std=R.WRITE_INIT_STD),
                    'log': log},
                   os.path.join(OUT_DIR, f'{save_stem}.pt'))
        np.save(os.path.join(OUT_DIR, f'{save_stem}_heldloss.npy'), pt)
        np.save(os.path.join(OUT_DIR, f'{save_stem}_f34kloss.npy'), fpt)
    del model, opt_m, opt_a
    torch.cuda.empty_cache()
    return log


def preflight(out, lr_adamw):
    if 'preflight' in out:
        return out['preflight']['micro']
    for micro in (G.EFF_BATCH, 16, 8):
        try:
            torch.cuda.reset_peak_memory_stats()
            model = factory()
            mu, dec, nod = muon_params_split(model)
            opt_m, opt_a = Muon(mu, lr=1e-3), torch.optim.AdamW(dec + nod,
                                                                lr=1e-4)
            for i in range(2):
                seqs = Q.TRAIN[i * G.EFF_BATCH:(i + 1) * G.EFF_BATCH]
                opt_m.zero_grad(set_to_none=True)
                opt_a.zero_grad(set_to_none=True)
                for j in range(0, G.EFF_BATCH, micro):
                    chunk = seqs[j:j + micro]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(chunk[:, :Q.T])
                    ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                         chunk[:, 1:Q.T + 1].reshape(-1))
                    loss = ce * (chunk.shape[0] / G.EFF_BATCH)
                    if j == 0:
                        loss = loss + COEFF * V8T.group_penalty(model)
                    loss.backward()
                opt_m.step()
                opt_a.step()
            peak = int(torch.cuda.max_memory_allocated() / 2 ** 20)
            del model, opt_m, opt_a
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"preflight muon: micro {micro} peak {peak} MiB", flush=True)
            if peak < G.MEM_BUDGET_MIB:
                out['preflight'] = {'micro': micro, 'peak_mib': peak}
                G.savej(JP, out)
                return micro
        except torch.cuda.OutOfMemoryError:
            print(f"preflight muon: micro {micro} OOM", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    raise RuntimeError("muon does not fit even at micro 8")


def main():
    out = G.loadj(JP)
    W2.patch_width(G.WIDTH)
    total_steps, spec, f34k_held = G.setup_data()
    lr_adamw, lr_src = resolve_lr_adamw()
    out['env'] = {'gpu': torch.cuda.get_device_name(0),
                  'torch': torch.__version__, 'cooc_substitute': True}
    out['data'] = spec
    out['arm'] = {'name': f'muon_{ARM}', 'group_coeff': COEFF,
                  'prox_coeff': PROX,
                  'architecture': {'combo': 'E1 per-slot RMSNorm slots',
                                   'combo3e5loss': 'E1 per-slot RMSNorm slots',
                                   'vanilla': 'vanilla MiniBilin A'}.get(
                                       ARM, 'slots (W1152)'),
                  'optimizer': 'muon(2D hidden) + adamw(wte, sub-2D)',
                  'lr_adamw': lr_adamw, 'lr_adamw_source': lr_src,
                  'penalty_in_loss': COEFF > 0,
                  'write_init_std': R.WRITE_INIT_STD}
    G.savej(JP, out)
    micro = preflight(out, lr_adamw)
    print(f"muon: micro {micro} (accum {G.EFF_BATCH // micro}), "
          f"lr_adamw {lr_adamw} from {lr_src}", flush=True)

    key = 'lrsweep_muon'
    if not CFG['sweep'] and key not in out:
        out[key] = {'chosen': 0.02, 'ranking': [0.02, 0.01],
                    'note': 'no re-sweep: 0.02 interior winner at w264 '
                            '(qk_e0m, E7) and at w1152 (muonbase sweep); '
                            '0.01 as divergence fallback', 'lr_adamw': lr_adamw}
        G.savej(JP, out)
    if key not in out:
        grid = sorted(MUON_LRS)
        res, widened = {}, 0
        while True:
            for lr in grid:
                if str(lr) in res:
                    continue
                print(f"-- muon w1152 lr sweep {lr} x {G.SWEEP_STEPS} steps",
                      flush=True)
                log = train_muon_run(lr, lr_adamw, G.SWEEP_STEPS, micro)
                res[str(lr)] = {'held100_scale_ce':
                                (None if log['diverged'] else
                                 round(log['final_held_ce'], 4)),
                                'diverged': log['diverged'],
                                'spikes': log['spikes']}
            ok = {k: v for k, v in res.items() if not v['diverged']}
            assert ok, "every muon sweep lr diverged"
            ranking = sorted(ok, key=lambda k: ok[k]['held100_scale_ce'])
            chosen = float(ranking[0])
            if TEST or widened >= 2:
                break
            if chosen == max(float(k) for k in res) \
                    and chosen * 2 <= WIDEN_CAP_HI:
                grid, widened = [chosen * 2], widened + 1
                print(f"muon: winner {chosen} at TOP edge -> widen", flush=True)
            elif chosen == min(float(k) for k in res) \
                    and chosen / 2 >= WIDEN_CAP_LO:
                grid, widened = [chosen / 2], widened + 1
                print(f"muon: winner {chosen} at BOTTOM edge -> widen",
                      flush=True)
            else:
                break
        out[key] = {'results': res, 'ranking': [float(k) for k in ranking],
                    'chosen': chosen, 'sweep_steps': G.SWEEP_STEPS,
                    'widened': widened, 'lr_adamw': lr_adamw}
        G.savej(JP, out)
    ranking = out[key]['ranking']
    print(f"muon lr chosen: {out[key]['chosen']} (ranking {ranking})",
          flush=True)

    if os.path.exists(os.path.join(OUT_DIR, f'{STEM}.pt')) and 'run' in out:
        print(f"{STEM}.pt exists -- done", flush=True)
        return
    for pick, lr in enumerate(ranking):
        print(f"==== training muonbase w1152 (muon lr {lr}"
              + (", fallback" if pick else "") + ") ====", flush=True)
        log = train_muon_run(lr, lr_adamw, total_steps, micro, save_stem=STEM,
                             f34k_held=f34k_held)
        if not log['diverged']:
            break
    out['run'] = {'lr_muon': lr, 'lr_adamw': lr_adamw, 'lr_fallback': pick > 0,
                  'held_ce_scale_bf16': log.get('final_held_ce'),
                  'held_ce_f34k_bf16': log.get('final_f34k_ce'),
                  'spikes': log['spikes'], 'diverged': log['diverged'],
                  'final_penalty': log.get('final_penalty'),
                  'peak_mem_mib': log.get('peak_mem_mib'),
                  'sec_per_step': log.get('sec_per_step_measured'),
                  'train_curve_every200': log['train_loss'],
                  'held100_scale_curve': log['held_ce']}
    G.savej(JP, out)
    print(json.dumps({k: out['run'][k] for k in
                      ('lr_muon', 'lr_adamw', 'held_ce_scale_bf16',
                       'held_ce_f34k_bf16', 'spikes', 'diverged')},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
