"""SCALE GATE at width 1152 (rented-box scale session, qk_s_ prefix).

One arm per invocation: `python qk_s_gate_run.py {vanilla|slots|gc3e5}` --
launch one per GPU via CUDA_VISIBLE_DEVICES. Arms (per the 2026-08-04
re-pricing in SCALE_RUN.md):

  vanilla  zero-init-write MiniBilin variant A, depth 12, width 1152
  slots    subspace-partitioned write slots + nonzero write init, NO lasso
  gc3e5    slots + group-lasso 3e-5 on reads (the readability point chosen
           from the E5 frontier)
  gc1e4    slots + group-lasso 1e-4 (the original unconditional-recipe base;
           doubles as the AdamW side of the width-1152 optimizer gate --
           the Muon side is qk_s_muon_run.py)

Fresh single-epoch no-memorization protocol at width 1152 (bilin18's width;
slot dim 48): train prefix = corpus_fresh shards 00..06 concatenated, rows
[0:298496] (docs 45367..267574), effective batch 32 x 9328 steps, each
sequence seen exactly once, identical data order epoch_order(0) across arms.
Scale held = shard06 LAST 1500 rows (never trained; rows 298500..299999 of
the concat -- the 4-row gap [298496:298500] is dropped, never used anywhere).
Also evaluated: fresh34k rows [33000:34500], the small-scale E-run held set,
for direct comparability with qk_e0/e1..e5 numbers.

Positive controls BEFORE training (every arm process): slots model with
identity projections + zeroed writes == vanilla-1152 at init; vectorized
group penalty == naive loop at slot size 48; grad-accumulation equivalence.

Micro-batch preflight: largest of {32, 16, 8} whose 2-step peak stays under
MEM_BUDGET_MIB on the arm's own architecture (32 GB cards here, not 40).
lr sweep per arm: 400 steps over {0.001, 0.002, 0.004}, auto-widened one
octave (up to 2 times) if the winner lands on a grid edge.

Conventions: bf16 autocast train + eval, 30*tanh(logits/30) (in the model),
nonzero write init std 0.02/sqrt(2*DEPTH)/sqrt(1152/384) via patch_width,
grad clip 1.0, AdamW betas (0.9, 0.95) wd 0.1 (2D only), warmup 250 cosine,
train-CE curve every 200 steps + held-100 CE every 2000 steps in the JSON.

This box lacks the original cooc corpus; /workspace/tensor_language/
data_fineweb_cooc_tokens.npy is a SUBSTITUTE (fresh34k rows [0:6000], pure
eval docs -- see the .SUBSTITUTE_NOTE.txt beside it). No old-held cooc
numbers are produced here.

Outputs (per arm): qk_s_w1152_{arm}.json, qk_s_w1152_{arm}.pt,
qk_s_w1152_{arm}_heldloss.npy (scale held, 1500 seqs per-token),
qk_s_w1152_{arm}_f34kloss.npy (fresh34k held, 1500 seqs per-token).
Paired stats are computed by qk_s_gate_pair.py once two arms exist.

TEST MODE (QK_S_TEST=1): 12-step sweep at one lr + 30-step run, held-50,
outputs under the scratchpad; exercises every code path on the real GPU.
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

import qk_tokenline_train as Q

# One-arm-per-GPU discipline: each process owns its CUDA_VISIBLE_DEVICES card
# outright. Q.gpu_guard parses nvidia-smi's FIRST line = physical GPU 0
# regardless of CUDA_VISIBLE_DEVICES, so on the second card it deadlocks
# against the first card's job and then raises. Neuter it BEFORE the imports
# below: qk_v9_common -> qk_deeproute_train_2 -> qk_tokenline_probe calls
# Q.gpu_guard() AT IMPORT TIME (this killed the E1 launch while GPU 0 was
# busy; the earlier arms only survived because GPU 0 happened to be free
# when they imported). Covers every runner that imports this module.
Q.gpu_guard = lambda *a, **k: None

import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_v9_common as C
import qk_w1152_train as W2
from qk_deeproute_train import DEPTH

TEST = os.environ.get('QK_S_TEST') == '1'
QK = C.QK
OUT_DIR = QK
if TEST:
    OUT_DIR = os.path.join(os.environ.get('QK_S_TEST_DIR', '/tmp'), 'qk_s_test')
    os.makedirs(OUT_DIR, exist_ok=True)

WIDTH = 1152
EFF_BATCH = 32
SWEEP_STEPS = 12 if TEST else 400
BASE_LRS = [0.001, 0.002, 0.004] if not TEST else [0.002]
MAX_WIDEN = 2
WIDEN_CAP_HI, WIDEN_CAP_LO = 0.016, 0.00025
MEM_BUDGET_MIB = 29000
HELD_N = 50 if TEST else 1500
ARMS = ('vanilla', 'slots', 'gc3e5', 'gc1e4')
COEFF = {'vanilla': 0.0, 'slots': 0.0, 'gc3e5': 3e-5, 'gc1e4': 1e-4}


def factory_for(arm):
    if arm == 'vanilla':
        return W2.make_vanilla1152
    return lambda: C.make_variant('W1152', None)


def jp_of(arm):
    return os.path.join(OUT_DIR, f'qk_s_w1152_{arm}.json')


def loadj(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def savej(p, out):
    json.dump(out, open(p, 'w'), indent=2)


# ---------------- data ----------------
def setup_data():
    """Fresh train prefix + scale held + f34k held; returns (steps, spec)."""
    shards = [np.load(f'{QK}/corpus_fresh/shard{i:02d}.npy', mmap_mode='r')
              for i in range(7)]
    total = sum(len(a) for a in shards)               # 300000
    n_train_max = total - 1500                        # scale held excluded
    steps = (30 if TEST else n_train_max // EFF_BATCH)
    need = steps * EFF_BATCH
    assert need <= n_train_max
    rows, got = [], 0
    for a in shards:
        take = min(len(a), need - got)
        if take > 0:
            rows.append(np.asarray(a[:take]))
            got += take
        if got >= need:
            break
    Q.TRAIN = torch.from_numpy(
        np.concatenate(rows).astype(np.int64)).to('cuda')
    Q.NTR = need
    Q.BATCH = EFF_BATCH
    Q.STEPS_PER_EPOCH = steps
    held = np.asarray(shards[6][-1500:][:HELD_N]).astype(np.int64)
    Q.HELD = torch.from_numpy(held).to('cuda')
    f34k = np.load(f'{QK}/corpus_fresh/fresh34k.npy',
                   mmap_mode='r')[33000:33000 + HELD_N].astype(np.int64)
    f34k_held = torch.from_numpy(f34k).to('cuda')
    spec = {'train': f'corpus_fresh shards 00..06 concat rows [0:{need}], '
                     'single pass, epoch_order(0)',
            'n_train': need, 'steps': steps, 'eff_batch': EFF_BATCH,
            'held_scale': f'shard06 last 1500 rows, first {HELD_N} used',
            'held_f34k': f'fresh34k rows [33000:{33000 + HELD_N}]'}
    print(f"data: {need} train rows, {steps} steps x eff batch {EFF_BATCH}; "
          f"held scale {HELD_N} + f34k {HELD_N}", flush=True)
    return steps, spec, f34k_held


@torch.no_grad()
def eval_data(model, data, per_token=False, n_seq=None, batch=16):
    """bf16-autocast eval on an explicit dataset (Q.eval_held convention)."""
    model.eval()
    hs = data if n_seq is None else data[:n_seq]
    tot, n, pts = 0.0, 0, []
    for i in range(0, len(hs), batch):
        b = hs[i:i + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(b[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        if per_token:
            pts.append(ce.cpu())
        tot += ce.sum().item()
        n += ce.numel()
    model.train()
    return tot / n, (torch.cat(pts).numpy() if per_token else None)


# ---------------- preflight ----------------
def preflight_micro(arm, out):
    if 'preflight' in out:
        return out['preflight']['micro']
    fac = factory_for(arm)
    for micro in (EFF_BATCH, 16, 8):
        model = opt = None
        try:
            torch.cuda.reset_peak_memory_stats()
            model = fac()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            for i in range(2):
                seqs = Q.TRAIN[i * EFF_BATCH:(i + 1) * EFF_BATCH]
                opt.zero_grad(set_to_none=True)
                for j in range(0, EFF_BATCH, micro):
                    chunk = seqs[j:j + micro]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(chunk[:, :Q.T])
                    ce = F.cross_entropy(
                        logits.float().reshape(-1, Q.V),
                        chunk[:, 1:Q.T + 1].reshape(-1))
                    loss = ce * (chunk.shape[0] / EFF_BATCH)
                    if j == 0 and COEFF[arm] > 0:
                        loss = loss + COEFF[arm] * V8T.group_penalty(model)
                    loss.backward()
                opt.step()
            peak = int(torch.cuda.max_memory_allocated() / 2 ** 20)
            del model, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"preflight {arm}: micro {micro} peak {peak} MiB "
                  f"(budget {MEM_BUDGET_MIB})", flush=True)
            if peak < MEM_BUDGET_MIB:
                out['preflight'] = {'micro': micro, 'peak_mib': peak}
                savej(jp_of(arm), out)
                return micro
        except torch.cuda.OutOfMemoryError:
            print(f"preflight {arm}: micro {micro} OOM", flush=True)
            if model is not None:
                del model
            if opt is not None:
                del opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    raise RuntimeError("width 1152 does not fit even at micro 8")


# ---------------- training ----------------
def train_run(arm, lr, total_steps, micro, save_stem=None, log_every=200,
              held_every=2000, f34k_held=None):
    Q.gpu_guard(min_free=7000)
    coeff = COEFF[arm]
    model = factory_for(arm)()
    decay, nodecay = [], []
    for nm, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': Q.WD},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < Q.WARMUP:
            return lr * (step + 1) / Q.WARMUP
        p = (step - Q.WARMUP) / max(1, total_steps - Q.WARMUP)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    log = {'arm': arm, 'lr': lr, 'group_coeff': coeff, 'steps': total_steps,
           'micro_batch': micro, 'eff_batch': EFF_BATCH, 'train_loss': [],
           'held_ce': [], 'spikes': 0, 'diverged': False}
    order = Q.epoch_order(0)                # identical across arms (same NTR)
    t0, run = time.time(), None
    step_times = []
    model.train()
    for step in range(total_steps):
        ts = time.time()
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        seqs = Q.TRAIN[order[step * EFF_BATCH:(step + 1) * EFF_BATCH]]
        opt.zero_grad(set_to_none=True)
        ce_step = 0.0
        for j in range(0, EFF_BATCH, micro):
            chunk = seqs[j:j + micro]
            frac = chunk.shape[0] / EFF_BATCH
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(chunk[:, :Q.T])
            ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                 chunk[:, 1:Q.T + 1].reshape(-1))
            loss = ce * frac
            if coeff > 0 and j == 0:
                loss = loss + coeff * V8T.group_penalty(model)
            loss.backward()
            ce_step += ce.item() * frac
        if not math.isfinite(ce_step) or ce_step > 30:
            log['diverged'] = True
            log['diverged_at'] = step
            print(f"  {arm} DIVERGED at step {step} (ce {ce_step})", flush=True)
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
        opt.step()
        if 20 <= step < 100:
            torch.cuda.synchronize()
            step_times.append(time.time() - ts)
        run = ce_step if run is None else 0.98 * run + 0.02 * ce_step
        if ce_step > run + 1.0:
            log['spikes'] += 1
        if step % log_every == 0:
            log['train_loss'].append([step, round(ce_step, 4), round(run, 4)])
            print(f"  {arm}[lr {lr}] step {step}/{total_steps} ce {ce_step:.4f}"
                  f" (ema {run:.4f}) {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB",
                  flush=True)
        if step == 100 and step_times:
            spt = sum(step_times) / len(step_times)
            log['sec_per_step_measured'] = round(spt, 3)
            print(f"  {arm}: measured {spt:.3f} s/step -> full run "
                  f"~{spt * total_steps / 3600:.2f} h", flush=True)
        if save_stem and step > 0 and step % held_every == 0:
            hce, _ = eval_data(model, Q.HELD, n_seq=100)
            log['held_ce'].append([step, round(hce, 4)])
            print(f"  {arm} step {step} held100 scale CE {hce:.4f}", flush=True)
    log['peak_mem_mib'] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    if log['diverged']:
        log['final_held_ce'] = float('inf')
        del model, opt
        torch.cuda.empty_cache()
        return log
    if coeff > 0:
        log['final_penalty'] = float(V8T.group_penalty(model).detach())
    if save_stem is None:                    # sweep run: held-100 CE only
        ce100, _ = eval_data(model, Q.HELD, n_seq=100)
        log['final_held_ce'] = ce100
        print(f"  sweep {arm} lr {lr}: held100 scale CE {ce100:.4f} "
              f"({log['spikes']} spikes, {time.time() - t0:.0f}s)", flush=True)
    else:
        hce, pt = eval_data(model, Q.HELD, per_token=True)
        log['final_held_ce'] = hce
        fce, fpt = eval_data(model, f34k_held, per_token=True)
        log['final_f34k_ce'] = fce
        print(f"== {arm} FINAL held scale CE {hce:.5f}  f34k CE {fce:.5f}  "
              f"({time.time() - t0:.0f}s, {log['spikes']} spikes)", flush=True)
        torch.save({'state_dict': model.state_dict(),
                    'variant': ('A' if arm == 'vanilla' else 'W1152'),
                    'config': dict(NL=DEPTH, D=WIDTH, NH=WIDTH // 64, HD=64,
                                   V=Q.V, EXP=Q.EXP, T=Q.T, BATCH=EFF_BATCH,
                                   micro_batch=micro, EPOCHS=1, SEED=Q.SEED,
                                   DATA_SEED=Q.DATA_SEED, lr=lr,
                                   group_coeff=coeff, WARMUP=Q.WARMUP,
                                   WD=Q.WD, GRAD_CLIP=Q.GRAD_CLIP,
                                   steps=total_steps,
                                   write_init_std=(None if arm == 'vanilla'
                                                   else R.WRITE_INIT_STD)),
                    'log': log},
                   os.path.join(OUT_DIR, f'{save_stem}.pt'))
        np.save(os.path.join(OUT_DIR, f'{save_stem}_heldloss.npy'), pt)
        np.save(os.path.join(OUT_DIR, f'{save_stem}_f34kloss.npy'), fpt)
    del model, opt
    torch.cuda.empty_cache()
    return log


# ---------------- lr sweep with auto-widening ----------------
def lr_sweep(arm, micro, out):
    key = 'lrsweep'
    if key in out:
        return out[key]['chosen'], out[key]['ranking']
    grid = sorted(BASE_LRS)
    res = {}
    widened = 0
    while True:
        for lr in grid:
            if str(lr) in res:
                continue
            print(f"-- {arm} w1152 lr sweep {lr} x {SWEEP_STEPS} steps",
                  flush=True)
            log = train_run(arm, lr, SWEEP_STEPS, micro)
            res[str(lr)] = {'held100_scale_ce': (None if log['diverged'] else
                                                 round(log['final_held_ce'], 4)),
                            'diverged': log['diverged'],
                            'spikes': log['spikes']}
        ok = {k: v for k, v in res.items() if not v['diverged']}
        assert ok, f"{arm}: every sweep lr diverged"
        ranking = sorted(ok, key=lambda k: ok[k]['held100_scale_ce'])
        chosen = float(ranking[0])
        if TEST or widened >= MAX_WIDEN:
            break
        if chosen == max(float(k) for k in res) and chosen * 2 <= WIDEN_CAP_HI:
            grid = [chosen * 2]
            widened += 1
            print(f"{arm}: winner {chosen} at grid TOP edge -> widen to "
                  f"{grid}", flush=True)
        elif chosen == min(float(k) for k in res) \
                and chosen / 2 >= WIDEN_CAP_LO:
            grid = [chosen / 2]
            widened += 1
            print(f"{arm}: winner {chosen} at grid BOTTOM edge -> widen to "
                  f"{grid}", flush=True)
        else:
            break
    out[key] = {'results': res, 'ranking': [float(k) for k in ranking],
                'chosen': chosen, 'sweep_steps': SWEEP_STEPS,
                'widened': widened,
                'note': (f'group-lasso {COEFF[arm]} active during sweep'
                         if COEFF[arm] > 0 else '')}
    savej(jp_of(arm), out)
    print(f"{arm} lr chosen: {chosen} (ranking {out[key]['ranking']})",
          flush=True)
    return chosen, out[key]['ranking']


# ---------------- positive controls ----------------
def run_controls(arm, out):
    if out.get('controls_ok'):
        return
    W2.controls()               # identity + penalty vectorization at width 1152
    W2.accum_control()          # micro-accumulated grads == one-shot grads
    out['controls_ok'] = True
    savej(jp_of(arm), out)


def main():
    arm = sys.argv[1]
    assert arm in ARMS, f"arm must be one of {ARMS}"
    stem = f'qk_s_w1152_{arm}'
    jp = jp_of(arm)
    out = loadj(jp)
    W2.patch_width(WIDTH)
    total_steps, spec, f34k_held = setup_data()
    out['env'] = {'gpu': torch.cuda.get_device_name(0),
                  'torch': torch.__version__,
                  'cooc_substitute': True}
    out['data'] = spec
    out['arm'] = {'name': arm, 'group_coeff': COEFF[arm],
                  'write_init_std': (None if arm == 'vanilla'
                                     else R.WRITE_INIT_STD)}
    savej(jp, out)
    run_controls(arm, out)
    micro = preflight_micro(arm, out)
    print(f"{arm}: micro {micro} (accum {EFF_BATCH // micro})", flush=True)
    chosen, ranking = lr_sweep(arm, micro, out)
    if os.path.exists(os.path.join(OUT_DIR, f'{stem}.pt')) and 'run' in out:
        print(f"{stem}.pt exists -- done", flush=True)
        return
    for pick, lr in enumerate(ranking):
        print(f"==== training {arm} w1152 (lr {lr}"
              + (", fallback" if pick else "") + ") ====", flush=True)
        log = train_run(arm, lr, total_steps, micro, save_stem=stem,
                        f34k_held=f34k_held)
        if not log['diverged']:
            break
    out['run'] = {'lr': lr, 'lr_fallback': pick > 0,
                  'held_ce_scale_bf16': log.get('final_held_ce'),
                  'held_ce_f34k_bf16': log.get('final_f34k_ce'),
                  'spikes': log['spikes'], 'diverged': log['diverged'],
                  'final_penalty': log.get('final_penalty'),
                  'peak_mem_mib': log.get('peak_mem_mib'),
                  'sec_per_step': log.get('sec_per_step_measured'),
                  'train_curve_every200': log['train_loss'],
                  'held100_scale_curve': log['held_ce']}
    savej(jp, out)
    print(json.dumps({k: out['run'][k] for k in
                      ('lr', 'held_ce_scale_bf16', 'held_ce_f34k_bf16',
                       'spikes', 'diverged')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
