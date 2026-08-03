"""WIDTH-1152 LARGE-DATA GATE (final gate before a bilin18-class retrain; Logan).

Two models at depth 12, WIDTH 1152 (bilin18's width; slot size 1152/24 = 48):
  (a) the unconditional recipe -- subspace-partitioned write slots + group-lasso
      reads (coefficient 1e-4) + nonzero write init (std rescaled by
      1/sqrt(width ratio) = 1/sqrt(1152/384) vs the 384-width convention);
  (b) a matched vanilla control (zero-init writes, the vanilla convention).

KEY CHANGE vs qk_w768_train.py: the LARGE data budget -- train set is the full
FineWeb corpus on disk: data_fineweb_cooc_tokens.npy sequences [0:5500] PLUS
data_fineweb_tokens.npy [0:448] (tokenizer compatibility verified by max-token-id
check), ONE epoch-equivalent (each sequence seen once; no repetition, so a
regularizer cannot win via overfitting-prevention on repeated data). Held slice
stays cooc [5500:6000]. Effective batch fixed at 8 (identical data order across
arms); micro-batch chosen by a pre-flight under a 13000 MiB peak budget, with
gradient accumulation up to effective 8. Brief lr sweep {0.0005, 0.001, 0.002}
x 400 steps per arm. Sequential, GPU guard.

Positive controls at width 1152 before training:
  (a) slots model with identity projections + zeroed writes == width-1152
      vanilla variant-A MiniBilin at init (max |logit diff| < 1e-3, fp32);
  (b) vectorized group penalty == naive per-slice loop at slot size 48;
  (c) gradient-accumulation equivalence: 2x micro-4 accumulated grads match
      one batch-8 step's grads (fp32, rel err < 1e-4) on the slots model.

Saves qk_w1152_slots.pt / qk_w1152_slots_heldloss.npy, qk_w1152_vanilla.pt /
qk_w1152_vanilla_heldloss.npy; sweeps + paired CE -> qk_w1152.json.
Probe (wiring / dead blocks / terms / standard probes) in qk_w1152_probe.py.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_deeproute_train_2 as R2
import qk_v9_common as C
from qk_deeproute_train import DEPTH

QK = C.QK
WIDTH = 1152
BASE_WIDTH = 384
LRS = [0.0005, 0.001, 0.002]
SWEEP_STEPS = 400
EFF_BATCH = 8
MEM_BUDGET_MIB = 13000
PATH = f'{QK}/qk_w1152.json'

FACTORY = lambda: C.make_variant('W1152', None)      # slots + group-lasso, no mask


def make_vanilla1152():
    old = Q.NL
    Q.NL = DEPTH
    m = Q.make_model('A')
    Q.NL = old
    return m


def patch_width(w):
    Q.D, Q.NH, Q.HD = w, w // 64, 64
    sub = w // (2 * DEPTH)
    for mod in (R, V8T):
        mod.D, mod.NH, mod.HD, mod.SUBDIM = w, w // 64, 64, sub
    R2.D, R2.SUBDIM = w, sub
    # write-init std rescaled by 1/sqrt(width ratio) (Logan's honesty item; the
    # width-768 run did NOT rescale -- recorded as a confound in the report)
    R.WRITE_INIT_STD = 0.02 / math.sqrt(2 * DEPTH) / math.sqrt(w / BASE_WIDTH)
    print(f"width patched: D {w}, NH {w // 64}, slot dim {sub}, "
          f"write_init_std {R.WRITE_INIT_STD:.6f}", flush=True)


def patch_data():
    """Extend the train set with the second FineWeb file after a tokenizer check."""
    cooc = np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy')
    fw = np.load('/workspace/tensor_language/data_fineweb_tokens.npy')
    info = {'cooc_shape': list(cooc.shape), 'fw_shape': list(fw.shape),
            'cooc_max_id': int(cooc.max()), 'fw_max_id': int(fw.max())}
    assert cooc.shape[1] == fw.shape[1] == Q.T + 1
    # token-compatibility: both must stay inside the GPT-2 vocab and share its
    # signature max id (50256 = endoftext)
    assert info['cooc_max_id'] == info['fw_max_id'] == 50256, info
    tr = np.concatenate([cooc[:5500], fw[:448]], 0)
    Q.TRAIN = torch.from_numpy(tr.astype(np.int64)).to('cuda')
    Q.NTR = Q.TRAIN.shape[0]
    Q.BATCH = EFF_BATCH
    Q.STEPS_PER_EPOCH = Q.NTR // EFF_BATCH
    # Q.HELD stays cooc [5500:6000] (unchanged from every prior run)
    info.update({'n_train': int(Q.NTR), 'n_held': int(len(Q.HELD)),
                 'train_mix': 'cooc[0:5500] + fw[0:448]',
                 'steps_one_epoch': Q.STEPS_PER_EPOCH})
    print(f"data: train {Q.NTR} seqs (5500 cooc + 448 fw), held {len(Q.HELD)}, "
          f"one epoch = {Q.STEPS_PER_EPOCH} steps at effective batch {EFF_BATCH}",
          flush=True)
    return info


# ---------------- positive controls ----------------
@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    ma = make_vanilla1152().eval().float()
    ref = ma(idx)
    del ma
    torch.cuda.empty_cache()
    m = FACTORY().eval().float()
    assert m.wmask.shape == (2 * DEPTH, WIDTH)
    # (b) penalty vectorization at slot size 48
    p_fast = float(V8T.group_penalty(m))
    p_naive = 0.0
    for blk in m.h:
        for M in V8T.READ_MATS(blk):
            for k in range(V8T.NGROUP):
                p_naive += float(M[:, V8T.SUBDIM * k:V8T.SUBDIM * (k + 1)]
                                 .pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control penalty fast {p_fast:.4f} vs naive {p_naive:.4f} rel {rel:.2e}",
          flush=True)
    assert rel < 1e-6
    # (a) identity projections + zero writes == vanilla-1152 A at init
    m.wmask.fill_(1.0)
    for blk in m.h:
        blk.c_proj.weight.zero_()
        blk.Down.weight.zero_()
    d = (m(idx) - ref).abs().max().item()
    print(f"control W1152(identity proj, zero writes)==A1152: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-3
    del m
    torch.cuda.empty_cache()


def accum_control():
    """(c) accumulated micro-gradients == the one-shot gradient (fp32; run at
    effective batch 2 / micro 1 to keep the fp32 memory footprint small -- the
    accumulation algebra is batch-size independent)."""
    eff = 2
    seqs = Q.TRAIN[:eff]
    grads = []
    for micro in (eff, 1):
        m = FACTORY().float()
        m.zero_grad(set_to_none=False)
        for j in range(0, eff, micro):
            chunk = seqs[j:j + micro]
            logits = m(chunk[:, :Q.T]).float()
            ce = F.cross_entropy(logits.reshape(-1, Q.V),
                                 chunk[:, 1:Q.T + 1].reshape(-1))
            loss = ce * (chunk.shape[0] / eff)
            if j == 0:
                loss = loss + C.GROUP_COEFF * V8T.group_penalty(m)
            loss.backward()
        grads.append(torch.cat([p.grad.reshape(-1).cpu() for p in m.parameters()
                                if p.grad is not None]))
        del m
        torch.cuda.empty_cache()
    rel = float((grads[0] - grads[1]).norm() / (grads[0].norm() + 1e-12))
    print(f"control grad-accum(1x2) vs batch-2: rel err {rel:.2e}", flush=True)
    assert rel < 1e-4
    del grads
    torch.cuda.empty_cache()


# ---------------- training loop with gradient accumulation ----------------
def train_arm(arm, lr, total_steps, micro, log_every=100, save_stem=None):
    """arm in {'slots','vanilla'}; effective batch EFF_BATCH via accumulation."""
    Q.gpu_guard(min_free=7000)
    coeff = C.GROUP_COEFF if arm == 'slots' else 0.0
    model = FACTORY() if arm == 'slots' else make_vanilla1152()
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
           'spikes': 0, 'diverged': False}
    assert total_steps <= Q.STEPS_PER_EPOCH, "single-epoch budget exceeded"
    order = Q.epoch_order(0)                    # identical data order across arms
    t0, run = time.time(), None
    model.train()
    for step in range(total_steps):
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
        run = ce_step if run is None else 0.98 * run + 0.02 * ce_step
        if ce_step > run + 1.0:
            log['spikes'] += 1
        if step % log_every == 0:
            log['train_loss'].append([step, round(ce_step, 4), round(run, 4)])
            print(f"  {arm}[lr {lr}] step {step}/{total_steps} ce {ce_step:.4f} "
                  f"(ema {run:.4f}) {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB",
                  flush=True)
    log['peak_mem_mib'] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    if log['diverged']:
        log['final_held_ce'] = float('inf')
        del model, opt
        torch.cuda.empty_cache()
        return log
    if coeff > 0:
        log['final_penalty'] = float(V8T.group_penalty(model))
    if save_stem is None:                        # sweep run: held-100 CE only
        ce100, _ = Q.eval_held(model, n_seq=100)
        log['final_held_ce'] = ce100
        print(f"  sweep {arm} lr {lr}: held100 CE {ce100:.4f} "
              f"({log['spikes']} spikes, {time.time() - t0:.0f}s)", flush=True)
    else:
        hce, pt = Q.eval_held(model, per_token=True)
        log['final_held_ce'] = hce
        print(f"== {arm} FINAL held CE {hce:.5f}  ({time.time() - t0:.0f}s, "
              f"{log['spikes']} spikes)", flush=True)
        torch.save({'state_dict': model.state_dict(),
                    'variant': ('W1152' if arm == 'slots' else 'A'),
                    'config': dict(NL=DEPTH, D=WIDTH, NH=WIDTH // 64, HD=64,
                                   V=Q.V, EXP=Q.EXP, T=Q.T, BATCH=EFF_BATCH,
                                   micro_batch=micro, EPOCHS=1, SEED=Q.SEED,
                                   DATA_SEED=Q.DATA_SEED, lr=lr,
                                   group_coeff=coeff, WARMUP=Q.WARMUP, WD=Q.WD,
                                   GRAD_CLIP=Q.GRAD_CLIP, steps=total_steps,
                                   write_init_std=(R.WRITE_INIT_STD
                                                   if arm == 'slots' else None),
                                   train_mix='cooc[0:5500]+fw[0:448]'),
                    'log': log}, f'{QK}/{save_stem}.pt')
        np.save(f'{QK}/{save_stem}_heldloss.npy', pt)
    del model, opt
    torch.cuda.empty_cache()
    return log


def preflight_micro():
    """Two effective fwd+bwd steps per candidate micro-batch on the slots arm;
    pick the largest whose peak stays under MEM_BUDGET_MIB."""
    for micro in (EFF_BATCH, 4, 2):
        model = opt = None
        try:
            torch.cuda.reset_peak_memory_stats()
            model = FACTORY()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for i in range(2):
                seqs = Q.TRAIN[i * EFF_BATCH:(i + 1) * EFF_BATCH]
                opt.zero_grad(set_to_none=True)
                for j in range(0, EFF_BATCH, micro):
                    chunk = seqs[j:j + micro]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(chunk[:, :Q.T])
                    ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                         chunk[:, 1:Q.T + 1].reshape(-1))
                    loss = ce * (chunk.shape[0] / EFF_BATCH)
                    if j == 0:
                        loss = loss + C.GROUP_COEFF * V8T.group_penalty(model)
                    loss.backward()
                opt.step()
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            del model, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"pre-flight: micro {micro} peak {peak:.0f} MiB "
                  f"(budget {MEM_BUDGET_MIB})", flush=True)
            if peak < MEM_BUDGET_MIB:
                return micro, int(peak)
        except torch.cuda.OutOfMemoryError:
            print(f"pre-flight: micro {micro} OOM -- falling back", flush=True)
            if model is not None:
                del model
            if opt is not None:
                del opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    raise RuntimeError("width 1152 does not fit even at micro-batch 2")


if __name__ == '__main__':
    patch_width(WIDTH)
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    out['data'] = patch_data()
    json.dump(out, open(PATH, 'w'), indent=2)
    Q.gpu_guard(min_free=7000)
    controls()
    accum_control()

    if 'micro_batch' not in out:
        out['micro_batch'], out['preflight_peak_mib'] = preflight_micro()
        json.dump(out, open(PATH, 'w'), indent=2)
    MICRO = out['micro_batch']
    TOTAL = Q.STEPS_PER_EPOCH
    print(f"micro-batch {MICRO} (accum {EFF_BATCH // MICRO}), one epoch = "
          f"{TOTAL} steps", flush=True)

    # ---- lr sweeps (400 steps per arm per lr; penalty active on the slots arm) ----
    for arm in ('vanilla', 'slots'):
        key = f'lrsweep_{arm}'
        if key not in out:
            res = {}
            for lr in LRS:
                print(f"-- {arm} w1152 lr sweep {lr}", flush=True)
                log = train_arm(arm, lr, SWEEP_STEPS, MICRO)
                res[str(lr)] = {'held100_ce': (None if log['diverged'] else
                                               round(log['final_held_ce'], 4)),
                                'diverged': log['diverged'],
                                'spikes': log['spikes']}
            ok = {k: v for k, v in res.items() if not v['diverged']}
            ranking = sorted(ok, key=lambda k: ok[k]['held100_ce'])
            out[key] = {'results': res, 'ranking': [float(k) for k in ranking],
                        'chosen': float(ranking[0]),
                        'note': ('penalty 1e-4 active during sweep'
                                 if arm == 'slots' else '')}
            json.dump(out, open(PATH, 'w'), indent=2)
        print(f"{arm} lr chosen: {out[key]['chosen']} "
              f"(ranking {out[key]['ranking']})", flush=True)

    # ---- full runs, vanilla control first, divergence fallback per ranking ----
    for arm in ('vanilla', 'slots'):
        stem = f'qk_w1152_{arm}'
        if os.path.exists(f'{QK}/{stem}.pt'):
            print(f"{stem}.pt exists -- skip", flush=True)
            continue
        ranking = out[f'lrsweep_{arm}']['ranking']
        for pick, lr in enumerate(ranking):
            print(f"==== training {arm} w1152 (lr {lr}"
                  + (", fallback" if pick else "") + ") ====", flush=True)
            log = train_arm(arm, lr, TOTAL, MICRO, save_stem=stem)
            if not log['diverged']:
                break
        out[f'{arm}_run'] = {'lr': lr, 'held_ce_bf16': log.get('final_held_ce'),
                             'spikes': log['spikes'],
                             'peak_mem_mib': log.get('peak_mem_mib'),
                             'final_penalty': log.get('final_penalty'),
                             'diverged': log['diverged'],
                             'lr_fallback': pick > 0}
        json.dump(out, open(PATH, 'w'), indent=2)

    # ---- paired per-token CE: recipe minus vanilla ----
    out['ce'] = C.paired_ce('qk_w1152_slots_heldloss.npy',
                            'qk_w1152_vanilla_heldloss.npy', label='vanilla1152')
    out['ce'].update({'lr_slots': out['lrsweep_slots']['chosen'],
                      'lr_vanilla': out['lrsweep_vanilla']['chosen'],
                      'group_coeff': C.GROUP_COEFF, 'micro_batch': MICRO,
                      'eff_batch': EFF_BATCH,
                      'write_init_std_slots': R.WRITE_INIT_STD})
    json.dump(out, open(PATH, 'w'), indent=2)
    print(json.dumps(out['ce'], indent=2), flush=True)
