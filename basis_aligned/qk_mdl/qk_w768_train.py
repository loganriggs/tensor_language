"""WIDTH-768 SCALE-UP (Logan's queue item 3): the free no-mask config of section 109
(subspace-partitioned write slots + group-lasso reads, FULL visibility -- the
unconditional recommendation) at width 768, depth 12, slot size 32 (24 slots x 32
dims = 768), vs a width-768 vanilla control. Does the free-ness and the 0.78 wiring
agreement survive doubling width?

Conventions: same corpus/split/seed/epochs (6 epochs, batch 8 -- pre-flight memory
test falls back to batch 4 for BOTH arms if width 768 does not fit; steps rescale to
keep the 6-epoch budget), group-lasso coefficient fixed 1e-4, nonzero write init,
brief lr re-sweep {0.001, 0.002} x 400 steps per arm (penalty active on the slots
arm). Width plumbing: Q/R/V8T/R2 module globals patched to D=768, NH=12, SUBDIM=32
for the whole script (Q.Block and the probe machinery read them dynamically).

Positive controls at width 768: (a) slots model with identity projections + zero
writes == width-768 variant-A MiniBilin at init; (b) group-penalty vectorization
equals the naive per-slice loop at slot size 32.

Saves qk_w768_slots.pt / qk_w768_slots_heldloss.npy, qk_w768_vanilla.pt /
qk_w768_vanilla_heldloss.npy; sweeps + CE -> qk_w768.json. Probe in qk_w768_probe.py.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_deeproute_train_2 as R2
import qk_window_train as WT
import qk_v9_common as C
from qk_deeproute_train import DEPTH

QK = C.QK
WIDTH = 768
LRS_768 = [0.001, 0.002]
EPOCHS = 6


def patch_width(w):
    Q.D, Q.NH, Q.HD = w, w // 64, 64
    sub = w // (2 * DEPTH)
    for mod in (R, V8T):
        mod.D, mod.NH, mod.HD, mod.SUBDIM = w, w // 64, 64, sub
    R2.D, R2.SUBDIM = w, sub
    print(f"width patched: D {w}, NH {w // 64}, slot dim {sub}", flush=True)


def patch_batch(b):
    Q.BATCH = b
    Q.STEPS_PER_EPOCH = Q.NTR // b
    V8T.STEPS = EPOCHS * Q.STEPS_PER_EPOCH


def make_vanilla768():
    old = Q.NL
    Q.NL = DEPTH
    m = Q.make_model('A')
    Q.NL = old
    return m


FACTORY = lambda: C.make_variant('W768', None)          # full visibility, no mask


# ---------------- positive controls at width 768 ----------------
@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    ma = make_vanilla768().eval().float()
    ref = ma(idx)
    del ma
    torch.cuda.empty_cache()
    m = FACTORY().eval().float()
    assert m.wmask.shape == (2 * DEPTH, WIDTH)
    # (b) penalty vectorization at slot size 32
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
    # (a) identity projections + zero writes == vanilla-768 A at init
    m.wmask.fill_(1.0)
    for blk in m.h:
        blk.c_proj.weight.zero_()
        blk.Down.weight.zero_()
    d = (m(idx) - ref).abs().max().item()
    print(f"control W768(identity proj, zero writes)==A768: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-3
    del m
    torch.cuda.empty_cache()


def preflight_batch():
    """Two fwd+bwd steps at batch 8; on OOM fall back to batch 4 for both arms."""
    for b in (8, 4):
        model = opt = None
        try:
            patch_batch(b)
            model = FACTORY()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for i in range(2):
                seqs = Q.TRAIN[i * b:(i + 1) * b]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(seqs[:, :Q.T])
                loss = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                       seqs[:, 1:Q.T + 1].reshape(-1))
                loss = loss + C.GROUP_COEFF * V8T.group_penalty(model)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            del model, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"pre-flight: batch {b} OK (peak {peak:.0f} MiB)", flush=True)
            return b
        except torch.cuda.OutOfMemoryError:
            print(f"pre-flight: batch {b} OOM -- falling back", flush=True)
            if model is not None:
                del model
            if opt is not None:
                del opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    raise RuntimeError("width 768 does not fit even at batch 4")


if __name__ == '__main__':
    patch_width(WIDTH)
    Q.gpu_guard(min_free=7000)
    controls()

    path = f'{QK}/qk_w768.json'
    out = json.load(open(path)) if os.path.exists(path) else {}

    if 'batch' not in out:
        out['batch'] = preflight_batch()
        json.dump(out, open(path, 'w'), indent=2)
    patch_batch(out['batch'])
    STEPS = V8T.STEPS
    print(f"batch {Q.BATCH}, {Q.STEPS_PER_EPOCH} steps/epoch, {STEPS} steps total",
          flush=True)

    # ---- lr sweeps ----
    if 'lrsweep_vanilla' not in out:
        out['lrsweep_vanilla'] = WT.sweep_arch('vanilla', LRS_768, width=WIDTH,
                                               batch=Q.BATCH)
        json.dump(out, open(path, 'w'), indent=2)
    LRV = out['lrsweep_vanilla']['chosen']
    print(f"vanilla768 lr chosen: {LRV}", flush=True)

    if 'lrsweep_slots' not in out:
        res = {}
        for lr in LRS_768:
            print(f"-- W768 slots lr sweep {lr} (gc {C.GROUP_COEFF})", flush=True)
            log = V8T.train_v8(lr, C.GROUP_COEFF, 400, log_every=100, save=False,
                               factory=FACTORY)
            res[str(lr)] = {'held100_ce': (None if log['diverged'] else
                                           round(log['final_held_ce'], 4)),
                            'diverged': log['diverged'], 'spikes': log['spikes']}
        ok = {k: v for k, v in res.items() if not v['diverged']}
        out['lrsweep_slots'] = {'results': res,
                                'chosen': float(min(ok, key=lambda k:
                                                    ok[k]['held100_ce'])),
                                'note': 'penalty 1e-4 active during sweep'}
        json.dump(out, open(path, 'w'), indent=2)
    LRS_CHOSEN = out['lrsweep_slots']['chosen']
    print(f"slots768 lr chosen: {LRS_CHOSEN}", flush=True)

    # ---- full runs (vanilla control first, then slots) ----
    if not os.path.exists(f'{QK}/qk_w768_vanilla.pt'):
        for pick, lr in enumerate([LRV] + [l for l in
                                           out['lrsweep_vanilla']['ranking']
                                           if l != LRV]):
            print(f"==== training vanilla768 (lr {lr}"
                  + (", fallback" if pick else "") + ") ====", flush=True)
            log = WT.train_model('vanilla', lr, STEPS, width=WIDTH, batch=Q.BATCH,
                                 name='qk_w768_vanilla')
            if not log['diverged']:
                break
        if os.path.exists(f'{QK}/qk_w768_vanilla.npy'):
            os.rename(f'{QK}/qk_w768_vanilla.npy',
                      f'{QK}/qk_w768_vanilla_heldloss.npy')
        out['vanilla_run'] = {'lr': lr, 'held_ce_bf16': log.get('final_held_ce'),
                              'spikes': log['spikes'],
                              'peak_mem_mib': log.get('peak_mem_mib'),
                              'diverged': log['diverged']}
        json.dump(out, open(path, 'w'), indent=2)
    else:
        print("qk_w768_vanilla.pt exists -- skip", flush=True)

    if not os.path.exists(f'{QK}/qk_w768_slots.pt'):
        print(f"==== training W768 slots (lr {LRS_CHOSEN}, gc {C.GROUP_COEFF}) ====",
              flush=True)
        log = V8T.train_v8(LRS_CHOSEN, C.GROUP_COEFF, STEPS, factory=FACTORY,
                           save_stem='qk_w768_slots')
        out['slots_run'] = {'lr': LRS_CHOSEN, 'group_coeff': C.GROUP_COEFF,
                            'held_ce_bf16': log['final_held_ce'],
                            'spikes': log['spikes'],
                            'final_penalty': log.get('final_penalty')}
        json.dump(out, open(path, 'w'), indent=2)
    else:
        print("qk_w768_slots.pt exists -- skip", flush=True)

    # ---- paired CE: slots vs the width-768 vanilla control ----
    out['ce'] = C.paired_ce('qk_w768_slots_heldloss.npy',
                            'qk_w768_vanilla_heldloss.npy', label='vanilla768')
    out['ce'].update({'lr_slots': LRS_CHOSEN, 'lr_vanilla': LRV,
                      'group_coeff': C.GROUP_COEFF, 'batch': out['batch']})
    json.dump(out, open(path, 'w'), indent=2)
    print(json.dumps(out['ce'], indent=2), flush=True)
