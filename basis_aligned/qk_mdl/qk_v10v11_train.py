"""Train the width-264 small-model set (Logan 2026-08-03): matched vanilla
control, V10-READALL, V11-AFFINE (decoder lasso), V11nl (no decoder lasso),
V11lr (rank-32 factored decoders). Single lr 0.002 everywhere (Logan authorized
skipping the per-arm sweep for speed -- quick signal, not budget precision).
6-epoch budget, batch from pre-flight (8, fallback 4), group-lasso 1e-4 on
reads (+ decoder groups where flagged), nonzero write init on slotted arms.

Saves qk_v264_vanilla / qk_v10 / qk_v11 / qk_v11nl / qk_v11lr .pt + _heldloss.npy;
CE table paired vs the matched control -> qk_v10.json ('ce') and qk_v11.json
('ce_*'). Probes in qk_v10v11_probe.py. Idempotent: finished arms are skipped.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_v8_train as V8T
import qk_v9_common as C
import qk_v10v11_common as W

QK = W.QK

ARMS = [
    # (stem, json file, json key, factory, group coeff)
    ('qk_v264_vanilla', 'qk_v10.json', 'control_run', W.make_control, 0.0),
    ('qk_v10',   'qk_v10.json', 'run',       W.make_v10, W.GC),
    ('qk_v11',   'qk_v11.json', 'run_v11',
     lambda: W.make_v11('V11', dec_lasso=True, cls=W.V11Route), W.GC),
    ('qk_v11nl', 'qk_v11.json', 'run_v11nl',
     lambda: W.make_v11('V11nl', dec_lasso=False, cls=W.V11Route), W.GC),
    ('qk_v11lr', 'qk_v11.json', 'run_v11lr',
     lambda: W.make_v11('V11lr', dec_lasso=True, cls=W.V11LRRoute), W.GC),
]


def preflight_batch():
    """Two fwd+bwd steps on the largest arm (V11 full affine); prefer batch 8,
    fall back to 4 on OOM or if the peak crowds the shared GPU (> 5300 MiB)."""
    for b in (8, 4):
        model = opt = None
        try:
            W.patch_batch(b)
            model = W.make_v11('V11', dec_lasso=True, cls=W.V11Route)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            torch.cuda.reset_peak_memory_stats()
            for i in range(2):
                seqs = Q.TRAIN[i * b:(i + 1) * b]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(seqs[:, :Q.T])
                loss = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                       seqs[:, 1:Q.T + 1].reshape(-1))
                loss = loss + W.GC * V8T.group_penalty(model)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            del model, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"pre-flight: batch {b} peak {peak:.0f} MiB", flush=True)
            if peak > 5300 and b == 8:
                print("pre-flight: batch 8 too hungry for the shared GPU -- "
                      "falling back to 4", flush=True)
                continue
            return b
        except torch.cuda.OutOfMemoryError:
            print(f"pre-flight: batch {b} OOM -- falling back", flush=True)
            for x in (model, opt):
                if x is not None:
                    del x
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    raise RuntimeError("width 264 does not fit even at batch 4")


def merge(path, key, val):
    out = json.load(open(path)) if os.path.exists(path) else {}
    out[key] = val
    json.dump(out, open(path, 'w'), indent=2)


if __name__ == '__main__':
    W.patch_width()
    Q.gpu_guard(min_free=5200)
    W.run_controls()

    meta_path = f'{QK}/qk_v10.json'
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if 'batch' not in meta:
        meta['batch'] = preflight_batch()
        json.dump(meta, open(meta_path, 'w'), indent=2)
    W.patch_batch(meta['batch'])
    STEPS = V8T.STEPS
    print(f"width {W.WIDTH}, batch {Q.BATCH}, {Q.STEPS_PER_EPOCH} steps/epoch, "
          f"{STEPS} steps total, single lr {W.LR}", flush=True)

    # exact parameter accounting (reported in the final prose)
    counts = {}
    for stem, _, key, factory, _gc in ARMS:
        m = factory()
        counts[key] = W.param_counts(m)
        del m
        torch.cuda.empty_cache()
    merge(meta_path, 'param_counts', counts)
    print('param counts:', json.dumps(counts), flush=True)

    for stem, jpath, key, factory, gc in ARMS:
        if os.path.exists(f'{QK}/{stem}.pt'):
            print(f"{stem}.pt exists -- skip", flush=True)
            continue
        print(f"==== training {stem} (lr {W.LR}, gc {gc}) ====", flush=True)
        log = V8T.train_v8(W.LR, gc, STEPS, factory=factory, save_stem=stem)
        merge(f'{QK}/{jpath}', key, {
            'lr': W.LR, 'group_coeff': gc, 'width': W.WIDTH, 'batch': Q.BATCH,
            'steps': STEPS, 'held_ce_bf16': log.get('final_held_ce'),
            'spikes': log['spikes'], 'diverged': log['diverged'],
            'final_penalty': log.get('final_penalty'),
            'single_lr_note': 'lr sweep skipped per Logan (quick signal); '
                              '0.002 used for every arm including the control'})
        if log['diverged']:
            print(f"{stem} DIVERGED -- recorded, moving on", flush=True)

    # ---- paired CE vs the matched width-264 vanilla control ----
    ctl = 'qk_v264_vanilla_heldloss.npy'
    if os.path.exists(f'{QK}/{ctl}'):
        for stem, jpath, key in (('qk_v10', 'qk_v10.json', 'ce'),
                                 ('qk_v11', 'qk_v11.json', 'ce_v11'),
                                 ('qk_v11nl', 'qk_v11.json', 'ce_v11nl'),
                                 ('qk_v11lr', 'qk_v11.json', 'ce_v11lr')):
            if os.path.exists(f'{QK}/{stem}_heldloss.npy'):
                merge(f'{QK}/{jpath}', key,
                      C.paired_ce(f'{stem}_heldloss.npy', ctl, label='control264'))
        # arm-vs-arm contrasts
        for a, b, key in (('qk_v11', 'qk_v10', 'ce_v11_minus_v10'),
                          ('qk_v11nl', 'qk_v11', 'ce_v11nl_minus_v11'),
                          ('qk_v11lr', 'qk_v11', 'ce_v11lr_minus_v11')):
            fa, fb = f'{QK}/{a}_heldloss.npy', f'{QK}/{b}_heldloss.npy'
            if os.path.exists(fa) and os.path.exists(fb):
                merge(f'{QK}/qk_v11.json', key,
                      C.paired_ce(f'{a}_heldloss.npy', f'{b}_heldloss.npy',
                                  label=b))
        print('CE tables saved', flush=True)
    print('train script done', flush=True)
