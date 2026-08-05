"""E3 CERTIFIED-ZERO ANNEAL at width 1152 on the COMBO checkpoint (scale
session, standing-directive item 4).

Ports qk_e3_anneal_run.py to the scale protocol, annealing the best width-1152
model (qk_s_w1152_combo.pt: slots + per-slot RMSNorm + proximal 1e-4 + Muon):

 1. Frobenius norms of all 2016 read groups (12 blocks x 7 read matrices x
    24 slot groups of 48 columns; tied-embedding readout excluded).
 2. Hard-zero every group below the MEDIAN norm (~50% zeroed; actual
    fraction reported -- proximal training may have exact zeros already).
 3. Fine-tune FT_STEPS with the zeros FROZEN (grad-masked + re-zeroed after
    every step), AdamW lr 2e-4 (50-step warmup then constant), penalty kept
    PROXIMAL at 1e-4 (tau = lr * warmup_factor * 1e-4, matching how the
    combo was trained; the small-scale E3 kept an in-loss lasso instead --
    convention difference recorded).
 4. FT data: fresh34k rows [6000:22000] (docs inside 20001..45366) --
    genuinely UNSEEN by every scale model, disjoint from the substitute-cooc
    probe rows [0:6000], the small-scale held rows [33000:34500], and the
    scale held set (shard06). Single visit, batch 16 x 1000 steps. This is
    cleaner than the small-scale E3, which had to revisit training data.

Positive control before annealing: applying the masking machinery with an
empty zero-set leaves every weight bit-identical AND the combo checkpoint
reproduces its recorded held CE.

Reports CE before / after zeroing / after fine-tune on BOTH held sets,
per-token npys after FT, paired vs combo. Outputs qk_s_w1152_e3anneal.*.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import qk_s_gate_run as G
import qk_s_e1_run as E1R
import qk_tokenline_train as Q
import qk_v8_train as V8T
import qk_w1152_train as W2
from qk_deeproute_train import DEPTH
from qk_s_muon_run import prox_group_lasso, READ_NAMES, NGROUP

STEM = 'qk_s_w1152_e3anneal'
JP = os.path.join(G.OUT_DIR, f'{STEM}.json')
SRC = 'qk_s_w1152_combo'
FT_STEPS = 1000
FT_BATCH = 16
FT_LR = 2e-4
FT_WARMUP = 50
PROX = 1e-4
SUB = 48


def group_norms(model):
    g = torch.zeros(DEPTH, len(READ_NAMES), NGROUP)
    for li, blk in enumerate(model.h):
        for mi, nm in enumerate(READ_NAMES):
            M = getattr(blk, nm).weight.detach().float()
            g[li, mi] = M.pow(2).view(M.shape[0], NGROUP, SUB) \
                         .sum(dim=(0, 2)).sqrt()
    return g


@torch.no_grad()
def apply_masks(model, zmask):
    for li, blk in enumerate(model.h):
        for mi, nm in enumerate(READ_NAMES):
            M = getattr(blk, nm).weight
            for k in range(NGROUP):
                if zmask[li, mi, k]:
                    M[:, SUB * k:SUB * (k + 1)] = 0.0


def mask_grads(model, zmask):
    for li, blk in enumerate(model.h):
        for mi, nm in enumerate(READ_NAMES):
            gr = getattr(blk, nm).weight.grad
            if gr is None:
                continue
            for k in range(NGROUP):
                if zmask[li, mi, k]:
                    gr[:, SUB * k:SUB * (k + 1)] = 0.0


def main():
    out = G.loadj(JP)
    W2.patch_width(G.WIDTH)
    _, spec, f34k_held = G.setup_data()
    out['env'] = {'gpu': torch.cuda.get_device_name(0),
                  'torch': torch.__version__, 'source': SRC}
    G.savej(JP, out)

    ck = torch.load(os.path.join(G.OUT_DIR, f'{SRC}.pt'),
                    map_location='cuda', weights_only=False)
    model = E1R.make_e1()
    model.load_state_dict(ck['state_dict'])
    model.float()

    # positive controls: empty-mask no-op + checkpoint CE reproduction
    if not out.get('controls_ok'):
        before = {nm: getattr(model.h[0], nm).weight.detach().clone()
                  for nm in READ_NAMES}
        empty = torch.zeros(DEPTH, len(READ_NAMES), NGROUP, dtype=torch.bool)
        apply_masks(model, empty)
        d = max(float((getattr(model.h[0], nm).weight - before[nm])
                      .abs().max()) for nm in READ_NAMES)
        assert d == 0.0, f"empty mask changed weights: {d}"
        ce0, _ = G.eval_data(model, Q.HELD)
        rec = ck['log'].get('final_held_ce')
        print(f"control: empty mask no-op OK; checkpoint CE {ce0:.5f} vs "
              f"recorded {rec:.5f}", flush=True)
        assert abs(ce0 - rec) < 2e-3
        out['controls_ok'] = True
        out['ce_before'] = {'scale_held': round(ce0, 5)}
        G.savej(JP, out)

    g = group_norms(model)
    thresh = float(g.median())
    zmask = g < thresh
    frac = float(zmask.float().mean())
    already_zero = float((g < 1e-8).float().mean())
    out['zeroing'] = {'median_norm': round(thresh, 6),
                      'frac_zeroed': round(frac, 4),
                      'frac_exactly_zero_pre': round(already_zero, 4),
                      'n_groups': int(g.numel())}
    apply_masks(model, zmask)
    ce_z, _ = G.eval_data(model, Q.HELD)
    ce_zf, _ = G.eval_data(model, f34k_held)
    out['ce_after_zeroing'] = {'scale_held': round(ce_z, 5),
                               'f34k': round(ce_zf, 5)}
    G.savej(JP, out)
    print(f"zeroed {frac:.1%} of {g.numel()} groups (median {thresh:.4f}); "
          f"CE scale {ce_z:.5f} f34k {ce_zf:.5f}", flush=True)

    # fine-tune on genuinely unseen fresh34k rows [6000:22000]
    ft = np.load(f'{G.QK}/corpus_fresh/fresh34k.npy',
                 mmap_mode='r')[6000:6000 + FT_STEPS * FT_BATCH]
    ft = torch.from_numpy(np.asarray(ft).astype(np.int64)).to('cuda')
    decay, nodecay = [], []
    for nm, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': Q.WD},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=FT_LR, betas=(0.9, 0.95))
    log = {'train_loss': [], 'spikes': 0}
    t0, run = time.time(), None
    model.train()
    for step in range(FT_STEPS):
        f = min(1.0, (step + 1) / FT_WARMUP)
        for gpg in opt.param_groups:
            gpg['lr'] = FT_LR * f
        seqs = ft[step * FT_BATCH:(step + 1) * FT_BATCH]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(seqs[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             seqs[:, 1:Q.T + 1].reshape(-1))
        l = ce.item()
        assert math.isfinite(l) and l < 30, f"FT diverged at {step}: {l}"
        opt.zero_grad(set_to_none=True)
        ce.backward()
        mask_grads(model, zmask)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
        opt.step()
        prox_group_lasso(model, FT_LR * f * PROX)
        apply_masks(model, zmask)            # re-certify zeros every step
        run = l if run is None else 0.98 * run + 0.02 * l
        if step % 100 == 0:
            log['train_loss'].append([step, round(l, 4), round(run, 4)])
            print(f"  FT step {step}/{FT_STEPS} ce {l:.4f} (ema {run:.4f}) "
                  f"{time.time() - t0:.0f}s", flush=True)

    # verify certification survived, then final evals
    g2 = group_norms(model)
    max_zeroed_norm = float(g2[zmask].max())
    assert max_zeroed_norm == 0.0, f"zeroed group revived: {max_zeroed_norm}"
    model.float()
    hce, pt = G.eval_data(model, Q.HELD, per_token=True)
    fce, fpt = G.eval_data(model, f34k_held, per_token=True)
    out['ft'] = {'steps': FT_STEPS, 'batch': FT_BATCH, 'lr': FT_LR,
                 'prox_coeff': PROX,
                 'data': f'fresh34k[6000:{6000 + FT_STEPS * FT_BATCH}] '
                         '(unseen, single visit)',
                 'train_curve': log['train_loss']}
    out['ce_after_ft'] = {'scale_held': round(hce, 5), 'f34k': round(fce, 5),
                          'max_zeroed_group_norm': max_zeroed_norm}
    # paired deltas vs the un-annealed combo
    for arr, suf, label in ((pt, 'heldloss', 'scale_held'),
                            (fpt, 'f34kloss', 'f34k')):
        base = np.load(f'{G.QK}/{SRC}_{suf}.npy')
        d = arr - base
        ds = d.reshape(-1, Q.T).mean(1)
        out[f'anneal_minus_combo_{label}'] = {
            'delta': round(float(d.mean()), 5),
            'se_seq': round(float(ds.std(ddof=1) / math.sqrt(len(ds))), 6)}
    G.savej(JP, out)
    torch.save({'state_dict': model.state_dict(), 'zmask': zmask,
                'config': dict(ck['config'], anneal_ft_steps=FT_STEPS,
                               anneal_ft_lr=FT_LR, anneal_prox=PROX),
                'log': log}, os.path.join(G.OUT_DIR, f'{STEM}.pt'))
    np.save(os.path.join(G.OUT_DIR, f'{STEM}_heldloss.npy'), pt)
    np.save(os.path.join(G.OUT_DIR, f'{STEM}_f34kloss.npy'), fpt)
    print(json.dumps({k: out[k] for k in
                      ('zeroing', 'ce_before', 'ce_after_zeroing',
                       'ce_after_ft')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
