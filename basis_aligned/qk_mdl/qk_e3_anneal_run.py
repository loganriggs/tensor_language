"""E3 ANNEAL-TO-CERTIFIED-ZEROS (fresh single-epoch protocol).

Takes the trained fresh-stream V8-base control (E0b; trains it first if the
chain order was violated), and certifies its read sparsity:

 1. Collect all group-lasso read groups: for each of the 12 blocks, each of
    the 7 input matrices (c_q,c_k,c_q2,c_k2,c_v,Left,Right), each of the 24
    slot column-groups -> 2016 groups with Frobenius norms. (The readout reads
    through the tied embedding and is excluded, as in training.)
 2. Hard-zero every group below the MEDIAN group norm (threshold chosen so
    ~50 percent of groups zero; actual fraction reported).
 3. Fine-tune 1000 steps with the zeros FROZEN (gradient masked to zero on the
    zeroed columns and columns re-zeroed after every optimizer step), lr 2e-4
    constant after 50-step warmup, group-lasso 1e-4 kept, batch 4 on the fresh
    train stream reshuffled with epoch_order(1) (second visit for the ~4000
    sequences used -- noted; no unseen data remains after the single-epoch
    main runs).
 4. Report held CE before zeroing / after zeroing / after fine-tune (fresh
    held, plus old cooc held), paired deltas vs E0a/E0b, and the CERTIFIED
    SPARSE WIRING DIAGRAM: per (consumer block, source slot) edge, which of
    the 7 read matrices survive, with total surviving norm; an edge whose
    source module writes AFTER the consumer runs only carries embedding
    content in those dims and is flagged.

Positive control: applying the masking machinery with an empty zero-set leaves
every weight bit-identical (max |delta| == 0). Results -> qk_e3.json,
checkpoint qk_e3_anneal.pt (+ fresh/old heldloss npys). Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, R2, DEPTH, F, torch

JP = E.jpath('qk_e3.json')
STEM = 'qk_e3_anneal'
FT_STEPS = 3 if E.SMOKE else 1000
FT_LR = 2e-4
FT_WARMUP = 1 if E.SMOKE else 50


def group_norms(model):
    """(block, matrix, slot) -> Frobenius norm of the read-column group."""
    g = torch.zeros(DEPTH, len(E.READ_NAMES), E.NGROUP)
    for li, blk in enumerate(model.h):
        for mi, nm in enumerate(E.READ_NAMES):
            M = getattr(blk, nm).weight.detach().float()
            g[li, mi] = M.pow(2).view(M.shape[0], E.NGROUP, E.SUB) \
                         .sum(dim=(0, 2)).sqrt()
    return g


def build_masks(g, thresh):
    """Boolean (block, matrix, slot) zero-mask: True = group is zeroed."""
    return g < thresh


@torch.no_grad()
def apply_masks(model, zmask):
    for li, blk in enumerate(model.h):
        for mi, nm in enumerate(E.READ_NAMES):
            M = getattr(blk, nm).weight
            for k in range(E.NGROUP):
                if zmask[li, mi, k]:
                    M[:, E.SUB * k:E.SUB * (k + 1)] = 0.0


def mask_grads(model, zmask):
    for li, blk in enumerate(model.h):
        for mi, nm in enumerate(E.READ_NAMES):
            gr = getattr(blk, nm).weight.grad
            if gr is None:
                continue
            for k in range(E.NGROUP):
                if zmask[li, mi, k]:
                    gr[:, E.SUB * k:E.SUB * (k + 1)] = 0.0


def surviving_edges(model, zmask):
    """Certified wiring diagram: per (consumer block, source slot) edge the
    surviving read matrices and their total norm."""
    g = group_norms(model)
    edges = []
    for li in range(DEPTH):
        for k in range(E.NGROUP):
            alive = [E.READ_NAMES[mi] for mi in range(len(E.READ_NAMES))
                     if not zmask[li, mi, k]]
            if not alive:
                continue
            tot = float(sum(g[li, mi, k] for mi in range(len(E.READ_NAMES))
                            if not zmask[li, mi, k]))
            edges.append({'consumer': f'block{li}',
                          'source_slot': R2.stream_name(k + 1),
                          'surviving_mats': alive,
                          'total_norm': round(tot, 4),
                          'source_written_before_consumer': bool(k < 2 * li)})
    edges.sort(key=lambda e: -e['total_norm'])
    return edges


def finetune(model, zmask):
    decay, nodecay = [], []
    for nm, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': Q.WD},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=FT_LR, betas=(0.9, 0.95))
    order = Q.epoch_order(1)                     # fresh-train reshuffle
    log = {'lr': FT_LR, 'steps': FT_STEPS, 'train_loss': [], 'held_ce': [],
           'spikes': 0}
    model.train()
    run, t0 = None, time.time()
    for step in range(FT_STEPS):
        for gpg in opt.param_groups:
            gpg['lr'] = FT_LR * min(1.0, (step + 1) / FT_WARMUP)
        seqs = Q.TRAIN[order[step * Q.BATCH:(step + 1) * Q.BATCH]]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(seqs[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             seqs[:, 1:Q.T + 1].reshape(-1))
        loss = ce + E.GC * V8T.group_penalty(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(model, zmask)                 # frozen zeros: no gradient in
        torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
        opt.step()
        apply_masks(model, zmask)                # belt and braces: re-zero
        l = ce.item()
        run = l if run is None else 0.98 * run + 0.02 * l
        if run is not None and l > run + 1.0:
            log['spikes'] += 1
        if step % (1 if E.SMOKE else 100) == 0:
            log['train_loss'].append([step, round(l, 4), round(run, 4)])
            print(f"  E3 ft step {step}/{FT_STEPS} ce {l:.4f} (ema {run:.4f}) "
                  f"{time.time() - t0:.0f}s", flush=True)
        if not E.SMOKE and step > 0 and step % 250 == 0:
            hce, _ = Q.eval_held(model, n_seq=100)
            log['held_ce'].append([step, round(hce, 4)])
            print(f"  E3 ft step {step} held100 fresh CE {hce:.4f}", flush=True)
    model.eval()
    del opt
    torch.cuda.empty_cache()
    return log


if __name__ == '__main__':
    E.setup()
    out = E.loadj(JP)
    if 'E3' in out and (E.SMOKE or os.path.exists(E.ckpath(STEM))):
        print("E3 already done -- only refreshing pairings", flush=True)
    else:
        # ---- get the trained E0b base (train it if the chain order broke) ----
        if E.SMOKE:
            m = E.make_e0b()
            torch.save({'state_dict': m.state_dict(), 'variant': 'E0b',
                        'config': {}, 'log': {}}, E.ckpath(E.E0B_STEM))
            del m
        elif not os.path.exists(E.ckpath(E.E0B_STEM)):
            print("E0b checkpoint missing -- training it first", flush=True)
            E.train_arm(E.E0B_STEM, E.jpath('qk_e0.json'), 'E0b',
                        E.make_e0b, E.GC)
        model, _ = E.load_arm(E.E0B_STEM, E.make_e0b)

        # ---- positive control: empty zero-set changes nothing ----
        snap = [getattr(b, nm).weight.detach().clone()
                for b in model.h for nm in E.READ_NAMES]
        empty = torch.zeros(DEPTH, len(E.READ_NAMES), E.NGROUP,
                            dtype=torch.bool)
        apply_masks(model, empty)
        dmax = max(float((getattr(b, nm).weight - s).abs().max())
                   for s, (b, nm) in zip(snap, [(b, nm) for b in model.h
                                                for nm in E.READ_NAMES]))
        print(f"control empty-mask leaves weights unchanged: max delta "
              f"{dmax:.2e}", flush=True)
        assert dmax == 0.0
        del snap

        # ---- threshold at the median group norm (~50 percent zeroed) ----
        g = group_norms(model)
        flat = g.flatten()
        thresh = float(flat.median())
        zmask = build_masks(g, thresh)
        frac = float(zmask.float().mean())
        n_batch = 2 if E.SMOKE else 100
        ce_before, _ = E.eval_ce(model, Q.HELD, n_seq=n_batch)
        apply_masks(model, zmask)
        ce_zeroed, _ = E.eval_ce(model, Q.HELD, n_seq=n_batch)
        print(f"E3: {int(zmask.sum())}/{zmask.numel()} groups zeroed "
              f"(frac {frac:.4f}, thresh {thresh:.5f}); held{n_batch} fresh CE "
              f"{ce_before:.4f} -> {ce_zeroed:.4f}", flush=True)
        if not E.SMOKE:
            _, pt0 = E.eval_ce(model, Q.HELD, per_token=True)
            np.save(f'{E.QK}/{STEM}_zeroed_heldloss.npy', pt0)

        # ---- fine-tune with frozen zeros ----
        ftlog = finetune(model, zmask)
        ce_ft, ptf = E.eval_ce(model, Q.HELD, per_token=True)
        # certify: zeroed groups are still exactly zero
        g_after = group_norms(model)
        leak = float(g_after[zmask].abs().max())
        print(f"E3 after ft: held fresh CE {ce_ft:.4f}; max zeroed-group norm "
              f"{leak:.2e}", flush=True)
        assert leak == 0.0

        rec = {'base': 'E0b (fresh single-epoch V8 base)',
               'n_groups': int(zmask.numel()),
               'n_zeroed': int(zmask.sum()),
               'zero_fraction': round(frac, 4),
               'threshold': thresh,
               'threshold_rule': 'median group norm (~50 percent zeroed)',
               f'held{n_batch}_fresh_ce_before_zero': round(ce_before, 5),
               f'held{n_batch}_fresh_ce_after_zero': round(ce_zeroed, 5),
               'held_fresh_ce_after_finetune_bf16': round(ce_ft, 5),
               'finetune': {'lr': FT_LR, 'steps': FT_STEPS,
                            'warmup': FT_WARMUP, 'group_coeff': E.GC,
                            'data': 'fresh train reshuffled epoch_order(1), '
                                    'second visit',
                            'train_curve': ftlog['train_loss'],
                            'held100_fresh_curve': ftlog['held_ce'],
                            'spikes': ftlog['spikes']},
               'max_zeroed_group_norm_after_ft': leak}
        E.merge(JP, 'E3', rec)
        edges = surviving_edges(model, zmask)
        E.merge(JP, 'certified_surviving_edges', {
            'n_edges_possible': DEPTH * E.NGROUP,
            'n_edges_surviving': len(edges),
            'edges': edges})
        if not E.SMOKE:
            np.save(f'{E.QK}/{STEM}_heldloss.npy', ptf)
            torch.save({'state_dict': model.state_dict(), 'variant': 'E0b',
                        'zmask': zmask, 'threshold': thresh,
                        'config': {'finetune_lr': FT_LR,
                                   'finetune_steps': FT_STEPS},
                        'log': ftlog}, E.ckpath(STEM))
        del model
        torch.cuda.empty_cache()

    if not E.SMOKE and os.path.exists(E.ckpath(STEM)):
        E.oldheld_record(STEM, E.make_e0b, JP, 'E3_oldheld')
        E.paired_fresh(STEM, JP, 'E3')
        if os.path.exists(f'{E.QK}/{STEM}_zeroed_heldloss.npy') \
                and os.path.exists(f'{E.QK}/{E.E0B_STEM}_heldloss.npy'):
            E.merge(JP, 'E3_zeroed_minus_e0b_fresh',
                    E.paired(f'{STEM}_zeroed_heldloss.npy',
                             f'{E.E0B_STEM}_heldloss.npy',
                             len(Q.HELD), 'e0b'))
    print('e3 anneal run done', flush=True)
