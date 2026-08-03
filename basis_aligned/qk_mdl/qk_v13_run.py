"""V13 train + probe (Logan update six). Trains V13 (rank-4 edge adapters) and
V13r1 (rank-1 sub-variant) at width 264, single lr 0.002, 6-epoch budget,
paired vs the qk_v264_vanilla control from the V10/V11 set; then the full
interpretability block per arm (V13-aware exact terms at blocks 4/6/9), the
adapter-magnitude wiring test, per-edge survival census, and edge-semantics
naming for the 5 largest surviving edges (pattern V_c contexts -> effect U_c
through the consumer: read-matrix fold shares for block consumers, token lens
for readout edges). Everything -> qk_v13.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import numpy as np
import torch

import qk_tokenline_train as Q
import qk_v8_train as V8T
import qk_deeproute_train_2 as R2
import qk_v9_common as C
import qk_v10v11_common as W
import qk_v13_common as X
from qk_deeproute_train import DEPTH

QK = Q.QK
JP = f'{QK}/qk_v13.json'
_ORIG_TD = R2.term_decomposition
_ORIG_WW = R2.weight_wiring


def merge(key, val):
    out = json.load(open(JP)) if os.path.exists(JP) else {}
    out[key] = val
    json.dump(out, open(JP, 'w'), indent=2)


def loadj():
    return json.load(open(JP)) if os.path.exists(JP) else {}


# ---------------- adapter analyses ----------------
def edge_census(model):
    rows = []
    for (li, si) in model.edges:
        rows.append({'consumer': ('readout' if li == DEPTH else f'block{li}'),
                     'source': R2.stream_name(si),
                     'product_norm': round(model.edge_product_norm(li, si), 4)})
    rows.sort(key=lambda r: -r['product_norm'])
    return rows


def adapter_wiring(model, probe_key):
    """Does adapter magnitude alone predict causal consumption over edges?"""
    out = loadj()
    dce = out[probe_key]['consumption_matrix']
    pairs = [(li, si) for (li, si) in model.edges]
    cau = [dce[str(li)][str(si)] for li, si in pairs]
    sup = [model.edge_product_norm(li, si) for li, si in pairs]
    eff = [k for k in range(len(pairs)) if cau[k] > C.EFFECTUAL]
    top_sup = set(np.argsort(sup)[::-1][:10].tolist())
    top_cau = set(np.argsort(cau)[::-1][:10].tolist())
    return {'n_edges': len(pairs),
            'spearman_all': round(R2.spearman(sup, cau), 4),
            'spearman_effectual': round(R2.spearman(
                [sup[k] for k in eff], [cau[k] for k in eff]), 4),
            'n_effectual': len(eff),
            'top10_precision': len(top_sup & top_cau) / 10.0}


READ_NAMES = ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v', 'Left', 'Right')


@torch.no_grad()
def edge_semantics(model, tokenizer, top_n=5):
    """Name the r channels of the largest surviving edges via the composite
    (consumer read weights) o A_e o (source write)."""
    U_emb = model.wte.weight.detach().float()
    ranked = sorted(model.edges,
                    key=lambda e: -model.edge_product_norm(*e))[:top_n]
    need = sorted({si - 1 for (_, si) in ranked})
    from qk_v10v11_probe import collect_module_writes, extreme_positions
    wdict = collect_module_writes(model, need)
    ids = Q.HELD[:96, :Q.T].cpu()
    out = {}
    for (li, si) in ranked:
        j = model.edge_index[(li, si)]
        Ue = model.ad_U[j].detach().float()          # (D, r) effect directions
        Ve = model.ad_V[j].detach().float()          # (D, r) source patterns
        Xw = wdict[si - 1]
        norms = Xw.norm(dim=-1)
        name = (f"{R2.stream_name(si)} -> "
                + ('readout' if li == DEPTH else f'block{li}'))
        p200 = torch.tensor(extreme_positions(norms, 200, min_gap=1))
        acts = Xw[p200].to(W.DEV) @ Ve.to(W.DEV)     # (200, r)
        imp = acts.abs().mean(0) * Ue.norm(dim=0).to(W.DEV)
        chans = []
        for c in torch.argsort(imp, descending=True).tolist():
            a_all = (Xw.to(W.DEV) @ Ve[:, c].to(W.DEV)).cpu()
            ctxs = [{'act': round(float(a_all[p]), 3),
                     'context': tokenizer.decode(
                         ids[p // Q.T, max(0, p % Q.T - 10):p % Q.T + 1].tolist())}
                    for p in extreme_positions(a_all.abs(), 3)]
            ch = {'channel': c, 'importance': round(float(imp[c]), 4),
                  'pattern_contexts': ctxs}
            uc = Ue[:, c].to(W.DEV)
            if li == DEPTH:                          # readout edge: token lens
                lg = U_emb @ uc
                ch['effect_top_tokens'] = [tokenizer.decode([t]) for t in
                                           torch.topk(lg, 8).indices.tolist()]
                ch['effect_bottom_tokens'] = [tokenizer.decode([t]) for t in
                                              torch.topk(-lg, 4).indices.tolist()]
            else:                                    # block edge: read-matrix fold
                blk = model.h[li]
                mats = [blk.c_q.weight, blk.c_k.weight, blk.c_q2.weight,
                        blk.c_k2.weight, blk.c_v.weight, blk.Left.weight,
                        blk.Right.weight]
                folds = [float((M.detach().float() @ uc).norm()) for M in mats]
                tot = sum(folds) + 1e-12
                ch['consumer_fold_shares'] = {
                    n: round(f / tot, 3) for n, f in zip(READ_NAMES, folds)}
                lg = U_emb @ uc                      # weak lens, context only
                ch['weak_token_lens'] = [tokenizer.decode([t]) for t in
                                         torch.topk(lg, 5).indices.tolist()]
            chans.append(ch)
        out[name] = {'product_norm': round(model.edge_product_norm(li, si), 3),
                     'channels': chans}
        print(f"  edge {name}: product {out[name]['product_norm']}, channels "
              f"{[c['channel'] for c in chans]}", flush=True)
    del wdict
    torch.cuda.empty_cache()
    return out


def probe_arm(stem, variant, rank, key):
    done = loadj()
    if key in done and 'probes' in done.get(key, {}):
        print(f"{key}: already probed -- skip", flush=True)
    else:
        model, ck = X.load_v13(stem, variant, rank)
        R2.term_decomposition = lambda m, v, b: X.term_decomposition_v13(m, b)
        try:
            C.full_probe(model, ck, save_cb=C.json_saver(JP, key))
        finally:
            R2.term_decomposition = _ORIG_TD
        del model
        torch.cuda.empty_cache()
    # adapter analyses (recomputed cheaply every run)
    model, ck = X.load_v13(stem, variant, rank)
    merge(f'edge_census_{key}', edge_census(model)[:30])
    merge(f'adapter_wiring_{key}', adapter_wiring(model, key))
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    merge(f'edge_semantics_{key}', edge_semantics(model, tokenizer))
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    W.patch_width()
    Q.gpu_guard(min_free=4500)
    X.v13_controls()

    meta = json.load(open(f'{QK}/qk_v10.json'))
    W.patch_batch(meta['batch'])
    STEPS = V8T.STEPS
    print(f"V13: width {W.WIDTH}, batch {Q.BATCH}, {STEPS} steps, "
          f"single lr {W.LR}", flush=True)

    counts = {}
    for variant, rank in (('V13', 4), ('V13r1', 1)):
        m = X.make_v13(variant, rank)
        pc = W.param_counts(m)
        pc['adapter_params'] = sum(p.numel() for p in m.ad_U) \
            + sum(p.numel() for p in m.ad_V)
        counts[variant] = pc
        del m
        torch.cuda.empty_cache()
    merge('param_counts', counts)
    print('param counts:', json.dumps(counts), flush=True)

    for stem, variant, rank, key in (('qk_v13', 'V13', 4, 'run_r4'),
                                     ('qk_v13r1', 'V13r1', 1, 'run_r1')):
        if os.path.exists(f'{QK}/{stem}.pt'):
            print(f"{stem}.pt exists -- skip", flush=True)
            continue
        print(f"==== training {stem} (lr {W.LR}, gc {W.GC}) ====", flush=True)
        log = V8T.train_v8(W.LR, W.GC, STEPS,
                           factory=(lambda v=variant, r=rank: X.make_v13(v, r)),
                           save_stem=stem)
        merge(key, {'lr': W.LR, 'group_coeff': W.GC, 'width': W.WIDTH,
                    'batch': Q.BATCH, 'steps': STEPS, 'rank': rank,
                    'held_ce_bf16': log.get('final_held_ce'),
                    'spikes': log['spikes'], 'diverged': log['diverged'],
                    'final_penalty': log.get('final_penalty')})

    ctl = 'qk_v264_vanilla_heldloss.npy'
    for stem, key in (('qk_v13', 'ce_r4'), ('qk_v13r1', 'ce_r1')):
        if os.path.exists(f'{QK}/{stem}_heldloss.npy'):
            merge(key, C.paired_ce(f'{stem}_heldloss.npy', ctl,
                                   label='control264'))
    for a, b, key in (('qk_v13', 'qk_v10', 'ce_r4_minus_v10'),
                      ('qk_v13', 'qk_v13r1', 'ce_r4_minus_r1')):
        if os.path.exists(f'{QK}/{a}_heldloss.npy') \
                and os.path.exists(f'{QK}/{b}_heldloss.npy'):
            merge(key, C.paired_ce(f'{a}_heldloss.npy', f'{b}_heldloss.npy',
                                   label=b))

    probe_arm('qk_v13', 'V13', 4, 'probe_r4')
    probe_arm('qk_v13r1', 'V13r1', 1, 'probe_r1')
    print('v13 run done', flush=True)
