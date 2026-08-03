"""V14 train + probe (Logan update seven). Trains V14a (window + value-lerp)
and V14b (window + attention-only token line) at width 384 with the full
qk_window_train lr-sweep protocol ({0.0005, 0.001, 0.002, 0.003} x 400, winner
gets the 4122-step budget), pairs held CE against the existing plain-N6 and
vanilla width-384 controls, then runs the window probe (token-determined per
layer, washout, linear-in-embedding, per-layer token-table causal tests) and
the consumption ablations (V14a: carried-value lamb and content ablations,
per-block lamb table; V14b: attention-line embedding mean-ablation, all blocks
and per block). Everything -> qk_v14.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_v14_common as X
import qk_v8_train as V8T
import qk_window_train_2 as QW2
from qk_v14_common import DEPTH

QK = Q.QK
JP = f'{QK}/qk_v14.json'
LR_GRID = [0.0005, 0.001, 0.002, 0.003]
V, T = Q.V, Q.T


def merge(key, val):
    out = json.load(open(JP)) if os.path.exists(JP) else {}
    out[key] = val
    json.dump(out, open(JP, 'w'), indent=2)


def loadj():
    return json.load(open(JP)) if os.path.exists(JP) else {}


def paired(pt_a, pt_b):
    dd = pt_a - pt_b
    ds = dd.reshape(len(Q.HELD), T).mean(1)
    return {'delta': float(dd.mean()),
            'se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
            'se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}


# ---------------- probe (body of qk_window_train_2.probe with a live model) ----------------
@torch.no_grad()
def probe_v14(model, ck, arch):
    D = model.wte.weight.shape[1]
    r = {'lr': ck['config']['lr'],
         'held_ce_bf16_train_eval': ck['log']['final_held_ce'],
         'spikes': ck['log']['spikes']}
    gids = Q.HELD[:QW2.N_COLLECT, :T].reshape(-1).cpu().numpy().astype(np.int64)
    ent, mw, entry_norm, attn_norm = QW2.collect_held(model, arch, D)
    r['entry_stream_norm'] = [round(v, 2) for v in entry_norm]
    r['attn_write_norm'] = [round(v, 3) for v in attn_norm]
    r['mlp_write_norm'] = [round(float(mw[:, li].norm(dim=1).mean()), 3)
                           for li in range(DEPTH)]
    r['washout_entry_r2'] = QW2.r2_curve(ent, gids)
    r['token_determined_mlp'] = QW2.r2_curve(mw, gids)
    del ent
    gmeans, lin = QW2.build_linear_tables(model, arch, D)
    lin_cpu = lin.float().cpu()
    gid_t = torch.from_numpy(gids)
    ve = []
    for l in range(DEPTH):
        X64 = mw[:, l, :].double()
        pred = lin_cpu[l][gid_t].double()
        resid = float((X64 - pred).pow(2).sum())
        tot = float((X64 - X64.mean(0)).pow(2).sum())
        ve.append(round(1 - resid / tot, 4))
    r['linear_held_variance_explained'] = ve
    del mw, lin_cpu
    base, base_pt = QW2.held_ce_sub(model, arch)
    r['base_ce_fp32'] = round(base, 5)
    causal = []
    for l in range(DEPTH):
        floor_tab = gmeans[l][None].expand(V, D).contiguous()
        cef, _ = QW2.held_ce_sub(model, arch, floor_tab, l)
        cel, _ = QW2.held_ce_sub(model, arch, lin[l], l)
        dfloor, dlin = cef - base, cel - base
        rec = round(1 - dlin / dfloor, 4) if dfloor > 1e-6 else None
        causal.append({'dce_floor': round(dfloor, 5), 'dce_lin': round(dlin, 5),
                       'lin_recovered': rec})
        print(f"  L{l}: tokdet {r['token_determined_mlp'][l]['r2']:.3f} "
              f"linVE {ve[l]:.3f} dCEfloor {dfloor:.4f} rec {rec}", flush=True)
    r['causal_per_layer'] = causal
    del gmeans, lin
    torch.cuda.empty_cache()
    return r


@torch.no_grad()
def held_ce_kw(model, **kw):
    """fp32 held CE with forward kwargs (ablation hooks)."""
    tot, n = 0.0, 0
    for i in range(0, len(Q.HELD), 8):
        b = Q.HELD[i:i + 8]
        logits = model(b[:, :T], **kw)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        tot += ce.sum().item()
        n += ce.numel()
    return tot / n


@torch.no_grad()
def ablations_v14a(model, base):
    r = {'lamb_table': [[round(float(v), 4) for v in row]
                        for row in model.lamb.detach().cpu()]}
    r['dce_lamb_zero_all'] = round(held_ce_kw(model, lamb_zero=True) - base, 5)
    # mean-ablate the carried CONTENT (keep lamb mixing, v0 -> held mean)
    s, n = None, 0
    for i in range(0, 96, 8):
        b = Q.HELD[i:i + 8, :T]
        h = F.rms_norm(model.wte(b), (model.D,))
        v0 = model.h[0].c_v(F.rms_norm(h, (model.D,))).view(
            b.shape[0], T, model.NH, model.HD)
        s = v0.double().sum((0, 1)) if s is None else s + v0.double().sum((0, 1))
        n += b.shape[0] * T
    v0m = (s / n).float()[None, None]
    r['dce_v0_mean_sub'] = round(held_ce_kw(model, v0_sub=v0m) - base, 5)
    return r


@torch.no_grad()
def ablations_v14b(model, base):
    # held-mean embedding vector
    s, n = None, 0
    for i in range(0, 96, 8):
        b = Q.HELD[i:i + 8, :T]
        e = F.rms_norm(model.wte(b), (model.D,))
        s = e.double().sum((0, 1)) if s is None else s + e.double().sum((0, 1))
        n += b.shape[0] * T
    em = (s / n).float()[None, None, :]
    r = {}
    sub_all = {l: em for l in range(DEPTH)}
    r['dce_attline_mean_all_blocks'] = round(
        held_ce_kw(model, emb_att_sub=sub_all) - base, 5)
    per = {}
    for l in range(DEPTH):
        per[f'block{l}'] = round(
            held_ce_kw(model, emb_att_sub={l: em}) - base, 5)
    r['dce_attline_mean_per_block'] = per
    return r


if __name__ == '__main__':
    Q.gpu_guard(min_free=6500)
    X.v14_controls()
    print(f"V14: width 384, batch {Q.BATCH}, {V8T.STEPS} steps, "
          f"lr grid {LR_GRID} x 400 (window-experiment protocol)", flush=True)

    for which in ('a', 'b'):
        stem = f'qk_v14{which}'
        key = f'run_{which}'
        out = loadj()
        if f'lrsweep_{which}' not in out:
            res = {}
            for lr in LR_GRID:
                print(f"-- V14{which} sweep lr {lr}", flush=True)
                log = V8T.train_v8(lr, 0.0, 400, log_every=100, save=False,
                                   factory=lambda w=which: X.make_v14(w))
                res[str(lr)] = {'held100_ce': (None if log['diverged'] else
                                               round(log['final_held_ce'], 4)),
                                'diverged': log['diverged'],
                                'spikes': log['spikes']}
            ok = {k: v for k, v in res.items() if not v['diverged']}
            merge(f'lrsweep_{which}',
                  {'results': res,
                   'chosen': float(min(ok, key=lambda k: ok[k]['held100_ce']))})
        LR = loadj()[f'lrsweep_{which}']['chosen']
        print(f"V14{which} lr chosen: {LR}", flush=True)
        if not os.path.exists(f'{QK}/{stem}.pt'):
            print(f"==== training {stem} (lr {LR}) ====", flush=True)
            log = V8T.train_v8(LR, 0.0, V8T.STEPS,
                               factory=lambda w=which: X.make_v14(w),
                               save_stem=stem)
            merge(key, {'lr': LR, 'width': 384, 'steps': V8T.STEPS,
                        'held_ce_bf16': log.get('final_held_ce'),
                        'spikes': log['spikes'], 'diverged': log['diverged']})

    # ---- paired CE vs plain N6 and vanilla (existing width-384 controls) ----
    n6 = np.load(f'{QK}/qk_window_heldloss_N6.npy')
    van = np.load(f'{QK}/qk_window_heldloss_vanilla.npy')
    for which in ('a', 'b'):
        f = f'{QK}/qk_v14{which}_heldloss.npy'
        if os.path.exists(f):
            pt = np.load(f)
            merge(f'ce_{which}', {'held_ce': float(pt.mean()),
                                  'N6_held_ce': float(n6.mean()),
                                  'vanilla_held_ce': float(van.mean()),
                                  'vs_N6': paired(pt, n6),
                                  'vs_vanilla': paired(pt, van)})

    # ---- probes + consumption ablations ----
    for which in ('a', 'b'):
        if not os.path.exists(f'{QK}/qk_v14{which}.pt'):
            continue
        out = loadj()
        if f'probe_{which}' in out:
            print(f"probe_{which} exists -- skip", flush=True)
            continue
        model, ck = X.load_v14(which)
        print(f"==== probing V14{which} ====", flush=True)
        r = probe_v14(model, ck, f'V14{which}')
        merge(f'probe_{which}', r)
        base = r['base_ce_fp32']
        abl = (ablations_v14a if which == 'a' else ablations_v14b)(model, base)
        merge(f'ablations_{which}', abl)
        del model
        torch.cuda.empty_cache()
    print('v14 run done', flush=True)
