"""LEVEL-5-LITE naming pass on the BEST of V9 / V9b (lower held CE), per Logan's
queue: floors + best substitution + naming diagnostics. Machinery verbatim from
qk_v8_level5.py (source-level mean-ablation floor per module; token-conditional-mean
and linear-in-normed-embedding substitution at source with recovered fractions;
meaning diagnostics: PC logit-lens, top input tokens by write norm, extreme-
activation snippets). Anatomy rows come from the winner's probe json. The names
themselves are composed by the analyst pass and recorded under 'level5_names'.

Saves everything into the winner's json (qk_v9.json or qk_v9b.json) under 'level5'.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, time
import torch

import qk_tokenline_train as Q
import qk_v8_level5 as P5
import qk_v9_common as C
from qk_deeproute_train import DEPTH

QK = C.QK


def pick_winner():
    ce9 = json.load(open(f'{QK}/qk_v9.json'))['ce']['held_ce']
    ce9b = json.load(open(f'{QK}/qk_v9b.json'))['ce']['held_ce']
    if ce9 <= ce9b:
        return ('qk_v9', 'V9', lambda li: C.window_vis(li, N=6), ce9)
    return ('qk_v9b', 'V9b', lambda li: C.dil_vis(li, dil=C.DIL_LOOSE), ce9b)


def main():
    Q.gpu_guard(min_free=7000)
    stem, variant, vis_fn, ce = pick_winner()
    print(f"level-5-lite winner: {variant} (held CE {ce:.4f})", flush=True)
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model, ck = C.load_variant(stem, variant, vis_fn)
    path = f'{QK}/{stem}.json'
    out = json.load(open(path))
    probe = out['probe']
    t0 = time.time()

    base = P5.ce_src(model)
    print(f"base fp32 CE {base:.5f}", flush=True)
    tables, cnts = P5.build_tables(model)
    print(f"tables built ({time.time() - t0:.0f}s)", flush=True)

    lvl = {'winner': variant, 'base_ce': round(base, 5)}
    mods = {}
    for l in range(DEPTH):
        for kind in ('attn', 'mlp'):
            name = f'{kind}{l}'
            tabs = tables[kind]
            floor_tab = tabs['gmeans'][l][None].expand(Q.V, Q.D).contiguous()
            dfloor = P5.ce_src(model, kind, l, floor_tab) - base
            dtok = P5.ce_src(model, kind, l, tabs['tok'][l]) - base
            dlin = P5.ce_src(model, kind, l, tabs['lin'][l]) - base
            m = {'dce_floor': round(dfloor, 5),
                 'dce_tokmean': round(dtok, 5),
                 'dce_linear': round(dlin, 5),
                 'tokmean_recovered': (round(1 - dtok / dfloor, 4)
                                       if dfloor > 1e-4 else None),
                 'linear_recovered': (round(1 - dlin / dfloor, 4)
                                      if dfloor > 1e-4 else None)}
            si = str(1 + 2 * l) if kind == 'attn' else str(2 + 2 * l)
            cons = []
            for li in probe['consumption_matrix']:
                if si in probe['consumption_matrix'][li]:
                    cons.append({'consumer': ('readout' if int(li) == DEPTH
                                              else f'block{li}'),
                                 'dce': probe['consumption_matrix'][li][si],
                                 'support': probe['weight_support_matrix']
                                 .get(li, {}).get(si)})
            cons.sort(key=lambda c: -(c['dce'] or 0))
            m['consumers'] = cons
            m['reads'] = [P5.Q_stream_name(s) for s in model.vis[l]]
            mods[name] = m
            print(f"{name}: floor {dfloor:+.4f} tokmean rec "
                  f"{m['tokmean_recovered']} linear rec {m['linear_recovered']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    lvl['modules'] = mods

    print("meaning diagnostics ...", flush=True)
    diag = P5.meaning_diagnostics(model, tokenizer, cnts, tables)
    for name in mods:
        mods[name]['meaning'] = diag[name]

    out['level5'] = lvl
    json.dump(out, open(path, 'w'), indent=2)
    print(f"saved {stem}.json (level5) in {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
