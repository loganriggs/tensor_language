"""E13 LEVEL-5 INTERPRETATION PASS on the fresh-recipe model (Logan
2026-08-05). The module-naming work V8/V9 got under the old multi-epoch
protocol (qk_v8_level5.py is the template verbatim where possible) has never
been done on a fresh-data recipe model: this runs it on qk_e9_a.pt (per-slot
norm + Muon + in-loss lasso 3e-5, single fresh epoch).

Per module (12 blocks x {attention, mlp}):
 1. HARVEST: top-activating held contexts (extreme write-norm snippets),
    top input tokens by conditional-mean write norm, PC-logit-lens.
 2. HYPOTHESES (the V8 substitution family + one addition): token-conditional
    mean table ("current-token lookup"), linear-in-normed-embedding table,
    constant (the mean floor itself -- an inert module), and RANK-2
    ("the module writes a 2-dimensional signal": live write projected onto
    its top-2 held principal components, substituted at source).
 3. VERIFY -- substitution gate at SOURCE (attn_sub/mlp_sub), paired held-CE
    recovery = 1 - dce_sub/dce_floor on 96 fresh held seqs, fp32.
 4. NAME only if the gate passes. Threshold operationalization of the V8
    pass: 'named' requires best recovery >= 0.75 (the V8 substitutable
    cluster sat at 0.79+) with a consequential floor (>= 0.01 nats);
    floor < 0.005 -> 'inert/dead'; otherwise 'unnamed (contextual)' with the
    best-failed hypothesis and its recovery (V8's 'resists' verdict).
Each verdict carries the module's WIRING ROW: what it reads (its block's
top read-group norms) and who consumes it (weight support + causal delta CE
from qk_e9.json's light probe).

Deliverables in qk_e13.json: 'level5' (full ledgers), 'named_module_table'
(verdicts + recovery numbers + 2-3 concrete example contexts per module),
and 'comparison_vs_v8' (fraction nameable, mean recovery, same threshold
applied to the multi-epoch V8 pass -- does fresh data make modules MORE or
LESS nameable? caveat recorded: V8 was width 384, 6-epoch cooc, AdamW,
dilation mask; E9a is width 264, fresh single epoch, Muon). GPU-light."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, DEPTH, F, torch
import qk_e7_evenout_run as E7R

JP = E.jpath('qk_e13.json')
ABL_N = 8 if E.SMOKE else 96
N_TABLE = 8 if E.SMOKE else 1500
N_SNIP = 8 if E.SMOKE else 48
MIN_COUNT = 5
NAME_THRESH = 0.75
FLOOR_MIN = 0.01
INERT_FLOOR = 0.005


@torch.no_grad()
def ce_src(model, kind=None, l=None, tab=None, lowrank=None):
    """fp32 CE over fresh HELD[:ABL_N]; module write replaced at SOURCE by
    tab[input_ids] (V, D lookup) or by its rank-k projection (lowrank =
    (mean, P) with P (D, k): live two-pass per batch)."""
    tot, n = 0.0, 0
    for i in range(0, ABL_N, 8):
        b = Q.HELD[i:i + 8]
        inp = b[:, :Q.T]
        kw = {}
        if tab is not None:
            sub = {l: tab[inp]}
            kw = {'attn_sub': sub} if kind == 'attn' else {'mlp_sub': sub}
        elif lowrank is not None:
            col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
            model(inp, collect=col)
            w = col[f'{kind}_write'][l].float()
            mu, P = lowrank
            proj = mu + (w - mu) @ P @ P.t()
            sub = {l: proj}
            kw = {'attn_sub': sub} if kind == 'attn' else {'mlp_sub': sub}
        logits = model(inp, **kw)
        ce = F.cross_entropy(logits.reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        tot += ce.sum().item()
        n += ce.numel()
    return tot / n


@torch.no_grad()
def build_tables(model):
    """Token-conditional-mean + ridge linear-in-embedding tables from the
    fresh train stream (verbatim qk_v8_level5 machinery)."""
    D, V = Q.D, Q.V
    sums = {k: torch.zeros(DEPTH, V, D, device=E.DEV) for k in ('attn', 'mlp')}
    cnts = torch.zeros(V, device=E.DEV)
    for i in range(0, N_TABLE, 8):
        b = Q.TRAIN[i:i + 8, :Q.T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b, collect=col)
        idsf = b.reshape(-1)
        for l in range(DEPTH):
            sums['attn'][l].index_add_(
                0, idsf, col['attn_write'][l].float().reshape(-1, D))
            sums['mlp'][l].index_add_(
                0, idsf, col['mlp_write'][l].float().reshape(-1, D))
        cnts.index_add_(0, idsf, torch.ones_like(idsf, dtype=torch.float32))
    Emb = F.rms_norm(model.wte.weight.detach(), (D,))
    Ea = torch.cat([Emb, torch.ones(V, 1, device=E.DEV)], 1)
    A = Ea.t() @ (cnts[:, None] * Ea)
    lam = 1e-4 * torch.trace(A) / (D + 1)
    Areg = A + lam * torch.eye(D + 1, device=E.DEV)
    out = {}
    for kind in ('attn', 'mlp'):
        gmeans = sums[kind].sum(1) / cnts.sum()
        tok = sums[kind] / cnts.clamp(min=1)[None, :, None]
        rare = cnts < MIN_COUNT
        for l in range(DEPTH):
            tok[l][rare] = gmeans[l]
        lin = torch.empty_like(sums[kind])
        for l in range(DEPTH):
            W = torch.linalg.solve(Areg, Ea.t() @ sums[kind][l])
            lin[l] = Ea @ W
        out[kind] = {'gmeans': gmeans, 'tok': tok, 'lin': lin}
        del sums[kind]
    torch.cuda.empty_cache()
    return out, cnts


@torch.no_grad()
def collect_writes(model):
    aws, mws = [], []
    for i in range(0, N_SNIP, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b, collect=col)
        aws.append(torch.stack([w.float().cpu()
                                for w in col['attn_write']], 0))
        mws.append(torch.stack([w.float().cpu()
                                for w in col['mlp_write']], 0))
    return torch.cat(aws, 1), torch.cat(mws, 1)     # (DEPTH, N_SNIP, T, D)


@torch.no_grad()
def meaning_diagnostics(model, tokenizer, cnts, tables, aw, mw):
    D = Q.D
    ids = Q.HELD[:N_SNIP, :Q.T].cpu()
    tgt = Q.HELD[:N_SNIP, 1:Q.T + 1].cpu()
    U = model.wte.weight.detach().float()
    diag, bases = {}, {}
    for kind, W in (('attn', aw), ('mlp', mw)):
        for l in range(DEPTH):
            X = W[l].reshape(-1, D)
            norms = X.norm(dim=-1)
            r = {'mean_write_norm': round(float(norms.mean()), 4)}
            Xc = (X - X.mean(0)).to(E.DEV)
            q = min(2, min(Xc.shape) - 1) or 1
            _, S, Vt = torch.pca_lowrank(Xc, q=max(q, 1), niter=4)
            pcs = []
            for pi in range(min(2, Vt.shape[1])):
                v = Vt[:, pi]
                lg = U @ v
                top = torch.topk(lg, 8).indices.tolist()
                bot = torch.topk(-lg, 8).indices.tolist()
                pcs.append({'explained_frac': round(float(
                    S[pi] ** 2 / Xc.pow(2).sum().clamp_min(1e-12)), 4),
                    'plus_tokens': [tokenizer.decode([t]) for t in top],
                    'minus_tokens': [tokenizer.decode([t]) for t in bot]})
            r['pc_logit_lens'] = pcs
            bases[f'{kind}{l}'] = (X.mean(0).to(E.DEV),
                                   Vt[:, :2].to(E.DEV).contiguous())
            tok_tab = tables[kind]['tok'][l]
            tn = tok_tab.norm(dim=-1).clone()
            tn[cnts < MIN_COUNT] = 0.0
            topt = torch.topk(tn, 10).indices.tolist()
            r['top_input_tokens_by_write_norm'] = [
                [tokenizer.decode([t]), round(float(tn[t]), 3)] for t in topt]
            order = torch.argsort(norms, descending=True)
            snips, used = [], []
            for p in order.tolist():
                si, ti = p // Q.T, p % Q.T
                if ti < 4 or any(u[0] == si and abs(u[1] - ti) < 16
                                 for u in used):
                    continue
                used.append((si, ti))
                ctx = ids[si, max(0, ti - 12):ti + 1].tolist()
                lg = U @ X[p].to(E.DEV)
                snips.append({
                    'norm': round(float(norms[p]), 3),
                    'context': tokenizer.decode(ctx),
                    'last_token': tokenizer.decode([ids[si, ti].item()]),
                    'true_next': tokenizer.decode([tgt[si, ti].item()]),
                    'write_top_logits': [tokenizer.decode([t]) for t in
                                         torch.topk(lg, 6).indices.tolist()],
                    'write_bottom_logits': [tokenizer.decode([t]) for t in
                                            torch.topk(-lg, 6).indices
                                            .tolist()]})
                if len(snips) == 4:
                    break
            r['extreme_snippets'] = snips
            diag[f'{kind}{l}'] = r
    return diag, bases


def wiring_row(model, kind, l, probe):
    """Reads: the block's top read-group norms. Consumers: weight support +
    causal delta CE per consumer from qk_e9.json's light probe."""
    S = Q.D // E.NGROUP
    blk = model.h[l]
    reads = None
    for nm in E.READ_NAMES:
        M = getattr(blk, nm).weight.detach().float()
        g = M.pow(2).view(M.shape[0], E.NGROUP, S).sum((0, 2))
        reads = g if reads is None else reads + g
    reads = reads.sqrt()
    def sname(k):
        return ('attn' if k % 2 == 0 else 'mlp') + str(k // 2)
    top_reads = [[sname(k), round(float(reads[k]), 3)]
                 for k in torch.argsort(reads, descending=True)[:5].tolist()]
    si = str(1 + 2 * l) if kind == 'attn' else str(2 + 2 * l)
    cons = []
    cm = probe.get('consumption_matrix', {})
    ws = probe.get('weight_support_matrix', {})
    for li in cm:
        if si in cm[li]:
            cons.append({'consumer': ('readout' if int(li) == DEPTH
                                      else f'block{li}'),
                         'dce': cm[li][si],
                         'support': ws.get(li, {}).get(si)})
    cons.sort(key=lambda c: -(c['dce'] or 0))
    return {'top_reads_by_group_norm': top_reads, 'consumers': cons[:8]}


def compose_verdict(name, m, meaning):
    """Mechanical naming from the gated hypotheses (analyst conventions from
    the V8 level5_names pass, automated)."""
    floor = m['dce_floor']
    recs = {'token-conditional-mean table': m['tokmean_recovered'],
            'linear-in-embedding transform': m['linear_recovered'],
            'rank-2 write': m['lowrank2_recovered']}
    recs = {k: v for k, v in recs.items() if v is not None}
    best = max(recs, key=lambda k: recs[k]) if recs else None
    best_rec = recs.get(best)
    tops = [t for t, _ in meaning['top_input_tokens_by_write_norm'][:4]]
    pc = meaning['pc_logit_lens'][0] if meaning['pc_logit_lens'] else {}
    if floor < INERT_FLOOR:
        return {'named': True, 'name': f'INERT (mean-ablation floor '
                f'{floor:+.4f} nats -- no consequential function)',
                'gate': 'floor below 0.005'}
    if best_rec is not None and best_rec >= NAME_THRESH \
            and floor >= FLOOR_MIN:
        if best == 'token-conditional-mean table':
            nm = (f'current-token table (write is a lookup of the input '
                  f'token; strongest on {", ".join(repr(t) for t in tops)})')
        elif best == 'linear-in-embedding transform':
            nm = (f'linear token-embedding transform (fixed linear image of '
                  f'the current token; strongest on '
                  f'{", ".join(repr(t) for t in tops)})')
        else:
            nm = (f'rank-2 signal (2-dim write; principal direction promotes '
                  f'{pc.get("plus_tokens", [])[:3]} vs '
                  f'{pc.get("minus_tokens", [])[:3]})')
        return {'named': True, 'name': nm,
                'gate': f'{best} recovers {best_rec:.2f} of the '
                        f'{floor:+.4f}-nat floor (threshold {NAME_THRESH})'}
    return {'named': False,
            'name': 'unnamed (contextual -- resists all table/low-rank '
                    'substitutions)',
            'best_failed_hypothesis': best,
            'best_failed_recovery': best_rec,
            'gate': f'best {best} recovers only '
                    f'{best_rec if best_rec is not None else float("nan")} '
                    f'of {floor:+.4f}'}


def v8_comparison(mods):
    """Same-threshold stats on this pass and the multi-epoch V8 level5."""
    def stats(modset, rec_keys):
        consequential = {k: m for k, m in modset.items()
                         if (m.get('dce_floor') or 0) >= FLOOR_MIN}
        best = {k: max([m.get(rk) for rk in rec_keys
                        if m.get(rk) is not None] or [None],
                       key=lambda x: -1e9 if x is None else x)
                for k, m in consequential.items()}
        best = {k: v for k, v in best.items() if v is not None}
        named = [k for k, v in best.items() if v >= NAME_THRESH]
        return {'n_modules': len(modset),
                'n_consequential_floor_ge_0.01': len(consequential),
                'n_named_recovery_ge_0.75': len(named),
                'fraction_nameable_of_consequential':
                    round(len(named) / max(1, len(consequential)), 3),
                'mean_best_recovery_consequential':
                    round(float(np.mean(list(best.values()))), 3)
                    if best else None}
    res = {'e13_fresh': stats(mods, ('tokmean_recovered', 'linear_recovered',
                                     'lowrank2_recovered'))}
    v8p = f'{E.QK}/qk_v8.json'
    if os.path.exists(v8p):
        v8mods = json.load(open(v8p)).get('level5', {}).get('modules', {})
        res['v8_multiepoch'] = stats(v8mods, ('tokmean_recovered',
                                              'linear_recovered'))
        res['caveats'] = ('V8: width 384, 6-epoch cooc, AdamW, dilation '
                          'mask, no rank-2 hypothesis; E9a: width 264, '
                          'fresh single epoch, Muon, per-slot norm, full '
                          'visibility')
    return res


def main():
    E.setup()
    if E.SMOKE and not os.path.exists(E.ckpath('qk_e9_a')):
        m = E7R.make_e7m1()
        torch.save({'state_dict': m.state_dict(), 'config': {}, 'log': {}},
                   E.ckpath('qk_e9_a'))
        del m
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model, _ = E.load_arm('qk_e9_a', E7R.make_e7m1)
    probe = E.loadj(E.jpath('qk_e9.json')).get('light_probe_E9a', {})
    out = E.loadj(JP)
    t0 = time.time()

    if 'level5' in out and 'named_module_table' in out:
        print("e13: already complete -- skip", flush=True)
        print('e13 level5 run done', flush=True)
        return

    base = ce_src(model)
    print(f"base fp32 CE {base:.5f}", flush=True)
    tables, cnts = build_tables(model)
    print(f"tables built ({time.time() - t0:.0f}s)", flush=True)
    aw, mw = collect_writes(model)
    diag, bases = meaning_diagnostics(model, tokenizer, cnts, tables, aw, mw)
    del aw, mw
    print(f"meaning diagnostics done ({time.time() - t0:.0f}s)", flush=True)

    lvl = {'base_ce': round(base, 5), 'checkpoint': 'qk_e9_a.pt',
           'thresholds': {'named': NAME_THRESH, 'floor_min': FLOOR_MIN,
                          'inert': INERT_FLOOR}}
    mods = {}
    Vv, Dd = Q.V, Q.D
    for l in range(DEPTH):
        for kind in ('attn', 'mlp'):
            name = f'{kind}{l}'
            tabs = tables[kind]
            floor_tab = tabs['gmeans'][l][None].expand(Vv, Dd).contiguous()
            dfloor = ce_src(model, kind, l, floor_tab) - base
            dtok = ce_src(model, kind, l, tabs['tok'][l]) - base
            dlin = ce_src(model, kind, l, tabs['lin'][l]) - base
            dlr2 = ce_src(model, kind, l, lowrank=bases[name]) - base
            m = {'dce_floor': round(dfloor, 5),
                 'dce_tokmean': round(dtok, 5),
                 'dce_linear': round(dlin, 5),
                 'dce_lowrank2': round(dlr2, 5)}
            for k, dv in (('tokmean_recovered', dtok),
                          ('linear_recovered', dlin),
                          ('lowrank2_recovered', dlr2)):
                m[k] = round(1 - dv / dfloor, 4) if dfloor > 1e-4 else None
            m['wiring'] = wiring_row(model, kind, l, probe)
            m['meaning'] = diag[name]
            mods[name] = m
            print(f"{name}: floor {dfloor:+.4f} tok {m['tokmean_recovered']} "
                  f"lin {m['linear_recovered']} lr2 "
                  f"{m['lowrank2_recovered']} ({time.time() - t0:.0f}s)",
                  flush=True)
    lvl['modules'] = mods
    E.merge(JP, 'level5', lvl)

    table = {}
    for name, m in mods.items():
        v = compose_verdict(name, m, m['meaning'])
        v['recovery'] = {'floor': m['dce_floor'],
                         'tokmean': m['tokmean_recovered'],
                         'linear': m['linear_recovered'],
                         'lowrank2': m['lowrank2_recovered']}
        v['connections'] = m['wiring']
        v['example_contexts'] = [
            {'context': s['context'], 'write_norm': s['norm'],
             'write_pushes': s['write_top_logits'][:4],
             'true_next': s['true_next']}
            for s in m['meaning']['extreme_snippets'][:3]]
        table[name] = v
    E.merge(JP, 'named_module_table', table)
    E.merge(JP, 'comparison_vs_v8', v8_comparison(mods))
    n_named = sum(1 for v in table.values() if v['named'])
    print(f"named {n_named}/24 modules; comparison saved "
          f"({time.time() - t0:.0f}s)", flush=True)
    print('e13 level5 run done', flush=True)


if __name__ == '__main__':
    main()
