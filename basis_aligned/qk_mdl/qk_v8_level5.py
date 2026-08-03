"""V8 LEVEL-5 INTERPRETATION PASS (companion to qk_v8_train.py / qk_v8_probe.py).
The four-ledger standard, for EVERY module (12 blocks x {attention, mlp}):

  FUNCTION        source-level mean-ablation floor: replace the module's write
                  everywhere it is consumed (stream + the mlp's internal read of
                  its own block's attention write) with the held-mean vector;
                  delta CE on 96 held seqs, fp32.
  SUBSTITUTABILITY  token-conditional-mean table and linear-in-normed-embedding
                  table substitution AT SOURCE (tables from 1500 train seqs,
                  ridge fit as in qk_deeproute_train_2.build_linear_tables);
                  recovered fraction = 1 - dce_sub / dce_floor.
  ANATOMY         wiring row assembled from qk_v8.json's probe: who consumes the
                  module (weight support + causal delta CE per consumer under the
                  V6 mask) and what the module reads (visible sources of its
                  block entry).
  MEANING (diagnostics for naming)  logit-lens of the write's top-2 principal
                  components through the tied unembedding (top/bottom tokens);
                  the input tokens with the largest conditional-mean write norm;
                  and the 4 extreme-activation held snippets (context decoded
                  with the GPT-2 tokenizer, plus the write's own direct-logit
                  top tokens at that position and the model's actual next token).

Saves everything into qk_v8.json under 'level5'. The names themselves (ledger 4's
verdicts) are composed from these diagnostics and recorded under 'level5_names'
by the analyst pass.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_v8_train as V8T
from qk_deeproute_train import DEPTH

QK = Q.QK
DEV = 'cuda'
D, V, T = Q.D, Q.V, Q.T
ABL_N = 96
N_TABLE = 1500
N_SNIP = 48                    # held seqs scanned for extreme activations
MIN_COUNT = 5


@torch.no_grad()
def ce_src(model, kind=None, l=None, tab=None):
    """fp32 CE over HELD[:ABL_N] with the module write of (kind, l) replaced at
    SOURCE by tab[input_ids] (tab: (V, D) lookup; mean floor = constant rows)."""
    tot, n = 0.0, 0
    for i in range(0, ABL_N, 8):
        b = Q.HELD[i:i + 8]
        inp = b[:, :T]
        kw = {}
        if tab is not None:
            sub = {l: tab[inp]}
            kw = {'attn_sub': sub} if kind == 'attn' else {'mlp_sub': sub}
        logits = model(inp, **kw)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        tot += ce.sum().item()
        n += ce.numel()
    return tot / n


@torch.no_grad()
def build_tables(model):
    """Token-count sums of both write kinds over TRAIN[:N_TABLE] -> per-module
    token-conditional-mean tables, global means, and ridge linear-in-normed-
    embedding tables (machinery as in qk_deeproute_train_2.build_linear_tables)."""
    sums = {k: torch.zeros(DEPTH, V, D, device=DEV) for k in ('attn', 'mlp')}
    cnts = torch.zeros(V, device=DEV)
    for i in range(0, N_TABLE, 8):
        b = Q.TRAIN[i:i + 8, :T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b, collect=col)
        idsf = b.reshape(-1)
        for l in range(DEPTH):
            sums['attn'][l].index_add_(0, idsf, col['attn_write'][l].float().reshape(-1, D))
            sums['mlp'][l].index_add_(0, idsf, col['mlp_write'][l].float().reshape(-1, D))
        cnts.index_add_(0, idsf, torch.ones_like(idsf, dtype=torch.float32))
    E = F.rms_norm(model.wte.weight.detach(), (D,))
    Ea = torch.cat([E, torch.ones(V, 1, device=DEV)], 1)
    A = Ea.t() @ (cnts[:, None] * Ea)
    lam = 1e-4 * torch.trace(A) / (D + 1)
    Areg = A + lam * torch.eye(D + 1, device=DEV)
    out = {}
    for kind in ('attn', 'mlp'):
        gmeans = sums[kind].sum(1) / cnts.sum()                    # (DEPTH, D)
        tok = sums[kind] / cnts.clamp(min=1)[None, :, None]
        rare = cnts < MIN_COUNT
        for l in range(DEPTH):
            tok[l][rare] = gmeans[l]                               # unseen -> global
        lin = torch.empty_like(sums[kind])
        for l in range(DEPTH):
            W = torch.linalg.solve(Areg, Ea.t() @ sums[kind][l])
            lin[l] = Ea @ W
        out[kind] = {'gmeans': gmeans, 'tok': tok, 'lin': lin}
        del sums[kind]
    torch.cuda.empty_cache()
    return out, cnts


@torch.no_grad()
def meaning_diagnostics(model, tokenizer, cnts, tables):
    """Per-module: PC logit-lens, top-input-token table, extreme snippets."""
    # collect writes + norms on HELD[:N_SNIP]
    aws, mws = [], []
    for i in range(0, N_SNIP, 8):
        b = Q.HELD[i:i + 8, :T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b, collect=col)
        aws.append(torch.stack([w.float().cpu() for w in col['attn_write']], 0))
        mws.append(torch.stack([w.float().cpu() for w in col['mlp_write']], 0))
    aw = torch.cat(aws, 1)                        # (DEPTH, N_SNIP, T, D)
    mw = torch.cat(mws, 1)
    ids = Q.HELD[:N_SNIP, :T].cpu()
    tgt = Q.HELD[:N_SNIP, 1:T + 1].cpu()
    U = model.wte.weight.detach().float()         # tied unembedding (V, D)
    diag = {}
    for kind, W in (('attn', aw), ('mlp', mw)):
        for l in range(DEPTH):
            X = W[l].reshape(-1, D)               # (N_SNIP*T, D)
            norms = X.norm(dim=-1)
            r = {'mean_write_norm': round(float(norms.mean()), 4)}
            # PC logit-lens
            Xc = (X - X.mean(0)).to(DEV)
            _, S, Vt = torch.pca_lowrank(Xc, q=2, niter=4)
            pcs = []
            for pi in range(2):
                v = Vt[:, pi]
                lg = U @ v
                top = torch.topk(lg, 8).indices.tolist()
                bot = torch.topk(-lg, 8).indices.tolist()
                pcs.append({'explained_frac': round(float(S[pi] ** 2 /
                                                          Xc.pow(2).sum()), 4),
                            'plus_tokens': [tokenizer.decode([t]) for t in top],
                            'minus_tokens': [tokenizer.decode([t]) for t in bot]})
            r['pc_logit_lens'] = pcs
            # top input tokens by conditional-mean write norm (count >= MIN_COUNT)
            tok_tab = tables[kind]['tok'][l]
            tn = tok_tab.norm(dim=-1).clone()
            tn[cnts < MIN_COUNT] = 0.0
            topt = torch.topk(tn, 10).indices.tolist()
            r['top_input_tokens_by_write_norm'] = [
                [tokenizer.decode([t]), round(float(tn[t]), 3)] for t in topt]
            # extreme snippets
            order = torch.argsort(norms, descending=True)
            snips, used = [], []
            for p in order.tolist():
                si, ti = p // T, p % T
                if ti < 4 or any(u[0] == si and abs(u[1] - ti) < 16 for u in used):
                    continue
                used.append((si, ti))
                ctx = ids[si, max(0, ti - 12):ti + 1].tolist()
                wvec = X[p].to(DEV)
                lg = U @ wvec
                snips.append({
                    'norm': round(float(norms[p]), 3),
                    'context': tokenizer.decode(ctx),
                    'last_token': tokenizer.decode([ids[si, ti].item()]),
                    'true_next': tokenizer.decode([tgt[si, ti].item()]),
                    'write_top_logits': [tokenizer.decode([t]) for t in
                                         torch.topk(lg, 6).indices.tolist()],
                    'write_bottom_logits': [tokenizer.decode([t]) for t in
                                            torch.topk(-lg, 6).indices.tolist()]})
                if len(snips) == 4:
                    break
            r['extreme_snippets'] = snips
            diag[f'{kind}{l}'] = r
    return diag


def main():
    Q.gpu_guard(min_free=7000)
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model, ck = V8T.load_v8()
    out = json.load(open(f'{QK}/qk_v8.json'))
    probe = out['probe']
    t0 = time.time()

    base = ce_src(model)
    print(f"base fp32 CE {base:.5f}", flush=True)
    tables, cnts = build_tables(model)
    print(f"tables built ({time.time() - t0:.0f}s)", flush=True)

    lvl = {'base_ce': round(base, 5)}
    mods = {}
    for l in range(DEPTH):
        for kind in ('attn', 'mlp'):
            name = f'{kind}{l}'
            tabs = tables[kind]
            floor_tab = tabs['gmeans'][l][None].expand(V, D).contiguous()
            dfloor = ce_src(model, kind, l, floor_tab) - base
            dtok = ce_src(model, kind, l, tabs['tok'][l]) - base
            dlin = ce_src(model, kind, l, tabs['lin'][l]) - base
            m = {'dce_floor': round(dfloor, 5),
                 'dce_tokmean': round(dtok, 5),
                 'dce_linear': round(dlin, 5),
                 'tokmean_recovered': (round(1 - dtok / dfloor, 4)
                                       if dfloor > 1e-4 else None),
                 'linear_recovered': (round(1 - dlin / dfloor, 4)
                                      if dfloor > 1e-4 else None)}
            # anatomy: consumers (support + causal) from the probe, under the mask
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
            m['reads'] = [Q_stream_name(s) for s in model.vis[l]]
            mods[name] = m
            print(f"{name}: floor {dfloor:+.4f} tokmean rec "
                  f"{m['tokmean_recovered']} linear rec {m['linear_recovered']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    lvl['modules'] = mods

    print("meaning diagnostics ...", flush=True)
    diag = meaning_diagnostics(model, tokenizer, cnts, tables)
    for name in mods:
        mods[name]['meaning'] = diag[name]

    out['level5'] = lvl
    json.dump(out, open(f'{QK}/qk_v8.json', 'w'), indent=2)
    print(f"saved qk_v8.json (level5) in {time.time() - t0:.0f}s", flush=True)


def Q_stream_name(i):
    if i == 0:
        return 'emb'
    j, kind = (i - 1) // 2, ('attn' if (i - 1) % 2 == 0 else 'mlp')
    return f'{kind}{j}'


if __name__ == '__main__':
    main()
