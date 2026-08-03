"""Probe the width-264 set: full interpretability block (causal consumption
graph, wiring agreement overall/effectual/top-10, dead blocks, exact terms at
blocks 4/6/9, standard probes) for V10 / V11 / V11nl / V11lr, plus the decoder
readability analysis Logan asked for:

- per-module decoder norms (slot-column Frobenius = the functional part,
  off-slot Frobenius = dead weight, bias norm) -> do unused decoders die?
- READOUT INTERPRETATION for the 5 largest-norm decoders: at extreme held
  activations of each module, the top promoted tokens of the RAW write w_i vs
  the DECODED write A_i w_i + b_i (does the learned decoder make the layer's
  readout meaning crisper?), plus a crispness metric: how often the true next
  token lands in the lens top-10 (raw vs decoded, top-200 extreme positions).
- V11lr: rank-1 channel reading for the 3 largest decoders -- each channel c is
  "when the write matches pattern V_c (see contexts), push logit direction U_c
  (see tokens)".

Wiring for the V11 arms: the readout row's weight-support prediction is the
decoder's slot-column Frobenius norm ||A_k[:, slot_k]||_F (the readout no longer
reads the raw slots through the tied embedding); block rows unchanged.

Merges into qk_v10.json ('probe') and qk_v11.json ('probe_*', 'decoder_*').
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import numpy as np
import torch

import qk_tokenline_train as Q
import qk_deeproute_train_2 as R2
import qk_v9_common as C
import qk_v10v11_common as W
from qk_deeproute_train import DEPTH

QK = W.QK
SUB = W.SUB
N_SNIP = 96
_ORIG_WW = R2.weight_wiring


def get_A(model, k):
    if hasattr(model, 'dec_U'):
        return model.dec_matrix(k)
    return model.dec[k].weight.detach().float()


def get_b(model, k):
    if hasattr(model, 'dec_U'):
        return model.dec_b[k].detach().float()
    return model.dec[k].bias.detach().float()


def ww_dec(model, variant):
    """R2.weight_wiring with the readout row replaced by decoder slot support."""
    scores, route = _ORIG_WW(model, variant)
    row = {}
    for k in range(2 * DEPTH):
        si = k + 1
        if si not in model.vis[DEPTH]:
            continue
        A = get_A(model, k)
        row[si] = float(A[:, SUB * k:SUB * (k + 1)].pow(2).sum().item() ** 0.5)
    scores[DEPTH] = row
    return scores, route


def decoder_norms(model):
    rows = []
    for k in range(2 * DEPTH):
        A = get_A(model, k)
        sl = A[:, SUB * k:SUB * (k + 1)]
        tot = float(A.pow(2).sum().item() ** 0.5)
        slf = float(sl.pow(2).sum().item() ** 0.5)
        rows.append({'module': R2.stream_name(k + 1),
                     'slot_frob': round(slf, 4),
                     'offslot_frob': round((max(tot ** 2 - slf ** 2, 0.0)) ** 0.5, 4),
                     'bias_norm': round(float(get_b(model, k).norm()), 4)})
    return rows


@torch.no_grad()
def collect_module_writes(model, modules):
    """writes[k] = (N_SNIP*T, D) fp32 CPU for each requested module index k
    (k = 2j attn_j, 2j+1 mlp_j)."""
    outs = {k: [] for k in modules}
    for i in range(0, N_SNIP, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b, collect=col)
        for k in modules:
            j, kind = k // 2, ('attn_write' if k % 2 == 0 else 'mlp_write')
            outs[k].append(col[kind][j].float().cpu())
    return {k: torch.cat(v).reshape(-1, W.WIDTH) for k, v in outs.items()}


def extreme_positions(norms, n, min_ti=4, min_gap=16, per_seq=None):
    order = torch.argsort(norms, descending=True)
    used, picks = [], []
    for p in order.tolist():
        si, ti = p // Q.T, p % Q.T
        if ti < min_ti:
            continue
        if any(u[0] == si and abs(u[1] - ti) < min_gap for u in used):
            continue
        if per_seq is not None and sum(1 for u in used if u[0] == si) >= per_seq:
            continue
        used.append((si, ti))
        picks.append(p)
        if len(picks) == n:
            break
    return picks


@torch.no_grad()
def decoder_interpretation(model, tokenizer, top_n=5, n_snip_show=3):
    """Raw-vs-decoded logit lens at extreme activations + hit@10 crispness."""
    U_emb = model.wte.weight.detach().float()
    # rank decoders by slot-column support
    slot_norm = [(k, float(get_A(model, k)[:, SUB * k:SUB * (k + 1)]
                           .pow(2).sum().item() ** 0.5)) for k in range(2 * DEPTH)]
    slot_norm.sort(key=lambda t: -t[1])
    top_modules = [k for k, _ in slot_norm[:top_n]]
    writes = collect_module_writes(model, top_modules)
    ids = Q.HELD[:N_SNIP, :Q.T].cpu()
    tgt = Q.HELD[:N_SNIP, 1:Q.T + 1].cpu()
    out = {}
    for k in top_modules:
        X = writes[k]
        norms = X.norm(dim=-1)
        A, b = get_A(model, k), get_b(model, k)
        name = R2.stream_name(k + 1)
        r = {'slot_frob': round(dict(slot_norm)[k], 3),
             'mean_write_norm': round(float(norms.mean()), 4)}
        # crispness over the top-200 extreme positions
        p200 = torch.tensor(extreme_positions(norms, 200, min_gap=1))
        Wx = X[p200].to(W.DEV)
        raw_lg = Wx @ U_emb.t()
        dec_lg = (Wx @ A.t() + b) @ U_emb.t()
        tt = tgt.reshape(-1)[p200].to(W.DEV)
        hit = lambda lg: float((lg.topk(10, dim=-1).indices ==
                                tt[:, None]).any(-1).float().mean())
        r['true_next_in_top10_raw'] = round(hit(raw_lg), 4)
        r['true_next_in_top10_decoded'] = round(hit(dec_lg), 4)
        # showcase snippets
        snips = []
        for p in extreme_positions(norms, n_snip_show):
            si, ti = p // Q.T, p % Q.T
            w = X[p].to(W.DEV)
            dv = A.to(W.DEV) @ w + b.to(W.DEV)
            lens = lambda v, n=8: [tokenizer.decode([t]) for t in
                                   torch.topk(U_emb @ v, n).indices.tolist()]
            snips.append({
                'norm': round(float(norms[p]), 3),
                'context': tokenizer.decode(ids[si, max(0, ti - 12):ti + 1].tolist()),
                'true_next': tokenizer.decode([tgt[si, ti].item()]),
                'raw_write_top_tokens': lens(w),
                'decoded_write_top_tokens': lens(dv)})
        r['extreme_snippets'] = snips
        out[name] = r
        print(f"  decoder {name}: slot_frob {r['slot_frob']}, hit@10 raw "
              f"{r['true_next_in_top10_raw']} -> decoded "
              f"{r['true_next_in_top10_decoded']}", flush=True)
    del writes
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def lowrank_channels(model, tokenizer, top_dec=3, top_ch=4):
    """V11lr: per-channel reading -- pattern V_c (contexts) -> direction U_c
    (tokens) -- for the largest decoders."""
    U_emb = model.wte.weight.detach().float()
    slot_norm = [(k, float(model.dec_matrix(k)[:, SUB * k:SUB * (k + 1)]
                           .pow(2).sum().item() ** 0.5)) for k in range(2 * DEPTH)]
    slot_norm.sort(key=lambda t: -t[1])
    mods = [k for k, _ in slot_norm[:top_dec]]
    writes = collect_module_writes(model, mods)
    ids = Q.HELD[:N_SNIP, :Q.T].cpu()
    out = {}
    for k in mods:
        X = writes[k]
        norms = X.norm(dim=-1)
        p200 = torch.tensor(extreme_positions(norms, 200, min_gap=1))
        Wx = X[p200].to(W.DEV)                       # (200, D)
        Uk = model.dec_U[k].detach().float()         # (D, r)
        Vk = model.dec_V[k].detach().float()
        acts = Wx @ Vk                               # (200, r) channel activations
        imp = acts.abs().mean(0) * Uk.norm(dim=0)    # importance per channel
        order = torch.argsort(imp, descending=True)[:top_ch].tolist()
        chans = []
        for c in order:
            a_all = (X.to(W.DEV) @ Vk[:, c]).cpu()   # activation on every position
            picks = extreme_positions(a_all.abs(), 3)
            ctxs = [{'act': round(float(a_all[p]), 3),
                     'context': tokenizer.decode(
                         ids[p // Q.T, max(0, p % Q.T - 10):p % Q.T + 1].tolist())}
                    for p in picks]
            lg = U_emb @ Uk[:, c].to(W.DEV)
            chans.append({
                'channel': c, 'importance': round(float(imp[c]), 4),
                'U_top_tokens': [tokenizer.decode([t]) for t in
                                 torch.topk(lg, 8).indices.tolist()],
                'U_bottom_tokens': [tokenizer.decode([t]) for t in
                                    torch.topk(-lg, 4).indices.tolist()],
                'pattern_contexts': ctxs})
        out[R2.stream_name(k + 1)] = {'slot_frob': round(dict(slot_norm)[k], 3),
                                      'channels': chans}
        print(f"  lowrank {R2.stream_name(k + 1)}: top channels "
              f"{[c['channel'] for c in chans]}", flush=True)
    del writes
    torch.cuda.empty_cache()
    return out


def probe_arm(stem, variant, kind, jpath, key):
    done = {}
    if os.path.exists(f'{QK}/{jpath}'):
        done = json.load(open(f'{QK}/{jpath}'))
    if key in done and 'probes' in done.get(key, {}):
        print(f"{key}: already probed -- skip", flush=True)
        return
    print(f"==== probing {stem} ====", flush=True)
    model, ck = W.load_arm(stem, variant, kind)
    if kind in ('v11', 'v11lr'):
        R2.weight_wiring = ww_dec
    try:
        C.full_probe(model, ck, save_cb=C.json_saver(f'{QK}/{jpath}', key))
    finally:
        R2.weight_wiring = _ORIG_WW
    if kind in ('v11', 'v11lr'):
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tag = key.replace('probe_', '')
        out = json.load(open(f'{QK}/{jpath}'))
        out[f'decoder_norms_{tag}'] = decoder_norms(model)
        json.dump(out, open(f'{QK}/{jpath}', 'w'), indent=2)
        interp = decoder_interpretation(model, tokenizer)
        out = json.load(open(f'{QK}/{jpath}'))
        out[f'decoder_interpretation_{tag}'] = interp
        if kind == 'v11lr':
            out['lowrank_channels_v11lr'] = lowrank_channels(model, tokenizer)
        json.dump(out, open(f'{QK}/{jpath}', 'w'), indent=2)
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    W.patch_width()
    Q.gpu_guard(min_free=3500)
    for stem, variant, kind, jpath, key in (
            ('qk_v10', 'V10', 'v10', 'qk_v10.json', 'probe'),
            ('qk_v11', 'V11', 'v11', 'qk_v11.json', 'probe_v11'),
            ('qk_v11nl', 'V11nl', 'v11', 'qk_v11.json', 'probe_v11nl'),
            ('qk_v11lr', 'V11lr', 'v11lr', 'qk_v11.json', 'probe_v11lr')):
        if not os.path.exists(f'{QK}/{stem}.pt'):
            print(f"{stem}.pt missing (diverged?) -- skip", flush=True)
            continue
        probe_arm(stem, variant, kind, jpath, key)
    print('probe script done', flush=True)
