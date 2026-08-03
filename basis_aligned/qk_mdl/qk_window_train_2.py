"""WINDOWED-LOOKBACK EXPERIMENT, probes + scale-up (companion to qk_window_train.py).

Per trained model (vanilla control + W-N for N in {1,2,4,6}, width 384; plus width-768
variants if the scale-up rule fires):
  1. Held CE with per-token PAIRED sequence-clustered standard errors vs the re-swept
     vanilla control (from qk_window_ce.json / recomputed for width 768).
  2. Token-determined fraction of each layer's mlp write (group-mean method + shuffled
     control, VERBATIM group_r2/shuffle_control from qk_tokenline_probe).
  3. Linear-in-embedding per layer (machinery from qk_denseform_2): ridge map from the
     current-token rms-normed embedding fit on train[:1500], reporting held variance
     explained AND the causal recovered fraction (substitute the layer's mlp write with
     the linear prediction; recovered = 1 - dCE_lin / dCE_floor, floor = global-mean
     ablation). fp32 forwards, full held slice.
  4. WASH-OUT CURVE: variance of each block's ENTRY stream explained by current-token
     identity (same group-mean method). Bad-priors prediction: small-N models relay the
     token only where needed -> more-contextual mid-stack.
  5. Entry-stream norms by block.

SCALE-UP RULE (Logan): if any W-N matches or beats the re-swept vanilla control (within
2 sequence-clustered standard errors or better), train a WIDTH-768 version of that N
(best qualifying) and of vanilla: lr re-swept briefly at {winner/2, winner, winner*2},
batch reduced to 4 if a 3-step memory trial exceeds 7 GB, full 6-epoch budget; then the
same measures at width 768. Otherwise skip and say so.

HYPOTHESIS SCORE: bad-priors predicts windowed >= vanilla on held CE AND more-contextual
(less token-determined, less linear-in-embedding) mid-stack writes; the alternative
(persistent embedding access genuinely valuable) predicts monotone degradation as N
shrinks. Everything -> qk_window.json.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
from qk_tokenline_probe import group_r2, shuffle_control
import qk_window_train as QW
from qk_window_train import WindowMini

QK = Q.QK
DEV = 'cuda'
V, T = Q.V, Q.T
DEPTH = QW.DEPTH
HELD, TRAIN = Q.HELD, Q.TRAIN
N_COLLECT = 200
N_TABLE = 1500
MIN_COUNT = 5
MIDSTACK = list(range(3, 9))          # layers for the mid-stack contextuality score


def model_path(arch, width):
    return f'{QK}/qk_window_{arch}' + ('_w768' if width == 768 else '') + '.pt'


def heldloss_path(arch, width):
    return f'{QK}/qk_window_heldloss_{arch}' + ('_w768' if width == 768 else '') + '.npy'


def load_model(arch, width):
    ck = torch.load(model_path(arch, width), map_location=DEV, weights_only=False)
    m = QW.make_arch(arch, width)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


# ---------------- vanilla forward with entry collection + mlp substitution ----------------
@torch.no_grad()
def vanilla_fwd(model, idx, mlp_sub=None, col=None):
    D = model.wte.weight.shape[1]
    NH, HD = D // 64, 64
    B, Tq = idx.shape
    x = F.rms_norm(model.wte(idx), (D,))
    cos = model.cos[None, :Tq, None, :]
    sin = model.sin[None, :Tq, None, :]
    mask = model.mask[:Tq, :Tq]
    for li, blk in enumerate(model.h):
        if col is not None:
            col['entry_norm'].append(x.detach().float().norm(dim=-1).mean().item())
            if 'entry' in col:
                col['entry'].append(x.detach().float().cpu())
        h = F.rms_norm(x, (D,))
        def qkf(lin):
            z = lin(h).view(B, Tq, NH, HD)
            return Q.apply_rot(F.rms_norm(z, (HD,)), cos, sin)
        q, k = qkf(blk.c_q), qkf(blk.c_k)
        q2, k2 = qkf(blk.c_q2), qkf(blk.c_k2)
        v = blk.c_v(h).view(B, Tq, NH, HD)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D)
        aw = blk.c_proj(y)
        x = x + aw
        if mlp_sub is not None and li in mlp_sub:
            mw = mlp_sub[li]
        else:
            hn = F.rms_norm(x, (D,))
            mw = blk.Down(blk.Left(hn) * blk.Right(hn)) + blk.Down_bias
        if col is not None:
            col['attn_write'].append(aw.detach())
            col['mlp_write'].append(mw.detach())
        x = x + mw
    x = F.rms_norm(x, (D,))
    return 30 * torch.tanh((x @ model.wte.weight.t()) / 30)


@torch.no_grad()
def run_collect(model, arch, idx):
    """One forward, collecting entries (cpu) + writes (gpu)."""
    col = {'entry': [], 'entry_norm': [], 'attn_write': [], 'mlp_write': []}
    if arch == 'vanilla':
        vanilla_fwd(model, idx, col=col)
    else:
        model(idx, collect=col)
    return col


@torch.no_grad()
def collect_held(model, arch, D):
    """Held[:N_COLLECT] entry streams + mlp writes, cpu fp32 (N, depth, D)."""
    ents, mws, enorm, anorm = [], [], [], []
    for i in range(0, N_COLLECT, 8):
        idx = HELD[i:i + 8, :T]
        col = run_collect(model, arch, idx)
        ents.append(torch.stack(col['entry'], 1))                       # (B, depth, T, D)
        mws.append(torch.stack([w.float().cpu() for w in col['mlp_write']], 1))
        enorm.append(col['entry_norm'])
        anorm.append([w.float().norm(dim=-1).mean().item() for w in col['attn_write']])
    ent = torch.cat(ents).transpose(1, 2).reshape(-1, DEPTH, D)
    mw = torch.cat(mws).transpose(1, 2).reshape(-1, DEPTH, D)
    return ent, mw, np.array(enorm).mean(0).tolist(), np.array(anorm).mean(0).tolist()


@torch.no_grad()
def build_linear_tables(model, arch, D):
    """train[:N_TABLE] sweep -> per-layer global means + ridge linear-in-normed-embedding
    prediction tables (depth, V, D). VERBATIM logic from qk_denseform_2."""
    sums = torch.zeros(DEPTH, V, D, device=DEV)
    cnts = torch.zeros(V, device=DEV)
    for i in range(0, N_TABLE, 8):
        b = TRAIN[i:i + 8, :T]
        col = run_collect(model, arch, b)
        idsf = b.reshape(-1)
        for l in range(DEPTH):
            sums[l].index_add_(0, idsf, col['mlp_write'][l].float().reshape(-1, D))
        cnts.index_add_(0, idsf, torch.ones_like(idsf, dtype=torch.float32))
    gmeans = sums.sum(1) / cnts.sum()
    E = F.rms_norm(model.wte.weight.detach(), (D,))
    Ea = torch.cat([E, torch.ones(V, 1, device=DEV)], 1)
    A = Ea.t() @ (cnts[:, None] * Ea)
    lam = 1e-4 * torch.trace(A) / (D + 1)
    Areg = A + lam * torch.eye(D + 1, device=DEV)
    lin = torch.empty_like(sums)
    for l in range(DEPTH):
        W = torch.linalg.solve(Areg, Ea.t() @ sums[l])
        lin[l] = Ea @ W
    return gmeans, lin


@torch.no_grad()
def held_ce_sub(model, arch, sub_table=None, layer=None, batch=8):
    """fp32 held CE with optional (V,D) per-token substitution of `layer`'s mlp write."""
    pts = []
    for i in range(0, len(HELD), batch):
        b = HELD[i:i + batch]
        inp = b[:, :T]
        sub = None if sub_table is None else {layer: sub_table[inp]}
        if arch == 'vanilla':
            logits = vanilla_fwd(model, inp, mlp_sub=sub)
        else:
            logits = model(inp, mlp_sub=sub)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        pts.append(ce.cpu())
    pt = torch.cat(pts).numpy()
    return float(pt.mean()), pt


def r2_curve(W, gids):
    elig = np.ones(len(gids), bool)
    out = []
    for li in range(DEPTH):
        X64 = W[:, li, :].double()
        r2, G, cov = group_r2(X64, gids, elig, MIN_COUNT)
        ctl = shuffle_control(X64, gids, cov)
        out.append({'r2': round(r2, 4), 'shuffle_ctl': round(ctl, 4),
                    'groups': G, 'coverage': int(cov.sum())})
    return out


def probe(arch, width):
    print(f"==== probing {arch} width {width} ====", flush=True)
    t0 = time.time()
    Q.gpu_guard()
    model, ck = load_model(arch, width)
    D = model.wte.weight.shape[1]
    r = {'lr': ck['config']['lr'], 'held_ce_bf16_train_eval': ck['log']['final_held_ce'],
         'spikes': ck['log']['spikes']}

    gids = HELD[:N_COLLECT, :T].reshape(-1).cpu().numpy().astype(np.int64)
    ent, mw, entry_norm, attn_norm = collect_held(model, arch, D)
    r['entry_stream_norm'] = [round(v, 2) for v in entry_norm]
    r['attn_write_norm'] = [round(v, 3) for v in attn_norm]
    r['mlp_write_norm'] = [round(float(mw[:, li].norm(dim=1).mean()), 3)
                           for li in range(DEPTH)]
    r['washout_entry_r2'] = r2_curve(ent, gids)
    r['token_determined_mlp'] = r2_curve(mw, gids)
    del ent

    gmeans, lin = build_linear_tables(model, arch, D)
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

    base, base_pt = held_ce_sub(model, arch)
    r['base_ce_fp32'] = round(base, 5)
    causal = []
    for l in range(DEPTH):
        floor_tab = gmeans[l][None].expand(V, D).contiguous()
        cef, ptf = held_ce_sub(model, arch, floor_tab, l)
        cel, ptl = held_ce_sub(model, arch, lin[l], l)
        dfloor, dlin = cef - base, cel - base
        rec = round(1 - dlin / dfloor, 4) if dfloor > 1e-6 else None
        causal.append({
            'dce_floor': round(dfloor, 5),
            'dce_floor_se': round(float((ptf - base_pt).std(ddof=1) / math.sqrt(len(ptf))), 6),
            'dce_lin': round(dlin, 5),
            'dce_lin_se': round(float((ptl - base_pt).std(ddof=1) / math.sqrt(len(ptl))), 6),
            'lin_recovered': rec})
        print(f"  L{l}: tokdet {r['token_determined_mlp'][l]['r2']:.3f}  "
              f"linVE {ve[l]:.3f}  dCEfloor {dfloor:.4f}  dCElin {dlin:.4f}  "
              f"rec {rec}", flush=True)
    r['causal_per_layer'] = causal
    print(f"  {arch} w{width} done in {time.time() - t0:.0f}s", flush=True)
    del model, gmeans, lin
    torch.cuda.empty_cache()
    return r


# ---------------- width-768 scale-up ----------------
def pick_batch_768(arch):
    """3-step memory trial at batch 8; fall back to 4 above 7 GB or on OOM."""
    Q.gpu_guard()
    try:
        model = QW.make_arch(arch, 768)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        order = Q.epoch_order(0)
        model.train()
        for i in range(3):
            seqs = TRAIN[order[i * 8:(i + 1) * 8]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :T])
            loss = F.cross_entropy(logits.float().reshape(-1, V),
                                   seqs[:, 1:T + 1].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        del model, opt
        torch.cuda.empty_cache()
        print(f"  w768 {arch} batch-8 trial peak {peak:.0f} MiB", flush=True)
        return (8, int(peak)) if peak < 7000 else (4, int(peak))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  w768 {arch} batch-8 trial OOM -> batch 4", flush=True)
        return 4, None


def train_768(arch, sweeps):
    """lr re-sweep at {winner/2, winner, winner*2} then full 6-epoch run at width 768."""
    name = f'qk_window_{arch}_w768'
    if os.path.exists(f'{QK}/{name}.pt'):
        print(f"{name} already trained -- skip", flush=True)
        return sweeps.get(f'{arch}_w768', {})
    w384 = sweeps[arch]['chosen']
    grid = [w384 / 2, w384, w384 * 2]
    batch, peak = pick_batch_768(arch)
    key = f'{arch}_w768'
    if key not in sweeps:
        sweeps[key] = QW.sweep_arch(arch, grid, width=768, batch=batch)
        sweeps[key]['batch'] = batch
        sweeps[key]['batch8_trial_peak_mib'] = peak
        json.dump(sweeps, open(f'{QK}/qk_window_lrsweep.json', 'w'), indent=2)
    ranking = list(sweeps[key]['ranking'])
    steps = QW.EPOCHS * (Q.NTR // batch)
    for pick, lr in enumerate(ranking):
        print(f"==== training {arch} w768 at lr {lr} batch {batch} ({steps} steps) "
              f"({'sweep winner' if pick == 0 else f'fallback #{pick}'}) ====", flush=True)
        log = QW.train_model(arch, lr, steps, width=768, batch=batch, name=name)
        if not log['diverged']:
            if pick > 0:
                log['lr_fallback_note'] = f"winner {ranking[0]} diverged; sweep rank {pick}"
            return sweeps[key]
        print(f"  {arch} w768 lr {lr} diverged; falling back per sweep ranking", flush=True)
    print(f"  {arch} w768: ALL sweep lrs diverged -- no model", flush=True)
    return sweeps[key]


def paired_ce(width, archs):
    lv = np.load(heldloss_path('vanilla', width))
    out = {'vanilla': {'held_ce': float(lv.mean())}}
    for a in archs:
        fn = heldloss_path(a, width)
        if not os.path.exists(fn):
            out[a] = {'missing': True}
            continue
        ln = np.load(fn)
        d = ln - lv
        ds = d.reshape(len(HELD), T).mean(1)
        out[a] = {'held_ce': float(ln.mean()), 'minus_vanilla': float(d.mean()),
                  'minus_vanilla_se_token': float(d.std(ddof=1) / math.sqrt(len(d))),
                  'minus_vanilla_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}
    return out


# ---------------- hypothesis scoring ----------------
def score(out):
    ce = out['ce_w384']
    ns = [n for n in QW.WINDOWS if not ce[f'N{n}'].get('missing')]
    diffs = {n: ce[f'N{n}']['minus_vanilla'] for n in ns}
    ses = {n: ce[f'N{n}']['minus_vanilla_se_seq'] for n in ns}
    qualified = [n for n in ns if diffs[n] <= 2 * ses[n]]
    # mid-stack contextuality vs vanilla
    pv = out['probes_w384']['vanilla']
    mid_td_v = float(np.mean([pv['token_determined_mlp'][l]['r2'] for l in MIDSTACK]))
    mid_ve_v = float(np.mean([pv['linear_held_variance_explained'][l] for l in MIDSTACK]))
    mid = {}
    for n in ns:
        p = out['probes_w384'][f'N{n}']
        mid[str(n)] = {
            'td_mid': round(float(np.mean([p['token_determined_mlp'][l]['r2']
                                           for l in MIDSTACK])), 4),
            'linVE_mid': round(float(np.mean([p['linear_held_variance_explained'][l]
                                              for l in MIDSTACK])), 4)}
    mono = all(diffs[a] >= diffs[b] - 1e-12 for a, b in zip([1, 2, 4], [2, 4, 6])
               if a in diffs and b in diffs) and all(d >= 0 for d in diffs.values())
    return {
        'ce_diff_vs_vanilla': {str(n): round(diffs[n], 5) for n in ns},
        'ce_se_seq': {str(n): round(ses[n], 5) for n in ns},
        'qualified_within_2se': qualified,
        'badpriors_pred1_windowed_matches_or_beats_vanilla': len(qualified) > 0,
        'midstack_layers': MIDSTACK,
        'vanilla_midstack': {'td_mid': round(mid_td_v, 4), 'linVE_mid': round(mid_ve_v, 4)},
        'windowed_midstack': mid,
        'badpriors_pred2_more_contextual_midstack': {
            str(n): (mid[str(n)]['td_mid'] < mid_td_v and
                     mid[str(n)]['linVE_mid'] < mid_ve_v) for n in ns},
        'alternative_monotone_degradation_as_N_shrinks': mono,
    }


if __name__ == '__main__':
    Q.gpu_guard()
    out = {}
    if os.path.exists(f'{QK}/qk_window.json'):
        out = json.load(open(f'{QK}/qk_window.json'))
    out['ce_w384'] = json.load(open(f'{QK}/qk_window_ce.json'))

    # secondary control: §108's re-swept vanilla (qk_deeproute V1, lr 0.002, same
    # 6-epoch/seed/data protocol and bf16 eval convention) beat this run's
    # protocol-chosen vanilla (lr 0.001). Paired comparison of every model vs it.
    dv = f'{QK}/qk_deeproute_heldloss_V1.npy'
    if os.path.exists(dv):
        lv2 = np.load(dv)
        sec = {'note': 'qk_deeproute V1 = vanilla depth-12 at lr 0.002 (§108), '
                       'same protocol; stronger than the lr-0.001 control here',
               'held_ce': float(lv2.mean())}
        for a in ['vanilla'] + [f'N{n}' for n in QW.WINDOWS]:
            fn = heldloss_path(a, 384)
            if os.path.exists(fn):
                ln = np.load(fn)
                d = ln - lv2
                ds = d.reshape(len(HELD), T).mean(1)
                sec[a] = {'minus_V1': float(d.mean()),
                          'minus_V1_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}
        out['ce_w384_vs_deeproute_vanilla_lr002'] = sec

    sweeps = json.load(open(f'{QK}/qk_window_lrsweep.json'))
    out['lr_sweeps'] = sweeps

    out.setdefault('probes_w384', {})
    for arch in QW.ARCHS:
        if arch in out['probes_w384']:
            continue
        if not os.path.exists(model_path(arch, 384)):
            out['probes_w384'][arch] = {'missing': True}
            continue
        out['probes_w384'][arch] = probe(arch, 384)
        json.dump(out, open(f'{QK}/qk_window.json', 'w'), indent=2)

    out['hypothesis_score'] = score(out)
    json.dump(out, open(f'{QK}/qk_window.json', 'w'), indent=2)
    print(json.dumps(out['hypothesis_score'], indent=2), flush=True)

    # ---- scale-up rule ----
    qual = out['hypothesis_score']['qualified_within_2se']
    if not qual:
        out['scaleup'] = {'qualified': [], 'skipped': True,
                          'note': 'no W-N within 2 sequence-clustered SEs of vanilla'}
        json.dump(out, open(f'{QK}/qk_window.json', 'w'), indent=2)
        print("scale-up rule: NO W-N matches vanilla -- skipping width 768", flush=True)
    else:
        best = min(qual, key=lambda n: out['ce_w384'][f'N{n}']['minus_vanilla'])
        print(f"scale-up rule fires: qualified {qual}, training N{best} + vanilla at "
              f"width 768", flush=True)
        out['scaleup'] = {'qualified': qual, 'chosen_N': best}
        for arch in ['vanilla', f'N{best}']:
            train_768(arch, sweeps)
        out['lr_sweeps'] = sweeps
        out['scaleup']['ce_w768'] = paired_ce(768, [f'N{best}'])
        out.setdefault('probes_w768', {})
        for arch in ['vanilla', f'N{best}']:
            if arch not in out['probes_w768'] and os.path.exists(model_path(arch, 768)):
                out['probes_w768'][arch] = probe(arch, 768)
                json.dump(out, open(f'{QK}/qk_window.json', 'w'), indent=2)
        json.dump(out, open(f'{QK}/qk_window.json', 'w'), indent=2)
        print(json.dumps(out['scaleup'].get('ce_w768', {}), indent=2), flush=True)
    print('saved qk_window.json', flush=True)
