"""DEPTH-ROUTING PROTOTYPES (Logan): find the most interpretable-while-performant
stream-assembly rule for a future more-interpretable bilin18 retrain, at small scale
(width 384, depth 12, mini-bilin conventions verbatim from qk_tokenline_train.py /
qk_denseform.py: FineWeb cooc corpus train [0:5500] held [5500:6000], T=512, batch 8,
6-epoch budget 4122 steps, same seed/init/data order, NO-softmax bilinear attention +
bilinear MLP, no value-lerp, tied embeddings, logit soft-cap).

Stream bookkeeping as in qk_denseform.py: v_0 = rms-normed embedding, then for each
block j its attention write a_j (stream 1+2j) and mlp write m_j (stream 2+2j); block
l's ENTRY h_l is REBUILT from previous streams by the variant's assembly rule; block
internals unchanged (attn reads rms_norm(h_l), a_l added, mlp reads rms_norm(h_l+a_l));
readout = the same assembly rule applied at the top (consumer index 12), then rms_norm
and tied unembed.

  V1 CONTROL      plain unit-weight sum (vanilla residual stream), lr re-swept.
  V2 NORMSQ       h_l = sum_i alpha_i v_i, alpha_i = (w_l.v_i)^2 / (sum_j (w_l.v_j)^2
                  + eps), one learned pseudo-query w_l per consumer, per-position.
                  Rational (the normalizer is a bubble) but nonneg + normalized.
  V3 SQ (pure TN) h_l = (1/n_l) sum_i (w_l.v_i)^2 v_i, n_l = number of visible
                  streams (the spec's fixed 1/l made well-defined at block 0).
  V4 BILIN (signed) h_l = (1/n_l) sum_i (wL_l.v_i)(wR_l.v_i) v_i - can be negative
                  (subtractive routing / erasure).
  V5 SUBSPACE     plain sum, but each of the 24 module writes is projected AT SOURCE
                  into its own fixed 16-dim coordinate block P_k (write k = dims
                  [16k:16k+16], k = 2j for a_j, 2j+1 for m_j; the embedding keeps the
                  full space - "its own" stream entry, unprojected, since 24 disjoint
                  16-dim blocks exactly cover the 384 dims for the 24 writes). All
                  layers read the full stream; zeros are provable from weight support.
  V6 DILATION     plain unit-weight sum over the dilated set: consumer l includes
                  block j's writes iff (l - j) in {1,2,4,8}; embedding visible iff
                  l in {1,2,4,8}, plus the base case l=0 (block 0 sees only the
                  embedding, else it has no input). Readout (consumer 12) sees blocks
                  {4,8,10,11}. Constant diagonal core, trivially TN.
  V7 COMBO        V3 kernel (w_l unit-normalized, always) + V5 projections + V6 mask.

LR TUNED PER ARCHITECTURE: sweep {0.0005, 0.001, 0.002, 0.003} x 400 steps, pick by
quick held CE (100 held seqs); full 4122-step run at the winner. If every sweep lr
diverges: for V3/V4 retry the sweep with w unit-normalized (the pre-registered
fallback), else record the divergence and move on.

Positive controls before training (catch assembly bugs): V1 == variant-A MiniBilin
exactly at init; V5 with identity projections == V1; V6 with a full visibility mask
== V1 (all fp32, max |logit diff| < 1e-3).

Saves qk_deeproute_V{k}.pt, qk_deeproute_heldloss_V{k}.npy (bf16 eval convention,
matching all prior heldloss files for paired comparisons), qk_deeproute_lrsweep.json,
and the CE table with paired per-token/per-sequence standard errors vs V1 ->
qk_deeproute.json (key 'ce'). Interpretability scoring in qk_deeproute_train_2.py.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qk_tokenline_train as Q

QK = Q.QK
DEV = 'cuda'
D, V, T, NH, HD = Q.D, Q.V, Q.T, Q.NH, Q.HD
DEPTH = 12
EPOCHS = 6
STEPS = EPOCHS * Q.STEPS_PER_EPOCH            # 4122
LRS = [0.0005, 0.001, 0.002, 0.003]
SWEEP_STEPS = 400
DIL = (1, 2, 4, 8)
SUBDIM = 16                                    # 24 disjoint 16-dim write subspaces
EPS = 1e-8
VARIANTS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']

KERNEL = {'V1': 'sum', 'V2': 'normsq', 'V3': 'sq', 'V4': 'bilin',
          'V5': 'sum', 'V6': 'sum', 'V7': 'sq'}
PROJ = {'V5', 'V7'}
MASKED = {'V6', 'V7'}
UNIT = {'V7'}                                  # w_l unit-normalized always

# DEAD-GRADIENT FIX (discovered in run 1, recorded in qk_deeproute.json): the
# multiplicative routing coefficient (w.v_i)^2 (or (wL.v)(wR.v)) has ZERO gradient
# with respect to v_i at v_i = 0, and the bilin18 zero-init convention for c_proj /
# Down starts every write at exactly 0 -> the zero-write manifold is a fixed point.
# Run 1 proof: after 4122 steps V2/V3/V4/V7 had max|c_proj| = max|Down| = 0 exactly;
# V2 7.7505 / V3 7.8181 / V4 8.2448 are pure BIGRAM readouts of the embedding, and
# V7 (readout blind to the embedding under the dilation mask) sat at log(50257) =
# 10.8249, the uniform predictor. Fix: kernel variants get a small nonzero write
# init (GPT-2 residual scaling, std 0.02/sqrt(2*DEPTH) ~ 0.004) from a dedicated
# generator; sum variants keep the zero-init convention (V1/V5/V6 were unaffected).
WRITE_INIT = {'V2', 'V3', 'V4', 'V7'}
WRITE_INIT_STD = 0.02 / math.sqrt(2 * DEPTH)


def visible(variant, li, depth=DEPTH):
    """Stream indices visible to consumer li (block li's entry; li==depth: readout).
    Stream 0 = embedding, block j's writes = streams 1+2j (attn), 2+2j (mlp)."""
    nstream = 1 + 2 * min(li, depth)
    if variant not in MASKED:
        return list(range(nstream))
    if li == 0:
        return [0]                             # base case: block 0 reads the embedding
    idxs = [0] if li in DIL else []
    for j in range(min(li, depth)):
        if li - j in DIL:
            idxs += [1 + 2 * j, 2 + 2 * j]
    return sorted(idxs)


class DeepRoute(nn.Module):
    def __init__(self, variant, depth=DEPTH, unit_norm=None):
        super().__init__()
        self.variant = variant
        self.depth = depth
        self.kernel = KERNEL[variant]
        self.proj = variant in PROJ
        self.unit = (variant in UNIT) if unit_norm is None else unit_norm
        # identical RNG consumption order to Q.MiniBilin / DenseMini: wte, then blocks
        self.wte = nn.Embedding(V, D)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.h = nn.ModuleList([Q.Block('A') for _ in range(depth)])
        if variant in WRITE_INIT:               # escape the dead-gradient fixed point
            gw = torch.Generator().manual_seed(888)
            for blk in self.h:
                for p in (blk.c_proj.weight, blk.Down.weight):
                    with torch.no_grad():
                        p.copy_(torch.randn(p.shape, generator=gw) * WRITE_INIT_STD)
        # routing params AFTER, from a separate generator (main RNG untouched)
        gen = torch.Generator().manual_seed(777)
        nrows = depth + 1                       # one row per block entry + readout
        if self.kernel in ('normsq', 'sq'):
            self.w = nn.ParameterList(
                [nn.Parameter(torch.randn(D, generator=gen) / math.sqrt(D))
                 for _ in range(nrows)])
        elif self.kernel == 'bilin':
            self.wL = nn.ParameterList(
                [nn.Parameter(torch.randn(D, generator=gen) / math.sqrt(D))
                 for _ in range(nrows)])
            self.wR = nn.ParameterList(
                [nn.Parameter(torch.randn(D, generator=gen) / math.sqrt(D))
                 for _ in range(nrows)])
        self.vis = [visible(variant, li, depth) for li in range(depth + 1)]
        wm = torch.zeros(2 * depth, D)
        for k in range(2 * depth):
            wm[k, SUBDIM * k: SUBDIM * (k + 1)] = 1.0
        self.register_buffer('wmask', wm)
        cos, sin = Q.rope_tables_exact(T, HD, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask', torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def row_w(self, li):
        """The (unit-normalized if flagged) pseudo-query for consumer li, fp32."""
        w = self.w[li].float()
        return w / (w.norm() + EPS) if self.unit else w

    def assemble(self, li, streams, sub=None, coef_out=None):
        """Build consumer li's entry from the stream list. sub: {stream_idx: tensor}
        replaces a source IN THIS CONSUMER'S ASSEMBLY ONLY. coef_out: dict collecting
        (visible_idxs, per-position coefficients (B,T,n)) per consumer."""
        idxs = self.vis[li]
        get = lambda i: (sub[i] if (sub is not None and i in sub) else streams[i])
        if self.kernel == 'sum':
            h = None
            for i in idxs:
                h = get(i) if h is None else h + get(i)
            if coef_out is not None:
                s0 = get(idxs[0])
                coef_out[li] = (idxs, torch.ones(*s0.shape[:2], len(idxs),
                                                 device=s0.device))
            return h
        ss = [get(i) for i in idxs]
        if self.kernel == 'sq':
            w = self.row_w(li)
            dots = torch.stack([s @ w.to(s.dtype) for s in ss], -1).float()
            c = dots.pow(2) / len(idxs)
        elif self.kernel == 'normsq':
            w = self.row_w(li)
            dots = torch.stack([s @ w.to(s.dtype) for s in ss], -1).float()
            num = dots.pow(2)
            c = num / (num.sum(-1, keepdim=True) + EPS)
        else:                                   # bilin
            wL = self.wL[li].float()
            wR = self.wR[li].float()
            c = torch.stack([(s @ wL.to(s.dtype)).float() * (s @ wR.to(s.dtype)).float()
                             for s in ss], -1) / len(idxs)
        if coef_out is not None:
            coef_out[li] = (idxs, c.detach())
        h = None
        for k in range(len(idxs)):
            t = c[..., k:k + 1] * ss[k]
            h = t if h is None else h + t
        return h

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None):
        B, Tq = idx.shape
        e = F.rms_norm(self.wte(idx), (D,))
        streams = [e]                           # v_0
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = F.rms_norm(x, (D,))

            def qk(lin):
                z = lin(hn).view(B, Tq, NH, HD)
                return Q.apply_rot(F.rms_norm(z, (HD,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NH, HD)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D)
            aw = blk.c_proj(y)                  # a_l
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = F.rms_norm(x, (D,))
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias  # m_l
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)                   # readout: same assembly rule at the top
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (D,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_deeproute(variant, unit_norm=None):
    torch.manual_seed(Q.SEED)                   # identical block/wte init across variants
    return DeepRoute(variant, DEPTH, unit_norm=unit_norm).to(DEV)


# ---------------- positive controls ----------------
@torch.no_grad()
def init_controls():
    idx = Q.HELD[:2, :T]
    Q.NL = DEPTH
    ma = Q.make_model('A').eval().float()
    ref = ma(idx)
    del ma
    torch.cuda.empty_cache()
    # V1 == A exactly
    m = make_deeproute('V1').eval().float()
    d = (m(idx) - ref).abs().max().item()
    print(f"control V1==A: max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del m
    # V5 with identity projections == V1 (== A)
    m = make_deeproute('V5').eval().float()
    m.wmask.fill_(1.0)
    d = (m(idx) - ref).abs().max().item()
    print(f"control V5(identity proj)==A: max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del m
    # V6 with full visibility == V1 (== A)
    m = make_deeproute('V6').eval().float()
    m.vis = [list(range(1 + 2 * min(li, DEPTH))) for li in range(DEPTH + 1)]
    d = (m(idx) - ref).abs().max().item()
    print(f"control V6(full mask)==A: max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del m
    torch.cuda.empty_cache()


# ---------------- training ----------------
def train_deeproute(variant, lr, total_steps, log_every=200, save=True,
                    unit_norm=None):
    Q.gpu_guard()
    model = make_deeproute(variant, unit_norm=unit_norm)
    decay, nodecay = [], []
    for nm, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)   # routing w's: no decay
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': Q.WD},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < Q.WARMUP:
            return lr * (step + 1) / Q.WARMUP
        p = (step - Q.WARMUP) / max(1, total_steps - Q.WARMUP)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    log = {'variant': variant, 'lr': lr, 'steps': total_steps,
           'unit_norm': model.unit, 'train_loss': [], 'held_ce': [],
           'spikes': 0, 'diverged': False}
    step, t0, run = 0, time.time(), None
    model.train()
    done = False
    for epoch in range(10 ** 6):
        order = Q.epoch_order(epoch)
        for i in range(Q.STEPS_PER_EPOCH):
            if step >= total_steps:
                done = True
                break
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            seqs = Q.TRAIN[order[i * Q.BATCH:(i + 1) * Q.BATCH]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :T])
            loss = F.cross_entropy(logits.float().reshape(-1, V),
                                   seqs[:, 1:T + 1].reshape(-1))
            l = loss.item()
            if not math.isfinite(l) or l > 30:
                log['diverged'] = True
                log['diverged_at'] = step
                print(f"  {variant} DIVERGED at step {step} (loss {l})", flush=True)
                done = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
            opt.step()
            run = l if run is None else 0.98 * run + 0.02 * l
            if l > run + 1.0:
                log['spikes'] += 1
            if step % log_every == 0:
                log['train_loss'].append([step, round(l, 4), round(run, 4)])
                print(f"  {variant} step {step}/{total_steps} loss {l:.4f} "
                      f"(ema {run:.4f}) {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB",
                      flush=True)
            if step % 2000 == 0 and step > 0:
                ce, _ = Q.eval_held(model, n_seq=100)
                log['held_ce'].append([step, round(ce, 4)])
                print(f"  {variant} step {step} held100 CE {ce:.4f}", flush=True)
            step += 1
        if done:
            break
    if log['diverged']:
        log['final_held_ce'] = float('inf')
        del model, opt
        torch.cuda.empty_cache()
        return log
    ce100, _ = Q.eval_held(model, n_seq=100)
    log['held100_ce'] = ce100
    if save:
        ce, pt = Q.eval_held(model, per_token=True)
        log['final_held_ce'] = ce
        print(f"== {variant} FINAL held CE {ce:.5f}  ({time.time() - t0:.0f}s, "
              f"{log['spikes']} spikes)", flush=True)
        torch.save({'state_dict': model.state_dict(), 'variant': variant,
                    'config': dict(NL=DEPTH, D=D, NH=NH, HD=HD, V=V, EXP=Q.EXP, T=T,
                                   BATCH=Q.BATCH, EPOCHS=EPOCHS, SEED=Q.SEED,
                                   DATA_SEED=Q.DATA_SEED, lr=lr, WARMUP=Q.WARMUP,
                                   WD=Q.WD, GRAD_CLIP=Q.GRAD_CLIP, steps=total_steps,
                                   unit_norm=model.unit),
                    'log': log}, f'{QK}/qk_deeproute_{variant}.pt')
        np.save(f'{QK}/qk_deeproute_heldloss_{variant}.npy', pt)
    else:
        log['final_held_ce'] = ce100
        print(f"  sweep {variant} lr {lr}: held100 CE {ce100:.4f} "
              f"({log['spikes']} spikes)", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return log


def sweep_variant(variant, unit_norm=None):
    res = {}
    for lr in LRS:
        print(f"-- {variant} sweep lr {lr}"
              + (f" (unit_norm={unit_norm})" if unit_norm is not None else ""),
              flush=True)
        log = train_deeproute(variant, lr, SWEEP_STEPS, log_every=100, save=False,
                              unit_norm=unit_norm)
        res[str(lr)] = {'held100_ce': (None if log['diverged']
                                       else round(log['final_held_ce'], 4)),
                        'diverged': log['diverged'], 'spikes': log['spikes'],
                        'last_ema': (log['train_loss'][-1][2]
                                     if log['train_loss'] else None)}
    return res


if __name__ == '__main__':
    print(f"deeproute run: depth {DEPTH}, {STEPS} steps ({EPOCHS} epochs), "
          f"sweep lrs {LRS} x {SWEEP_STEPS} steps, batch {Q.BATCH}", flush=True)
    Q.gpu_guard()
    init_controls()

    sweep_file = f'{QK}/qk_deeproute_lrsweep.json'
    sweeps = json.load(open(sweep_file)) if os.path.exists(sweep_file) else {}

    for variant in VARIANTS:
        if os.path.exists(f'{QK}/qk_deeproute_{variant}.pt'):
            print(f"{variant} already trained -- skip", flush=True)
            continue
        # ---- per-architecture lr sweep (cached) ----
        if variant not in sweeps:
            res = sweep_variant(variant)
            entry = {'results': res, 'unit_norm_fallback': False}
            ok = {k: v for k, v in res.items() if not v['diverged']}
            if not ok and variant in ('V3', 'V4'):
                print(f"{variant}: all sweep lrs diverged -> unit-norm fallback",
                      flush=True)
                res2 = sweep_variant(variant, unit_norm=True)
                entry = {'results_raw': res, 'results': res2,
                         'unit_norm_fallback': True}
                ok = {k: v for k, v in res2.items() if not v['diverged']}
            if not ok:
                entry['chosen'] = None
                print(f"{variant}: DIVERGED at every sweep lr -- recorded, moving on",
                      flush=True)
            else:
                entry['chosen'] = float(min(ok, key=lambda k: ok[k]['held100_ce']))
            sweeps[variant] = entry
            json.dump(sweeps, open(sweep_file, 'w'), indent=2)
        entry = sweeps[variant]
        if entry['chosen'] is None:
            print(f"{variant}: no viable lr, skipping full run", flush=True)
            continue
        lr = entry['chosen']
        un = True if entry.get('unit_norm_fallback') else None
        print(f"==== training {variant} full run (lr {lr}"
              + (", unit-norm fallback" if un else "") + ") ====", flush=True)
        log = train_deeproute(variant, lr, STEPS, unit_norm=un)
        if log['diverged']:
            # full-run divergence at the sweep winner: retry once at the next-best lr
            res = sweeps[variant]['results']
            ok = sorted([k for k, v in res.items() if not v['diverged']
                         and float(k) != lr], key=lambda k: res[k]['held100_ce'])
            sweeps[variant]['full_run_diverged_at_lr'] = lr
            if ok:
                lr2 = float(ok[0])
                sweeps[variant]['retry_lr'] = lr2
                json.dump(sweeps, open(sweep_file, 'w'), indent=2)
                print(f"{variant}: full run diverged at lr {lr}; retry at {lr2}",
                      flush=True)
                log = train_deeproute(variant, lr2, STEPS, unit_norm=un)
            if log['diverged']:
                sweeps[variant]['full_run_result'] = 'diverged'
                json.dump(sweeps, open(sweep_file, 'w'), indent=2)
                print(f"{variant}: full run DIVERGED -- recorded", flush=True)

    # ---- CE table: paired per-token differences vs V1 ----
    out = {}
    if os.path.exists(f'{QK}/qk_deeproute.json'):
        out = json.load(open(f'{QK}/qk_deeproute.json'))
    ce = {}
    l1 = None
    if os.path.exists(f'{QK}/qk_deeproute_heldloss_V1.npy'):
        l1 = np.load(f'{QK}/qk_deeproute_heldloss_V1.npy')
    for variant in VARIANTS:
        fn = f'{QK}/qk_deeproute_heldloss_{variant}.npy'
        if not os.path.exists(fn):
            ce[variant] = {'status': 'diverged/missing',
                           'sweep': sweeps.get(variant, {})}
            continue
        lv = np.load(fn)
        row = {'held_ce': float(lv.mean()),
               'lr': sweeps.get(variant, {}).get('chosen')}
        if l1 is not None and variant != 'V1':
            dd = lv - l1
            ds = dd.reshape(len(Q.HELD), T).mean(1)
            row['minus_V1'] = float(dd.mean())
            row['minus_V1_se_token'] = float(dd.std(ddof=1) / math.sqrt(len(dd)))
            row['minus_V1_se_seq'] = float(ds.std(ddof=1) / math.sqrt(len(ds)))
        ce[variant] = row
    out['ce'] = ce
    out['lrsweep'] = sweeps
    json.dump(out, open(f'{QK}/qk_deeproute.json', 'w'), indent=2)
    print(json.dumps(ce, indent=2), flush=True)
