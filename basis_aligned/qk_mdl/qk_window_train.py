"""WINDOWED-LOOKBACK EXPERIMENT (Logan, follow-up to §106-§107): each block can only
see the last N layers' writes. Hypothesis: the standard architecture becomes dependent
on the persistently-available input embedding and learns BAD PRIORS; a finite window
removes the crutch and the model must actively re-encode/relay context.

ARCHITECTURE W-N (depth 12, width 384, block internals verbatim from
qk_tokenline_train.py, no value-lerp): keep the module-write list
[emb, a_0, m_0, a_1, m_1, ...] with emb = rms-normed embedding. Block l's ENTRY stream
is the PLAIN UNIT SUM of the writes (attention AND mlp) of blocks max(0, l-N) .. l-1,
PLUS emb only if l < N. Consumers read through rms_norm as usual. The readout reads the
sum of the last N blocks' writes (+ emb if depth < N). N in {1, 2, 4, 6}; N=1 is a pure
bucket brigade (zero carry). POSITIVE CONTROL: N = depth+1 with the vanilla zero-init
convention reproduces variant A exactly at init (verified below).

INIT DEVIATION (necessary, documented): the vanilla convention zero-inits c_proj and
Down. In a windowed model that provably DEADLOCKS training: with all writes zero, every
block's Jacobian with respect to its entry is zero (both write paths go through the
zero matrices), and — unlike vanilla — there is no residual carry to the readout, so
gradient reaches only the Down_bias of the last N blocks and the wave can never
propagate back (for N=1 the model is pinned at the unigram forever). Demonstrated
empirically by demo_deadlock() below. Fix: windowed models init c_proj.weight and
Down.weight ~ Normal(0, 0.02) (drawn AFTER the RNG-order-matched construction, so all
shared weights still match variant A bit-for-bit). The vanilla control keeps its own
zero-init convention (noted as a confound in the report).

LEARNING RATE TUNED PER ARCHITECTURE (Logan's explicit ask): for each N and for a
vanilla depth-12 control re-swept the same way, sweep {0.0005, 0.001, 0.002, 0.003}
for 400 steps, pick by held-100-sequence CE, then run the full 6-epoch budget
(4122 steps, batch 8) at the winner. Same seed / data order as §106-§107 (FineWeb cooc
corpus, train [0:5500], held [5500:6000]). If the full run diverges (NaN), fall back
to the next-best sweep lr (never per-run hand tuning) and note it. GPU guard as usual.

Saves qk_window_N{1,2,4,6}.pt + qk_window_vanilla.pt, per-token held losses
qk_window_heldloss_*.npy (bf16 eval convention, paired-comparable), sweep results
qk_window_lrsweep.json, CE table qk_window_ce.json. Probes + scale-up in
qk_window_train_2.py -> qk_window.json.
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
V, T = Q.V, Q.T
DEPTH = 12
EPOCHS = 6
STEPS = EPOCHS * Q.STEPS_PER_EPOCH            # 4122
WINDOWS = [1, 2, 4, 6]
LR_GRID = [0.0005, 0.001, 0.002, 0.003]
SWEEP_STEPS = 400
SWEEP_EVAL_SEQS = 100
WRITE_INIT_STD = 0.02
ARCHS = ['vanilla'] + [f'N{n}' for n in WINDOWS]


class WindowMini(nn.Module):
    """W-N: block entries and readout rebuilt from the last N blocks' writes only."""

    def __init__(self, depth, window, zero_write_init=False):
        super().__init__()
        self.depth, self.window = depth, window
        # capture dims at construction (Q globals may be patched for width scale-up)
        self.D, self.NH, self.HD, self.EXP = Q.D, Q.NH, Q.HD, Q.EXP
        self.variant = f'W{window}'
        # identical RNG order to Q.MiniBilin: wte first, then blocks
        self.wte = nn.Embedding(V, self.D)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.h = nn.ModuleList([Q.Block('A') for _ in range(depth)])
        if not zero_write_init:
            # AFTER matched construction: break the write-path deadlock (see header)
            for blk in self.h:
                nn.init.normal_(blk.c_proj.weight, std=WRITE_INIT_STD)
                nn.init.normal_(blk.Down.weight, std=WRITE_INIT_STD)
        cos, sin = Q.rope_tables_exact(T, self.HD, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask', torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def entry_parts(self, l, e_norm, writes):
        """Module streams visible at block l's entry (l = depth -> readout)."""
        parts = [e_norm] if l < self.window else []
        for j in range(max(0, l - self.window), l):
            parts.append(writes[j][0])
            parts.append(writes[j][1])
        return parts

    def forward(self, idx, collect=None, mlp_sub=None):
        B, Tq = idx.shape
        Dm, NHm, HDm = self.D, self.NH, self.HD
        e_norm = F.rms_norm(self.wte(idx), (Dm,))
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        writes = []                                        # per block (a_l, m_l)
        for l, blk in enumerate(self.h):
            parts = self.entry_parts(l, e_norm, writes)
            x = parts[0]
            for p in parts[1:]:
                x = x + p
            if collect is not None:
                collect['entry_norm'].append(x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach().float().cpu())
            h = F.rms_norm(x, (Dm,))
            def qkf(lin):
                z = lin(h).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)
            q, k = qkf(blk.c_q), qkf(blk.c_k)
            q2, k2 = qkf(blk.c_q2), qkf(blk.c_k2)
            v = blk.c_v(h).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)                             # a_l
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                hn = F.rms_norm(x, (Dm,))
                mw = blk.Down(blk.Left(hn) * blk.Right(hn)) + blk.Down_bias   # m_l
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            writes.append((aw, mw))
        parts = self.entry_parts(self.depth, e_norm, writes)
        x = parts[0]
        for p in parts[1:]:
            x = x + p
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_window(depth, window, width=384, zero_write_init=False):
    oldD, oldNH, oldHD = Q.D, Q.NH, Q.HD
    Q.D, Q.HD, Q.NH = width, 64, width // 64
    torch.manual_seed(Q.SEED)
    m = WindowMini(depth, window, zero_write_init=zero_write_init).to(DEV)
    Q.D, Q.NH, Q.HD = oldD, oldNH, oldHD
    return m


def make_vanilla(depth, width=384):
    oldD, oldNH, oldHD, oldNL = Q.D, Q.NH, Q.HD, Q.NL
    Q.D, Q.HD, Q.NH, Q.NL = width, 64, width // 64, depth
    m = Q.make_model('A')
    Q.D, Q.NH, Q.HD, Q.NL = oldD, oldNH, oldHD, oldNL
    return m


def make_arch(arch, width=384):
    if arch == 'vanilla':
        if width == 384:
            return make_vanilla(DEPTH, width)
        # Q.MiniBilin.forward reads module-global dims (384) at forward time, so at
        # other widths build vanilla as W-(depth+1) with zero write-init -- proven
        # EXACTLY equal to variant A by init_control() (entry = full accumulated
        # stream including emb, identical RNG order, identical zero-init writes).
        return make_window(DEPTH, DEPTH + 1, width, zero_write_init=True)
    return make_window(DEPTH, int(arch[1:]), width)


# ---------------- positive control: W-(depth+1) with zero-init == vanilla A ----------------
@torch.no_grad()
def init_control():
    mw = make_window(DEPTH, DEPTH + 1, zero_write_init=True)
    ma = make_vanilla(DEPTH)
    mw.eval().float(); ma.eval().float()
    idx = Q.HELD[:2, :T]
    d = (mw(idx) - ma(idx)).abs().max().item()
    print(f"init control: max |logit(W{DEPTH + 1} zero-init) - logit(A)| = {d:.2e}", flush=True)
    assert d < 1e-3, f"windowed forward does not reduce to vanilla (max diff {d})"
    del mw, ma
    torch.cuda.empty_cache()


# ---------------- deadlock demonstration (why the init deviation is needed) ----------------
def demo_deadlock(steps=80, lr=0.001):
    """W-1 with the vanilla zero-init convention: loss must stay pinned near the
    best-constant (unigram) level because no gradient can reach anything but the
    last block's Down_bias."""
    torch.manual_seed(Q.SEED)
    model = make_window(DEPTH, 1, zero_write_init=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)   # no decay: isolate gradient reach
    model.train()
    order = Q.epoch_order(0)
    first = last = None
    for i in range(steps):
        seqs = Q.TRAIN[order[i * Q.BATCH:(i + 1) * Q.BATCH]]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(seqs[:, :T])
        loss = F.cross_entropy(logits.float().reshape(-1, V), seqs[:, 1:T + 1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    # which parameters moved at all?
    ref = make_window(DEPTH, 1, zero_write_init=True)
    moved = [nm for (nm, p), (_, p0) in zip(model.named_parameters(), ref.named_parameters())
             if (p.detach() - p0.detach()).abs().max().item() > 1e-7]
    res = {'steps': steps, 'first_loss': round(first, 4), 'last_loss': round(last, 4),
           'n_params_moved': len(moved), 'moved': moved[:8]}
    print(f"deadlock demo (W-1, zero-init): loss {first:.3f} -> {last:.3f} after {steps} "
          f"steps; parameters moved: {len(moved)} ({moved[:4]}...)", flush=True)
    del model, ref, opt
    torch.cuda.empty_cache()
    return res


# ---------------- generic training loop (conventions verbatim from Q.train_variant) ----------------
def train_model(arch, lr, total_steps, width=384, batch=None, name=None,
                log_every=200, save=True):
    Q.gpu_guard()
    batch = batch or Q.BATCH
    steps_per_epoch = Q.NTR // batch
    model = make_arch(arch, width)
    decay, nodecay = [], []
    for nm, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': Q.WD},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    def lr_at(step):
        if step < Q.WARMUP:
            return lr * (step + 1) / Q.WARMUP
        p = (step - Q.WARMUP) / max(1, total_steps - Q.WARMUP)
        return lr * 0.5 * (1 + math.cos(math.pi * p))
    log = {'arch': arch, 'width': width, 'lr': lr, 'steps': total_steps, 'batch': batch,
           'train_loss': [], 'held_ce': [], 'spikes': 0, 'diverged': False}
    step, t0, run = 0, time.time(), None
    model.train()
    done = False
    for epoch in range(10 ** 6):
        order = Q.epoch_order(epoch)
        for i in range(steps_per_epoch):
            if step >= total_steps:
                done = True; break
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            seqs = Q.TRAIN[order[i * batch:(i + 1) * batch]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :T])
            loss = F.cross_entropy(logits.float().reshape(-1, V),
                                   seqs[:, 1:T + 1].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
            opt.step()
            l = loss.item()
            if not math.isfinite(l):
                log['diverged'] = True
                log['diverged_at_step'] = step
                print(f"  {arch} DIVERGED (non-finite loss) at step {step}", flush=True)
                done = True; break
            run = l if run is None else 0.98 * run + 0.02 * l
            if run is not None and l > run + 1.0:
                log['spikes'] += 1
            if step % log_every == 0:
                log['train_loss'].append([step, round(l, 4), round(run, 4)])
                print(f"  {arch} step {step}/{total_steps} loss {l:.4f} "
                      f"(ema {run:.4f}) {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB", flush=True)
            if step % 2000 == 0 and step > 0:
                ce, _ = Q.eval_held(model, n_seq=100)
                log['held_ce'].append([step, round(ce, 4)])
                print(f"  {arch} step {step} held100 CE {ce:.4f}", flush=True)
            step += 1
        if done:
            break
    log['peak_mem_mib'] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    if log['diverged']:
        del model, opt
        torch.cuda.empty_cache()
        return log
    ce, pt = Q.eval_held(model, per_token=True)
    log['final_held_ce'] = ce
    print(f"== {name or arch} FINAL held CE {ce:.5f}  ({time.time() - t0:.0f}s, "
          f"{log['spikes']} spikes)", flush=True)
    if save:
        torch.save({'state_dict': model.state_dict(), 'arch': arch,
                    'config': dict(NL=DEPTH, D=width, NH=width // 64, HD=64, V=V,
                                   EXP=Q.EXP, T=T, BATCH=batch, EPOCHS=EPOCHS,
                                   SEED=Q.SEED, DATA_SEED=Q.DATA_SEED, lr=lr,
                                   WARMUP=Q.WARMUP, WD=Q.WD, GRAD_CLIP=Q.GRAD_CLIP,
                                   steps=total_steps,
                                   write_init_std=(None if arch == 'vanilla'
                                                   else WRITE_INIT_STD)),
                    'log': log}, f'{QK}/{name}.pt')
        np.save(f'{QK}/{name.replace("qk_window_", "qk_window_heldloss_")}.npy', pt)
    del model, opt
    torch.cuda.empty_cache()
    return log


# ---------------- per-architecture lr sweep (held-100 selection) ----------------
def sweep_arch(arch, grid, width=384, batch=None, sweep_steps=SWEEP_STEPS):
    rows = {}
    for lr in grid:
        print(f"-- sweep {arch} w{width} lr {lr}", flush=True)
        Q.gpu_guard()
        model = make_arch(arch, width)
        b = batch or Q.BATCH
        opt = torch.optim.AdamW(
            [{'params': [p for _, p in model.named_parameters() if p.dim() >= 2],
              'weight_decay': Q.WD},
             {'params': [p for _, p in model.named_parameters() if p.dim() < 2],
              'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.95))
        def lr_at(step):
            if step < Q.WARMUP:
                return lr * (step + 1) / Q.WARMUP
            p = (step - Q.WARMUP) / max(1, sweep_steps - Q.WARMUP)
            return lr * 0.5 * (1 + math.cos(math.pi * p))
        model.train()
        spikes, run, diverged = 0, None, False
        order = Q.epoch_order(0)
        steps_per_epoch = Q.NTR // b
        for step in range(sweep_steps):
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            i = step % steps_per_epoch
            seqs = Q.TRAIN[order[i * b:(i + 1) * b]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :T])
            loss = F.cross_entropy(logits.float().reshape(-1, V),
                                   seqs[:, 1:T + 1].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
            opt.step()
            l = loss.item()
            if not math.isfinite(l):
                diverged = True; break
            run = l if run is None else 0.98 * run + 0.02 * l
            if l > run + 1.0:
                spikes += 1
        if diverged:
            rows[str(lr)] = {'held100_ce': None, 'spikes': spikes, 'diverged': True}
            print(f"   {arch} lr {lr}: DIVERGED", flush=True)
        else:
            ce, _ = Q.eval_held(model, n_seq=SWEEP_EVAL_SEQS)
            rows[str(lr)] = {'held100_ce': round(ce, 4), 'spikes': spikes,
                             'diverged': False, 'train_ema': round(run, 4)}
            print(f"   {arch} lr {lr}: held100 CE {ce:.4f} ({spikes} spikes)", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    ok = {k: v for k, v in rows.items() if not v['diverged']}
    ranking = sorted(ok, key=lambda k: ok[k]['held100_ce'])
    return {'results': rows, 'ranking': [float(k) for k in ranking],
            'chosen': float(ranking[0])}


if __name__ == '__main__':
    print(f"windowed-lookback run: depth {DEPTH}, windows {WINDOWS}, {STEPS} steps "
          f"({EPOCHS} epochs), batch {Q.BATCH}, lr grid {LR_GRID}", flush=True)
    Q.gpu_guard()
    init_control()

    sweep_file = f'{QK}/qk_window_lrsweep.json'
    sweeps = json.load(open(sweep_file)) if os.path.exists(sweep_file) else {}

    if 'deadlock_demo' not in sweeps:
        sweeps['deadlock_demo'] = demo_deadlock()
        json.dump(sweeps, open(sweep_file, 'w'), indent=2)

    for arch in ARCHS:
        if arch not in sweeps:
            sweeps[arch] = sweep_arch(arch, LR_GRID)
            json.dump(sweeps, open(sweep_file, 'w'), indent=2)
        print(f"{arch}: chosen lr {sweeps[arch]['chosen']} "
              f"(ranking {sweeps[arch]['ranking']})", flush=True)

    for arch in ARCHS:
        name = f'qk_window_{arch}'
        if os.path.exists(f'{QK}/{name}.pt'):
            print(f"{name} already trained -- skip", flush=True)
            continue
        ranking = list(sweeps[arch]['ranking'])
        trained = False
        for pick, lr in enumerate(ranking):
            print(f"==== training {arch} at lr {lr} "
                  f"({'sweep winner' if pick == 0 else f'fallback #{pick}'}) ====", flush=True)
            log = train_model(arch, lr, STEPS, name=name)
            if not log['diverged']:
                if pick > 0:
                    log['lr_fallback_note'] = (f"winner {ranking[0]} diverged at full "
                                               f"budget; used sweep rank {pick}")
                trained = True
                break
            print(f"  {arch} full run at lr {lr} diverged; falling back per sweep "
                  f"ranking", flush=True)
        if not trained:
            print(f"  {arch}: ALL sweep lrs diverged at full budget -- recorded, no "
                  f"model saved", flush=True)

    # ---- paired per-token CE table vs the re-swept vanilla control ----
    lv = np.load(f'{QK}/qk_window_heldloss_vanilla.npy')
    out = {'vanilla': {'held_ce': float(lv.mean()),
                       'lr': sweeps['vanilla']['chosen']}}
    for n in WINDOWS:
        fn = f'{QK}/qk_window_heldloss_N{n}.npy'
        if not os.path.exists(fn):
            out[f'N{n}'] = {'missing': True}
            continue
        ln = np.load(fn)
        d = ln - lv
        ds = d.reshape(len(Q.HELD), T).mean(1)
        out[f'N{n}'] = {'held_ce': float(ln.mean()),
                        'lr': sweeps[f'N{n}']['chosen'],
                        'minus_vanilla': float(d.mean()),
                        'minus_vanilla_se_token': float(d.std(ddof=1) / math.sqrt(len(d))),
                        'minus_vanilla_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}
    json.dump(out, open(f'{QK}/qk_window_ce.json', 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
