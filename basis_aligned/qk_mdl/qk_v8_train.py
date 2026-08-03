"""V8 (follow-up to RESULTS_l0_mdl.md section 108, the deeproute architecture matrix):
the matrix's recommended combination for an interpretable retrain.

  V8 = V5 subspace-partitioned writes   (each of the 24 module writes projected at
                                         source into its own fixed 16-dim slot;
                                         the embedding keeps the full space)
     + V6 static dilation read mask     (consumer l reads block j's writes iff
                                         l - j in {1,2,4,8}; embedding visible iff
                                         l in {1,2,4,8}, plus the base case l=0;
                                         readout = consumer 12 sees blocks
                                         {4,8,10,11})
     + GROUP-SPARSITY pressure on reads (group lasso: for every block-consumer
                                         input matrix M in {c_q,c_k,c_q2,c_k2,c_v,
                                         Left,Right}, sum over the 24 16-column
                                         slot groups of ||M[:, 16k:16k+16]||_F,
                                         coefficient swept {1e-5,1e-4,1e-3} on
                                         short runs; pick the LARGEST coefficient
                                         costing < 0.02 nats vs no penalty on the
                                         short-run held CE. The readout's input
                                         matrix is the TIED embedding - penalizing
                                         it would shrink the embedding itself, so
                                         the readout is excluded from the penalty;
                                         its wiring stays readable from the V6 mask
                                         plus wte slot support.)

Kernel is the plain unit-weight SUM (no multiplicative routing), so the section-108
dead-gradient fixed point does not apply; per the follow-up spec the writes still get
the NONZERO init (std 0.02/sqrt(2*depth) ~ 0.004, generator seed 888, the same
convention the kernel variants used) rather than the zero init of the sum variants.

Depth 12, width 384, lr swept {0.001, 0.002, 0.003} x 400 steps (V1's re-swept
winner was 0.002), 6-epoch budget (4122 steps), data/seed conventions verbatim from
qk_deeproute_train.py / qk_tokenline_train.py. GPU guard, sequential runs.

Positive controls before training:
  (a) V8 with identity projections + full visibility mask + zeroed writes ==
      variant-A MiniBilin at init (max |logit diff| < 1e-3, fp32);
  (b) the vectorized group penalty equals a naive per-slice loop (rel err < 1e-6).

Saves qk_v8.pt, qk_v8_heldloss.npy (bf16 eval convention, paired vs
qk_deeproute_heldloss_V1.npy), sweeps + CE table -> qk_v8.json.
Interpretability scoring in qk_v8_probe.py, level-5 pass in qk_v8_level5.py.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_deeproute_train as R
from qk_deeproute_train import DeepRoute, DEPTH, SUBDIM

QK = Q.QK
DEV = 'cuda'
D, V, T = Q.D, Q.V, Q.T
NH, HD = Q.NH, Q.HD
EPOCHS = 6
STEPS = EPOCHS * Q.STEPS_PER_EPOCH               # 4122
LRS = [0.001, 0.002, 0.003]
COEFFS = [1e-5, 1e-4, 1e-3]
COEFF_BUDGET = 0.02                              # nats vs no-penalty short run
SWEEP_STEPS = 400
NGROUP = 2 * DEPTH                               # 24 slot groups cover all 384 dims

# register V8 in the deeproute tables: sum kernel + V5 projections + V6 mask +
# nonzero write init (spec'd std 0.02/sqrt(2*depth), generator 888 convention)
R.KERNEL['V8'] = 'sum'
R.PROJ.add('V8')
R.MASKED.add('V8')
R.WRITE_INIT.add('V8')


class V8Route(DeepRoute):
    """DeepRoute('V8') plus source-level attention-write substitution (attn_sub),
    needed by the level-5 pass; forward otherwise verbatim from DeepRoute."""

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        e = F.rms_norm(self.wte(idx), (D,))
        streams = [e]
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
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:   # source-level replacement
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = F.rms_norm(x, (D,))
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (D,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_v8():
    torch.manual_seed(Q.SEED)                    # identical block/wte init to V1..V7
    return V8Route('V8', DEPTH).to(DEV)


def load_v8():
    ck = torch.load(f'{QK}/qk_v8.pt', map_location=DEV, weights_only=False)
    m = make_v8()
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


READ_MATS = lambda blk: (blk.c_q.weight, blk.c_k.weight, blk.c_q2.weight,
                         blk.c_k2.weight, blk.c_v.weight, blk.Left.weight,
                         blk.Right.weight)


def group_penalty(model):
    """Sum over blocks, input matrices, and 24 slot groups of the group Frobenius
    norm ||M[:, 16k:16k+16]||_F (group lasso on read columns)."""
    tot = None
    for blk in model.h:
        for M in READ_MATS(blk):
            g = (M.pow(2).view(M.shape[0], NGROUP, SUBDIM).sum(dim=(0, 2))
                 + 1e-12).sqrt().sum()
            tot = g if tot is None else tot + g
    return tot


# ---------------- positive controls ----------------
@torch.no_grad()
def init_controls():
    idx = Q.HELD[:2, :T]
    Q.NL = DEPTH
    ma = Q.make_model('A').eval().float()
    ref = ma(idx)
    del ma
    torch.cuda.empty_cache()
    m = make_v8().eval().float()
    # (b) penalty vectorization check, on the nonzero-init model
    p_fast = float(group_penalty(m))
    p_naive = 0.0
    for blk in m.h:
        for M in READ_MATS(blk):
            for k in range(NGROUP):
                p_naive += float(M[:, SUBDIM * k:SUBDIM * (k + 1)].pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control penalty fast {p_fast:.4f} vs naive {p_naive:.4f} rel {rel:.2e}",
          flush=True)
    assert rel < 1e-6
    # (a) identity projections + full mask + zeroed writes == variant A at init
    m.wmask.fill_(1.0)
    m.vis = [list(range(1 + 2 * min(li, DEPTH))) for li in range(DEPTH + 1)]
    for blk in m.h:
        blk.c_proj.weight.zero_()
        blk.Down.weight.zero_()
    d = (m(idx) - ref).abs().max().item()
    print(f"control V8(identity proj, full mask, zero writes)==A: "
          f"max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del m
    torch.cuda.empty_cache()


# ---------------- training (adapted from qk_deeproute_train.train_deeproute) ----------------
def train_v8(lr, coeff, total_steps, log_every=200, save=True, factory=None,
             save_stem='qk_v8'):
    Q.gpu_guard(min_free=7000)
    model = (factory or make_v8)()
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

    log = {'variant': model.variant, 'lr': lr, 'group_coeff': coeff, 'steps': total_steps,
           'train_loss': [], 'held_ce': [], 'spikes': 0, 'diverged': False}
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
            ce = F.cross_entropy(logits.float().reshape(-1, V),
                                 seqs[:, 1:T + 1].reshape(-1))
            loss = ce + coeff * group_penalty(model) if coeff > 0 else ce
            l = ce.item()                        # spike/diverge on the CE part
            if not math.isfinite(l) or l > 30:
                log['diverged'] = True
                log['diverged_at'] = step
                print(f"  V8 DIVERGED at step {step} (loss {l})", flush=True)
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
                print(f"  V8[lr {lr}, gc {coeff}] step {step}/{total_steps} "
                      f"ce {l:.4f} (ema {run:.4f}) {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB",
                      flush=True)
            if step % 2000 == 0 and step > 0:
                hce, _ = Q.eval_held(model, n_seq=100)
                log['held_ce'].append([step, round(hce, 4)])
                print(f"  V8 step {step} held100 CE {hce:.4f}", flush=True)
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
    log['final_penalty'] = float(group_penalty(model))
    if save:
        hce, pt = Q.eval_held(model, per_token=True)
        log['final_held_ce'] = hce
        print(f"== V8 FINAL held CE {hce:.5f}  ({time.time() - t0:.0f}s, "
              f"{log['spikes']} spikes, penalty {log['final_penalty']:.2f})",
              flush=True)
        torch.save({'state_dict': model.state_dict(), 'variant': 'V8',
                    'config': dict(NL=DEPTH, D=D, NH=NH, HD=HD, V=V, EXP=Q.EXP,
                                   T=T, BATCH=Q.BATCH, EPOCHS=EPOCHS, SEED=Q.SEED,
                                   DATA_SEED=Q.DATA_SEED, lr=lr, group_coeff=coeff,
                                   WARMUP=Q.WARMUP, WD=Q.WD,
                                   GRAD_CLIP=Q.GRAD_CLIP, steps=total_steps),
                    'log': log}, f'{QK}/{save_stem}.pt')
        np.save(f'{QK}/{save_stem}_heldloss.npy', pt)
    else:
        log['final_held_ce'] = ce100
        print(f"  sweep lr {lr} gc {coeff}: held100 CE {ce100:.4f} "
              f"({log['spikes']} spikes)", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return log


if __name__ == '__main__':
    print(f"V8 run: depth {DEPTH}, {STEPS} steps ({EPOCHS} epochs), lr sweep {LRS}, "
          f"group-coeff sweep {COEFFS} (budget {COEFF_BUDGET} nats), "
          f"sweeps x {SWEEP_STEPS} steps", flush=True)
    Q.gpu_guard(min_free=7000)
    init_controls()

    out = json.load(open(f'{QK}/qk_v8.json')) if os.path.exists(f'{QK}/qk_v8.json') else {}

    # ---- stage 1: lr sweep at zero penalty ----
    if 'lrsweep' not in out:
        res = {}
        for lr in LRS:
            print(f"-- lr sweep {lr} (no penalty)", flush=True)
            log = train_v8(lr, 0.0, SWEEP_STEPS, log_every=100, save=False)
            res[str(lr)] = {'held100_ce': (None if log['diverged'] else
                                           round(log['final_held_ce'], 4)),
                            'diverged': log['diverged'], 'spikes': log['spikes']}
        ok = {k: v for k, v in res.items() if not v['diverged']}
        chosen = float(min(ok, key=lambda k: ok[k]['held100_ce']))
        out['lrsweep'] = {'results': res, 'chosen': chosen}
        json.dump(out, open(f'{QK}/qk_v8.json', 'w'), indent=2)
    LR = out['lrsweep']['chosen']
    print(f"lr chosen: {LR}", flush=True)

    # ---- stage 2: group-coefficient sweep at the chosen lr ----
    if 'coeffsweep' not in out:
        base_ce = out['lrsweep']['results'][str(LR)]['held100_ce']
        res = {'0': {'held100_ce': base_ce, 'cost_vs_zero': 0.0}}
        for c in COEFFS:
            print(f"-- coeff sweep {c} (lr {LR})", flush=True)
            log = train_v8(LR, c, SWEEP_STEPS, log_every=100, save=False)
            ce = None if log['diverged'] else round(log['final_held_ce'], 4)
            res[str(c)] = {'held100_ce': ce,
                           'cost_vs_zero': (None if ce is None else
                                            round(ce - base_ce, 4)),
                           'diverged': log['diverged'], 'spikes': log['spikes']}
        viable = [c for c in COEFFS
                  if res[str(c)]['cost_vs_zero'] is not None
                  and res[str(c)]['cost_vs_zero'] < COEFF_BUDGET]
        chosen = max(viable) if viable else 0.0
        out['coeffsweep'] = {'results': res, 'chosen': chosen,
                             'rule': f'largest coeff costing < {COEFF_BUDGET} nats '
                                     f'vs no penalty on the {SWEEP_STEPS}-step run'}
        json.dump(out, open(f'{QK}/qk_v8.json', 'w'), indent=2)
    COEFF = out['coeffsweep']['chosen']
    print(f"group coeff chosen: {COEFF}", flush=True)

    # ---- stage 3: full run ----
    if not os.path.exists(f'{QK}/qk_v8.pt'):
        print(f"==== training V8 full run (lr {LR}, group coeff {COEFF}) ====",
              flush=True)
        log = train_v8(LR, COEFF, STEPS)
        out['full_run'] = {'lr': LR, 'group_coeff': COEFF,
                           'held_ce_bf16': log['final_held_ce'],
                           'spikes': log['spikes'],
                           'final_penalty': log.get('final_penalty')}
        json.dump(out, open(f'{QK}/qk_v8.json', 'w'), indent=2)
    else:
        print("qk_v8.pt exists -- skip training", flush=True)

    # ---- CE table: paired per-token differences vs the section-108 V1 control ----
    lv = np.load(f'{QK}/qk_v8_heldloss.npy')
    l1 = np.load(f'{QK}/qk_deeproute_heldloss_V1.npy')
    dd = lv - l1
    ds = dd.reshape(len(Q.HELD), T).mean(1)
    out['ce'] = {'V8_held_ce': float(lv.mean()), 'V1_held_ce': float(l1.mean()),
                 'minus_V1': float(dd.mean()),
                 'minus_V1_se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
                 'minus_V1_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds))),
                 'lr': LR, 'group_coeff': COEFF}
    json.dump(out, open(f'{QK}/qk_v8.json', 'w'), indent=2)
    print(json.dumps(out['ce'], indent=2), flush=True)
