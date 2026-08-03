"""DENSEFORMER-STYLE VARIANT E (Logan, follow-up to the token-line experiment §106 and
its depth extension): a learned scalar for EVERY previous module when forming each
block's input.

Architecture E (depth-weighted module summation). Keep the list of module output
streams: s_0 = rms-normed embedding, then for each block j its attention write a_j and
its mlp write m_j (writes computed exactly as in the §106 code, qk_tokenline_train.py).
At block l's ENTRY the stream is REBUILT:
    x = w_{l,0} * s_0 + sum_{j<l} ( w_{l,aj} * a_j + w_{l,mj} * m_j )
with every w a learned scalar INITIALIZED AT 1.0 -- so at init E is EXACTLY variant A
vanilla (the rebuilt sum equals the accumulated residual stream; verified by a
positive-control forward comparison below). Within the block the §106 structure is
verbatim: attention reads rms_norm(x); its write a_l is added to x before the mlp
reads rms_norm(x + a_l); both writes go into the stream list. Before the readout norm,
one final learned combination row w_out (also init 1.0). Total scalars (depth+1)^2
(25 at depth 4, 169 at depth 12) -- trivial next to 30M params. No clips on the w.

TRAINING verbatim from qk_tokenline_train.py: width 384, 6 heads, head dim 64, vocab
50257, tied embeddings, FineWeb cooc corpus train [0:5500] held [5500:6000], T=512,
batch 8, lr 0.001 (frozen from the §106 variant-A sweep), 6-epoch budget (4122 steps),
same seed/init/data order (Block modules built in identical RNG order so init weights
match A bit-for-bit), AdamW wd 0.1 (scalars in the no-decay group like B's b), cosine
decay, warmup 250, grad clip 1.0. Depths 4 and 12, sequential, GPU guard.

Saves qk_denseform_{4,12}.pt + qk_denseform_heldloss_{4,12}.npy (per-token held losses,
bf16 eval convention matching qk_tokenline_heldloss_AS.npy / qk_tldepth_heldloss_*.npy
for PAIRED comparisons). CE table -> qk_denseform_ce.json (probes in qk_denseform_2.py).
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
DEPTHS = [4, 12]
EPOCHS = 6
STEPS = EPOCHS * Q.STEPS_PER_EPOCH            # 4122


class DenseMini(nn.Module):
    """Variant E: per-block-entry learned combination over ALL previous module streams."""
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.variant = 'E'
        # identical RNG consumption order to Q.MiniBilin: wte first, then blocks
        self.wte = nn.Embedding(V, D)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.h = nn.ModuleList([Q.Block('A') for _ in range(depth)])   # buffers a,b unused
        # combination rows AFTER (no RNG consumed by constant init): block l reads
        # 1 + 2l streams; the readout row reads 1 + 2*depth
        self.mix = nn.ParameterList([nn.Parameter(torch.ones(1 + 2 * l))
                                     for l in range(depth)])
        self.w_out = nn.Parameter(torch.ones(1 + 2 * depth))
        cos, sin = Q.rope_tables_exact(T, HD, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask', torch.tril(torch.ones(T, T, dtype=torch.bool)))

    @staticmethod
    def combine(w, streams):
        x = w[0] * streams[0]
        for i in range(1, len(streams)):
            x = x + w[i] * streams[i]
        return x

    def forward(self, idx, collect=None, mlp_sub=None):
        B, Tq = idx.shape
        e_norm = F.rms_norm(self.wte(idx), (D,))
        streams = [e_norm]                                 # s_0
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        for l, blk in enumerate(self.h):
            x = self.combine(self.mix[l], streams)          # REBUILT entry stream
            if collect is not None:
                collect['entry_norm'].append(x.detach().float().norm(dim=-1).mean().item())
            h = F.rms_norm(x, (D,))
            def qk(lin):
                z = lin(h).view(B, Tq, NH, HD)
                return Q.apply_rot(F.rms_norm(z, (HD,)), cos, sin)
            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(h).view(B, Tq, NH, HD)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D)
            aw = blk.c_proj(y)                              # a_l
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                hn = F.rms_norm(x, (D,))
                mw = blk.Down(blk.Left(hn) * blk.Right(hn)) + blk.Down_bias   # m_l
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = self.combine(self.w_out, streams)               # final combination row
        x = F.rms_norm(x, (D,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_dense(depth):
    torch.manual_seed(Q.SEED)
    return DenseMini(depth).to(DEV)


def mix_matrix(model):
    """Rows = block entries + readout row; columns = [s0, a_0, m_0, a_1, m_1, ...]."""
    rows = []
    ncol = 1 + 2 * model.depth
    for l in range(model.depth):
        w = model.mix[l].detach().float().cpu().tolist()
        rows.append(w + [None] * (ncol - len(w)))
    rows.append(model.w_out.detach().float().cpu().tolist())
    return rows


@torch.no_grad()
def init_control(depth=4):
    """Positive control: at init (all w = 1) E must equal variant A exactly."""
    me = make_dense(depth)
    Q.NL = depth
    ma = Q.make_model('A')
    me.eval().float(); ma.eval().float()
    idx = Q.HELD[:2, :T]
    d = (me(idx) - ma(idx)).abs().max().item()
    print(f"init control depth {depth}: max |logit(E)-logit(A)| = {d:.2e}", flush=True)
    assert d < 1e-3, f"E at init does not reproduce A (max diff {d})"
    del me, ma
    torch.cuda.empty_cache()


def train_dense(depth, lr, total_steps, log_every=200):
    Q.gpu_guard()
    model = make_dense(depth)
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
    log = {'variant': 'E', 'depth': depth, 'lr': lr, 'steps': total_steps,
           'train_loss': [], 'held_ce': [], 'spikes': 0}
    step, t0, run = 0, time.time(), None
    model.train()
    done = False
    for epoch in range(10 ** 6):
        order = Q.epoch_order(epoch)
        for i in range(Q.STEPS_PER_EPOCH):
            if step >= total_steps:
                done = True; break
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            seqs = Q.TRAIN[order[i * Q.BATCH:(i + 1) * Q.BATCH]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :T])
            loss = F.cross_entropy(logits.float().reshape(-1, V),
                                   seqs[:, 1:T + 1].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
            opt.step()
            l = loss.item()
            run = l if run is None else 0.98 * run + 0.02 * l
            if run is not None and l > run + 1.0:
                log['spikes'] += 1
            if step % log_every == 0:
                log['train_loss'].append([step, round(l, 4), round(run, 4)])
                print(f"  E{depth} step {step}/{total_steps} loss {l:.4f} "
                      f"(ema {run:.4f}) {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB", flush=True)
            if step % 2000 == 0 and step > 0:
                ce, _ = Q.eval_held(model, n_seq=100)
                log['held_ce'].append([step, round(ce, 4)])
                print(f"  E{depth} step {step} held100 CE {ce:.4f}", flush=True)
            step += 1
        if done:
            break
    ce, pt = Q.eval_held(model, per_token=True)
    log['final_held_ce'] = ce
    log['mix_matrix'] = mix_matrix(model)
    allw = torch.cat([model.mix[l].detach().float().cpu() for l in range(depth)]
                     + [model.w_out.detach().float().cpu()])
    log['w_stats'] = {'n': int(allw.numel()), 'min': float(allw.min()),
                      'max': float(allw.max()),
                      'n_moved_gt_0.5': int(((allw - 1).abs() > 0.5).sum())}
    print(f"== E depth {depth} FINAL held CE {ce:.5f}  w range "
          f"[{log['w_stats']['min']:.3f}, {log['w_stats']['max']:.3f}]  "
      f"moved>0.5: {log['w_stats']['n_moved_gt_0.5']}/{log['w_stats']['n']}  "
          f"({time.time() - t0:.0f}s, {log['spikes']} spikes)", flush=True)
    torch.save({'state_dict': model.state_dict(), 'variant': 'E',
                'config': dict(NL=depth, D=D, NH=NH, HD=HD, V=V, EXP=Q.EXP, T=T,
                               BATCH=Q.BATCH, EPOCHS=EPOCHS, SEED=Q.SEED,
                               DATA_SEED=Q.DATA_SEED, lr=lr, WARMUP=Q.WARMUP,
                               WD=Q.WD, GRAD_CLIP=Q.GRAD_CLIP, steps=total_steps),
                'log': log}, f'{QK}/qk_denseform_{depth}.pt')
    np.save(f'{QK}/qk_denseform_heldloss_{depth}.npy', pt)
    del model, opt
    torch.cuda.empty_cache()
    return log


BASE_HELDLOSS = {  # existing 6-epoch baselines, same held slice / eval convention
    4:  {'A': 'qk_tokenline_heldloss_AS.npy', 'B': 'qk_tokenline_heldloss_BS.npy'},
    12: {'A': 'qk_tldepth_heldloss_A12.npy',  'B': 'qk_tldepth_heldloss_B12.npy'},
}

if __name__ == '__main__':
    LR = json.load(open(f'{QK}/qk_tokenline_lrsweep.json'))['chosen']
    print(f"variant E run: depths {DEPTHS}, {STEPS} steps ({EPOCHS} epochs), lr {LR}, "
          f"batch {Q.BATCH}", flush=True)
    init_control(4)
    for depth in DEPTHS:
        if os.path.exists(f'{QK}/qk_denseform_{depth}.pt'):
            print(f"depth {depth} already trained -- skip", flush=True)
            continue
        print(f"==== training E depth {depth} ====", flush=True)
        train_dense(depth, LR, STEPS)

    # ---- paired per-token CE vs the existing A and B baselines ----
    out = {}
    for depth in DEPTHS:
        le = np.load(f'{QK}/qk_denseform_heldloss_{depth}.npy')
        row = {'held_ce_E': float(le.mean())}
        for v, fn in BASE_HELDLOSS[depth].items():
            lb = np.load(f'{QK}/{fn}')
            assert len(lb) == len(le)
            d = le - lb
            ds = d.reshape(len(Q.HELD), T).mean(1)
            row[f'held_ce_{v}'] = float(lb.mean())
            row[f'E_minus_{v}'] = float(d.mean())
            row[f'E_minus_{v}_se_token'] = float(d.std(ddof=1) / math.sqrt(len(d)))
            row[f'E_minus_{v}_se_seq'] = float(ds.std(ddof=1) / math.sqrt(len(ds)))
        out[str(depth)] = row
    json.dump(out, open(f'{QK}/qk_denseform_ce.json', 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
