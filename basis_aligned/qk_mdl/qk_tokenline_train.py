"""TOKEN-LINE EXPERIMENT (Logan): does per-layer embedding re-injection (a) improve
language-modeling cross-entropy and (b) change what downstream layers compute?

Four mini-bilin18 models, identical except the per-block stream-entry mix
    x <- a*x + b*e0   (+ attn/mlp writes inside the block)
  A VANILLA        a=1 fixed, b=0 fixed              (no token line)
  B CLAMPED LINE   a=1 fixed, b learned (init 1), e0 = rms-normed embedding
  C FULL MIX       a,b both learned (init 1),       e0 = rms-normed embedding
  D UNCLAMPED LINE a=1 fixed, b learned (init 1), e0 = RAW embedding (magnitude kept)

Architecture (mini bilin18, conventions verbatim from tier2_model.py reference_forward
+ jacclust/tt_model.py): rms-normed embedding input; per block entry mix; attn input
rms_norm; per-head QK rms_norm THEN rotary (y1 = x1*c + x2*s, y2 = -x1*s + x2*c, exact
fp32 tables); NO-softmax bilinear attention pattern (q.k/hd)*(q2.k2/hd), causal-masked,
UNNORMALIZED; NO value-lerp (a.lamb dropped entirely); c_proj zero-init; bilinear MLP
Left*Right -> Down (Down zero-init, Down_bias); final rms_norm; TIED embedding/
unembedding; logits soft-cap 30*tanh(./30).

4 layers, width 384, 6 heads, head dim 64, vocab 50257 (data max id 50256, GPT-2).
Data /workspace/tensor_language/data_fineweb_cooc_tokens.npy (6000x513): train [0:5500],
held [5500:6000]. Fixed seed, identical init and data order across variants. AdamW,
cosine decay; lr picked by a short sweep on variant A ONLY, then FROZEN for all.
Sequential training with a GPU guard (<4500 MiB free -> sleep 20s), footprint < 6 GB.

Saves qk_tokenline_{A,B,C,D}.pt (weights + config + train log + mixing scalars) and
qk_tokenline_heldloss_{V}.npy (per-token held losses for paired standard errors).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- GPU guard (pattern from qk_redteam_arcs.py) ----------------
def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out")

# ---------------- config ----------------
NL, D, NH = 4, 384, 6
HD = D // NH                       # 64
V = 50257
EXP = 4                            # bilinear MLP expansion factor (bilin18 uses 4)
T = 512                            # inputs [:, :512] -> targets [:, 1:513]
BATCH = 8
EPOCHS = 15
SEED = 0
DATA_SEED = 1234
WARMUP = 250
GRAD_CLIP = 1.0
WD = 0.1

# ---------------- model ----------------
def rope_tables_exact(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    t = torch.arange(T, dtype=torch.float32)
    fr = torch.outer(t, inv)
    return fr.cos().to(device), fr.sin().to(device)

def apply_rot(x, c, s):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

class Block(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.c_q  = nn.Linear(D, D, bias=False)
        self.c_k  = nn.Linear(D, D, bias=False)
        self.c_q2 = nn.Linear(D, D, bias=False)
        self.c_k2 = nn.Linear(D, D, bias=False)
        self.c_v  = nn.Linear(D, D, bias=False)
        self.c_proj = nn.Linear(D, D, bias=False)
        self.c_proj.weight.data.zero_()
        self.Left  = nn.Linear(D, EXP * D, bias=False)
        self.Right = nn.Linear(D, EXP * D, bias=False)
        self.Down  = nn.Linear(EXP * D, D, bias=False)
        self.Down.weight.data.zero_()
        self.Down_bias = nn.Parameter(torch.zeros(D))
        # mixing scalars: the ONLY difference between variants
        if variant == 'A':
            self.register_buffer('a', torch.tensor(1.0))
            self.register_buffer('b', torch.tensor(0.0))
        elif variant in ('B', 'D'):
            self.register_buffer('a', torch.tensor(1.0))
            self.b = nn.Parameter(torch.tensor(1.0))
        elif variant == 'C':
            self.a = nn.Parameter(torch.tensor(1.0))
            self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, e0, cos, sin, mask, collect=None):
        x = self.a * x + self.b * e0
        if collect is not None:
            collect['entry_norm'].append(x.detach().float().norm(dim=-1).mean().item())
        h = F.rms_norm(x, (D,))
        B, Tq, _ = h.shape

        def qk(lin):
            z = lin(h).view(B, Tq, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cos, sin)

        q, k = qk(self.c_q), qk(self.c_k)
        q2, k2 = qk(self.c_q2), qk(self.c_k2)
        v = self.c_v(h).view(B, Tq, NH, HD)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)          # UNNORMALIZED, no softmax
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D)
        aw = self.c_proj(y)
        x = x + aw
        mw = self.Down(self.Left(F.rms_norm(x, (D,))) * self.Right(F.rms_norm(x, (D,)))) + self.Down_bias
        x = x + mw
        if collect is not None:
            collect['attn_write'].append(aw.detach())
            collect['mlp_write'].append(mw.detach())
        return x

class MiniBilin(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.wte = nn.Embedding(V, D)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.h = nn.ModuleList([Block(variant) for _ in range(NL)])
        cos, sin = rope_tables_exact(T, HD, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask', torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def forward(self, idx, collect=None):
        B, Tq = idx.shape
        e_raw = self.wte(idx)
        e_norm = F.rms_norm(e_raw, (D,))
        x = e_norm                                        # stream start: same for ALL variants
        e0 = e_raw if self.variant == 'D' else e_norm     # the token line (D: raw magnitude kept)
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        for blk in self.h:
            x = blk(x, e0, cos, sin, mask, collect)
        x = F.rms_norm(x, (D,))
        logits = x @ self.wte.weight.t()                  # TIED unembedding
        return 30 * torch.tanh(logits / 30)

def make_model(variant):
    torch.manual_seed(SEED)                               # identical init across variants
    return MiniBilin(variant).to(DEV)

# ---------------- data ----------------
DATA = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy')
                        .astype(np.int64)).to(DEV)
TRAIN, HELD = DATA[:5500], DATA[5500:]
NTR = TRAIN.shape[0]
STEPS_PER_EPOCH = NTR // BATCH                            # 687 (drop remainder, fixed schedule)

def epoch_order(epoch):
    g = torch.Generator().manual_seed(DATA_SEED + epoch)  # identical across variants
    return torch.randperm(NTR, generator=g)

# ---------------- train / eval ----------------
def loss_of(model, seqs):
    logits = model(seqs[:, :T]).float()
    return F.cross_entropy(logits.reshape(-1, V), seqs[:, 1:T + 1].reshape(-1))

@torch.no_grad()
def eval_held(model, n_seq=None, per_token=False):
    model.eval()
    hs = HELD if n_seq is None else HELD[:n_seq]
    tot, n, pt = 0.0, 0, []
    for i in range(0, len(hs), BATCH):
        b = hs[i:i + BATCH]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(b[:, :T])
        logits = logits.float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        if per_token:
            pt.append(ce.cpu())
        tot += ce.sum().item(); n += ce.numel()
    model.train()
    return (tot / n, torch.cat(pt).numpy() if per_token else None)

def train_variant(variant, lr, total_steps, log_every=200, save=True, suffix=''):
    gpu_guard()
    model = make_model(variant)
    decay, nodecay = [], []
    for nm, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': WD},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    def lr_at(step):
        if step < WARMUP:
            return lr * (step + 1) / WARMUP
        p = (step - WARMUP) / max(1, total_steps - WARMUP)
        return lr * 0.5 * (1 + math.cos(math.pi * p))
    log = {'variant': variant, 'lr': lr, 'steps': total_steps, 'train_loss': [],
           'held_ce': [], 'spikes': 0}
    step, t0, run = 0, time.time(), None
    model.train()
    done = False
    for epoch in range(10 ** 6):
        order = epoch_order(epoch)
        for i in range(STEPS_PER_EPOCH):
            if step >= total_steps:
                done = True; break
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            seqs = TRAIN[order[i * BATCH:(i + 1) * BATCH]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :T])
            loss = F.cross_entropy(logits.float().reshape(-1, V),
                                   seqs[:, 1:T + 1].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            l = loss.item()
            run = l if run is None else 0.98 * run + 0.02 * l
            if run is not None and l > run + 1.0:
                log['spikes'] += 1
            if step % log_every == 0:
                log['train_loss'].append([step, round(l, 4), round(run, 4)])
                print(f"  {variant} step {step}/{total_steps} loss {l:.4f} "
                      f"(ema {run:.4f}) {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB", flush=True)
            if step % 2000 == 0 and step > 0:
                ce, _ = eval_held(model, n_seq=100)
                log['held_ce'].append([step, round(ce, 4)])
                print(f"  {variant} step {step} held100 CE {ce:.4f}", flush=True)
            step += 1
        if done:
            break
    ce, pt = eval_held(model, per_token=True)
    log['final_held_ce'] = ce
    mix = {'a': [float(b.a.detach()) for b in model.h],
           'b': [float(b.b.detach()) for b in model.h]}
    log['mix'] = mix
    print(f"== {variant} FINAL held CE {ce:.5f}  a={mix['a']}  b={mix['b']}  "
          f"({time.time() - t0:.0f}s, {log['spikes']} spikes)", flush=True)
    if save:
        torch.save({'state_dict': model.state_dict(), 'variant': variant,
                    'config': dict(NL=NL, D=D, NH=NH, HD=HD, V=V, EXP=EXP, T=T,
                                   BATCH=BATCH, EPOCHS=EPOCHS, SEED=SEED,
                                   DATA_SEED=DATA_SEED, lr=lr, WARMUP=WARMUP,
                                   WD=WD, GRAD_CLIP=GRAD_CLIP, steps=total_steps),
                    'log': log}, f'{QK}/qk_tokenline_{variant}{suffix}.pt')
        np.save(f'{QK}/qk_tokenline_heldloss_{variant}{suffix}.npy', pt)
    del model, opt
    torch.cuda.empty_cache()
    return log

# ---------------- main ----------------
if __name__ == '__main__':
    total_steps = EPOCHS * STEPS_PER_EPOCH
    print(f"data {tuple(DATA.shape)} max id {int(DATA.max())}; "
          f"{STEPS_PER_EPOCH} steps/epoch x {EPOCHS} epochs = {total_steps} steps", flush=True)

    # ---- lr sweep on variant A ONLY (short), then freeze ----
    sweep_file = f'{QK}/qk_tokenline_lrsweep.json'
    if os.path.exists(sweep_file):
        LR = json.load(open(sweep_file))['chosen']
        print(f"lr sweep cached: chosen {LR}", flush=True)
    else:
        results = {}
        for lr in [3e-4, 1e-3, 3e-3]:
            print(f"-- sweep lr {lr}", flush=True)
            log = train_variant('A', lr, total_steps=400, log_every=100, save=False)
            tail = [t[2] for t in log['train_loss'][-2:]]
            results[str(lr)] = {'ema_tail': tail, 'spikes': log['spikes'],
                                'last_ema': log['train_loss'][-1][2]}
        # choose: lowest final ema among runs with <=5 spikes
        ok = {k: v for k, v in results.items() if v['spikes'] <= 5} or results
        LR = float(min(ok, key=lambda k: ok[k]['last_ema']))
        json.dump({'results': results, 'chosen': LR}, open(sweep_file, 'w'), indent=2)
        print(f"lr sweep results {json.dumps(results)}; CHOSEN {LR}", flush=True)

    # ---- full runs, sequential, identical hyperparameters ----
    for variant in ['A', 'B', 'C', 'D']:
        if os.path.exists(f'{QK}/qk_tokenline_{variant}.pt'):
            print(f"variant {variant} already trained -- skip", flush=True)
            continue
        print(f"==== training variant {variant} (lr {LR}) ====", flush=True)
        train_variant(variant, LR, total_steps)

    # ---- paired per-token differences ----
    losses = {v: np.load(f'{QK}/qk_tokenline_heldloss_{v}.npy') for v in 'ABCD'}
    out = {v: {'held_ce': float(losses[v].mean())} for v in 'ABCD'}
    for v in 'BCD':
        d = losses[v] - losses['A']
        out[v]['minus_A'] = float(d.mean())
        out[v]['minus_A_se'] = float(d.std(ddof=1) / math.sqrt(len(d)))
        # sequence-clustered SE (tokens within a sequence are correlated)
        ds = d.reshape(len(HELD), T).mean(1)
        out[v]['minus_A_se_seq'] = float(ds.std(ddof=1) / math.sqrt(len(ds)))
    json.dump(out, open(f'{QK}/qk_tokenline_ce.json', 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
