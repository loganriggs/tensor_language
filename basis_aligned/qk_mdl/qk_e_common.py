"""Shared machinery for the E-series interpretable-architecture experiments
(E0a/E0b controls, E1 per-slot RMSNorm, E2 CP-rank-capped bilinear MLPs,
E3 anneal-to-certified-zeros, E4 typed token slot).

PROTOCOL (Logan spec updates one + two, 2026-08-03): SINGLE EPOCH ON FRESH
DATA AT BATCH 16. Every width-264 arm trains exactly one pass of 8250 steps x
batch 16 = 132,000 fresh sequences read from the sharded corpus
    {QK}/corpus_fresh/shard00..06.npy   (uint16 rows of 513, concatenated in
                                         shard order, prefix taken)
with the fixed shuffle epoch_order(0) over those rows, IDENTICAL across arms.
corpus_fresh/fresh34k.npy is a PURE eval corpus (never trained on): held = its
rows [33000:34500]. Batch 16 is preflighted once (qk_e_batch.json: peak memory
+ measured step time; drops to 8 only if peak > 14 GiB). The family lr is
chosen by a 400-step sweep {0.002, 0.003, 0.004} on the E0b base arm (stored
in qk_e0.json['lrsweep'], used by every AdamW arm). The old cooc held set
[5500:6000] is ALSO evaluated per arm ({stem}_oldheldloss.npy) for
cross-corpus comparability; the existing 6-epoch checkpoints are NOT valid
controls here, so the chain trains fresh-stream controls first: E0a = vanilla
width 264, E0b = V8 base (partitioned write slots + group-lasso 1e-4 on reads
+ nonzero write init). All deltas are paired vs E0a and E0b on the fresh held
set. Two Muon-optimizer control arms (E0a/E0b architectures with Muon on the
2D hidden matrices, qk_e0m_muon_run.py; Muon lr swept {0.01, 0.02, 0.04}) and
four fresh-data readout-family replications (V10, V11, V11nl, V13r1,
qk_er_readout_run.py) share the exact same data order and protocol.

Probe note: the interpretability probes reuse qk_deeproute_train_2 machinery,
whose HELD/TRAIN bindings were taken at import time and therefore still point
at the OLD cooc corpus (consumption graph / linear tables / causal recovery run
on cooc data, comparable with the existing family probes; the models never saw
cooc data at all). The token-determined collection in qk_v8_probe reads Q.HELD
at call time and therefore uses the FRESH held set. Every arm is probed
identically, so within-family comparisons are unaffected.

SMOKE MODE (QK_SMOKE=1): reroutes every 'cuda' device request to CPU *before*
the harness modules are imported, shrinks to width 24 / 2 heads / head-dim 12 /
slot-dim 1 / T 64 / batch 2 / 3 steps, and redirects all outputs to
QK_SMOKE_DIR. Never touches the GPU.
"""
import os
import json
import math
import time

SMOKE = os.environ.get('QK_SMOKE') == '1'
SMOKE_DIR = os.environ.get(
    'QK_SMOKE_DIR',
    '/tmp/claude-0/-workspace-tensor-language/a6c5fb86-7bce-48e6-bb32-8679e85cbf66/'
    'scratchpad/qk_e_smoke')

import torch                                    # noqa: E402  (pre-harness patch)
import torch.nn as nn                           # noqa: E402
import torch.nn.functional as F                 # noqa: E402

if SMOKE:
    os.makedirs(SMOKE_DIR, exist_ok=True)

    def _cpu_dev(a):
        if isinstance(a, str) and a.startswith('cuda'):
            return 'cpu'
        if isinstance(a, torch.device) and a.type == 'cuda':
            return torch.device('cpu')
        return a

    _t_to = torch.Tensor.to
    def _tto(self, *args, **kw):
        args = tuple(_cpu_dev(a) for a in args)
        if 'device' in kw:
            kw['device'] = _cpu_dev(kw['device'])
        return _t_to(self, *args, **kw)
    torch.Tensor.to = _tto

    _m_to = torch.nn.Module.to
    def _mto(self, *args, **kw):
        args = tuple(_cpu_dev(a) for a in args)
        if 'device' in kw:
            kw['device'] = _cpu_dev(kw['device'])
        return _m_to(self, *args, **kw)
    torch.nn.Module.to = _mto

    _ac = torch.autocast
    class _SmokeAutocast(_ac):
        def __init__(self, device_type='cpu', dtype=None, **kw):
            super().__init__('cpu', dtype=torch.bfloat16)
    torch.autocast = _SmokeAutocast
    torch.cuda.empty_cache = lambda: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0

import numpy as np                              # noqa: E402
import qk_tokenline_train as Q                  # noqa: E402
import qk_deeproute_train as R                  # noqa: E402
import qk_v8_train as V8T                       # noqa: E402
import qk_deeproute_train_2 as R2               # noqa: E402
import qk_v8_probe as P8                        # noqa: E402
import qk_v9_common as C                        # noqa: E402
import qk_v10v11_common as W                    # noqa: E402
from qk_deeproute_train import DEPTH            # noqa: E402

QK = Q.QK
DEV = 'cpu' if SMOKE else 'cuda'
SHARD_DIR = f'{QK}/corpus_fresh'
HELD_PATH = f'{SHARD_DIR}/fresh34k.npy'          # PURE eval corpus, never trained
OLD_HELD = Q.HELD                    # cooc rows [5500:6000], bound BEFORE patching
NGROUP = 2 * DEPTH                   # 24 slot groups
GC = C.GROUP_COEFF                   # 1e-4 group-lasso on reads
LR = 0.002                           # fallback lr (sweep winner normally used)
LRS = (0.002, 0.003, 0.004)          # family AdamW lr sweep grid
MUON_LRS = (0.01, 0.02, 0.04)        # Muon lr sweep grid
STEPS = 8250                         # fixed step budget (batch 16 -> 132k seqs)
READ_NAMES = ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v', 'Left', 'Right')

E0A_STEM = 'qk_e0a_vanilla264'
E0B_STEM = 'qk_e0b_v8base264'

_DONE = False
SUB = None
WIDTH = None


def _load_train_prefix(need):
    """Concatenate corpus_fresh shards in order and take the prefix."""
    rows, got = [], 0
    for i in range(7):
        a = np.load(f'{SHARD_DIR}/shard{i:02d}.npy', mmap_mode='r')
        take = min(len(a), need - got)
        if take > 0:
            rows.append(np.asarray(a[:take]))
            got += take
        if got >= need:
            break
    assert got == need, f"sharded corpus too small: {got} < {need}"
    return torch.from_numpy(np.concatenate(rows).astype(np.int64)).to(DEV)


def preflight_batch():
    """One-time batch-16 memory/speed preflight on the E0b architecture
    (qk_e_batch.json). Drops to batch 8 only if peak > 14 GiB."""
    p = f'{QK}/qk_e_batch.json'
    if os.path.exists(p):
        return json.load(open(p))
    print("preflight: batch-16 memory/step-time on E0b ...", flush=True)
    res = {'batch': 16}
    try:
        torch.cuda.reset_peak_memory_stats()
        model = make_e0b()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                betas=(0.9, 0.95), weight_decay=Q.WD)
        order = Q.epoch_order(999)               # any rows; peek is harmless
        ts = []
        for s in range(6):
            seqs = Q.TRAIN[order[s * 16:(s + 1) * 16]]
            t0 = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :Q.T])
            ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                 seqs[:, 1:Q.T + 1].reshape(-1))
            loss = ce + GC * V8T.group_penalty(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
            opt.step()
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        res['peak_mib_batch16'] = round(peak)
        res['sec_per_step_batch16'] = round(sum(ts[2:]) / len(ts[2:]), 4)
        res['batch'] = 8 if peak > 14 * 1024 else 16
        del model, opt
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        res = {'batch': 8, 'peak_mib_batch16': 'OOM'}
        torch.cuda.empty_cache()
    json.dump(res, open(p, 'w'), indent=2)
    print(f"preflight: {json.dumps(res)}", flush=True)
    return res


def setup():
    """Patch the harness to width 264 (or the smoke config) and install the
    fresh single-epoch sharded corpus at the preflighted batch. Idempotent."""
    global _DONE, SUB, WIDTH
    if _DONE:
        return
    if SMOKE:
        Q.D, Q.NH, Q.HD = 24, 2, 12
        for mod in (R, V8T):
            mod.D, mod.NH, mod.HD, mod.SUBDIM = 24, 2, 12, 1
        R2.D, R2.SUBDIM = 24, 1
        W.SUB, W.WIDTH = 1, 24                   # V11/V13 decoder slot indexing
        Q.T = R.T = V8T.T = W.T = 64
        Q.BATCH = 2
        Q.WARMUP = 1
        sh0 = np.load(f'{SHARD_DIR}/shard00.npy', mmap_mode='r')
        held = np.load(HELD_PATH, mmap_mode='r')
        Q.TRAIN = torch.from_numpy(np.asarray(sh0[:40]).astype(np.int64)).to(DEV)
        Q.HELD = torch.from_numpy(
            np.asarray(held[33000:33016]).astype(np.int64)).to(DEV)
        Q.NTR = 40
        Q.STEPS_PER_EPOCH = 20
        V8T.STEPS = 3
        Q.gpu_guard = lambda *a, **k: None
        print("SMOKE setup: width 24, 2 heads, head 12, slot 1, T 64, batch 2, "
              "3 steps, CPU only", flush=True)
    else:
        W.patch_width()                          # 264 / 6 heads / 44 / slot 11
        need = STEPS * 16                        # 132,000 rows (batch-16 budget)
        Q.TRAIN = _load_train_prefix(need)
        held = np.load(HELD_PATH)[33000:34500].astype(np.int64)
        Q.HELD = torch.from_numpy(held).to(DEV)
        Q.NTR = need
        pf = preflight_batch()
        Q.BATCH = pf['batch']
        Q.STEPS_PER_EPOCH = STEPS                # fixed 8250-step budget
        V8T.STEPS = STEPS
        print(f"fresh single-epoch setup: {Q.NTR} shard rows loaded, "
              f"{V8T.STEPS} steps x batch {Q.BATCH} "
              f"({V8T.STEPS * Q.BATCH} seqs, single visit each, "
              f"epoch_order(0) shared across arms); held fresh34k[33000:34500] "
              f"({len(Q.HELD)} seqs)", flush=True)
    WIDTH = Q.D
    SUB = Q.D // NGROUP
    _DONE = True


def get_lr():
    """Family AdamW lr: the E0b sweep winner (qk_e0.json), fallback 0.002."""
    if SMOKE:
        return LR
    d = loadj(jpath('qk_e0.json'))
    if 'lrsweep' in d:
        return d['lrsweep']['chosen']
    print("WARNING: lr sweep result missing -- using fallback 0.002", flush=True)
    return LR


def lr_sweep(jp, key, lrs, trainfn):
    """Short-run lr sweep; trainfn(lr, steps) -> train log. Idempotent."""
    out = loadj(jp)
    if key in out:
        return out[key]['chosen']
    steps = 2 if SMOKE else 400
    res = {}
    for lr in lrs:
        print(f"-- {key}: sweep lr {lr} x {steps} steps", flush=True)
        log = trainfn(lr, steps)
        ce = None if log['diverged'] else round(log['final_held_ce'], 4)
        res[str(lr)] = {'held100_fresh_ce': ce, 'diverged': log['diverged'],
                        'spikes': log['spikes']}
    ok = {k: v for k, v in res.items() if v['held100_fresh_ce'] is not None}
    chosen = float(min(ok, key=lambda k: ok[k]['held100_fresh_ce'])) \
        if ok else float(lrs[0])
    merge(jp, key, {'results': res, 'chosen': chosen, 'sweep_steps': steps})
    print(f"{key}: chosen {chosen}", flush=True)
    return chosen


# ---------------- json bookkeeping ----------------
def jpath(name):
    return os.path.join(SMOKE_DIR if SMOKE else QK, name)


def loadj(jp):
    return json.load(open(jp)) if os.path.exists(jp) else {}


def merge(jp, key, val):
    out = loadj(jp)
    out[key] = val
    json.dump(out, open(jp, 'w'), indent=2)


def ckpath(stem):
    return os.path.join(SMOKE_DIR if SMOKE else QK, f'{stem}.pt')


# ---------------- factories ----------------
def make_e0a():
    """Vanilla width-264 control: sum kernel, full visibility, no slot
    projections, zero-init writes (== variant-A MiniBilin at init)."""
    R.KERNEL['E0a'] = 'sum'
    torch.manual_seed(Q.SEED)
    return V8T.V8Route('E0a', DEPTH).to(DEV)


def make_e0b():
    """V8 base at width 264: partitioned write slots + nonzero write init +
    full visibility (group-lasso 1e-4 applied by the trainer)."""
    C.register('E0b')
    torch.manual_seed(Q.SEED)
    return V8T.V8Route('E0b', DEPTH).to(DEV)


def load_arm(stem, factory):
    ck = torch.load(ckpath(stem), map_location=DEV, weights_only=False)
    m = factory()
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


# ---------------- training wrapper (curves saved into the JSON) ----------------
def train_arm(stem, jp, key, factory, gc, lr=None, trainer=None, extra=None):
    """One fresh-data single-epoch run (default trainer V8T.train_v8, Muon arms
    pass their own); lr defaults to the family sweep winner. Saves the FULL
    curve (train CE every 200 steps + held-100 fresh CE at 2000/4000/6000/8000)
    into the result JSON per Logan's request. Idempotent on the checkpoint."""
    out = loadj(jp)
    if key in out and (SMOKE or os.path.exists(ckpath(stem))):
        print(f"{key}: already trained -- skip", flush=True)
        return out[key]
    lr = get_lr() if lr is None else lr
    tf = trainer or V8T.train_v8
    print(f"==== training {stem} (lr {lr}, gc {gc}, {V8T.STEPS} steps x batch "
          f"{Q.BATCH}, single epoch) ====", flush=True)
    log = tf(lr, gc, V8T.STEPS, factory=factory, save_stem=stem,
             save=not SMOKE)
    rec = {'lr': lr, 'group_coeff': gc, 'width': Q.D, 'batch': Q.BATCH,
           'steps': V8T.STEPS, 'epochs': 1,
           'corpus': f'corpus_fresh shards prefix [{V8T.STEPS * Q.BATCH}], '
                     'single pass',
           'held_fresh': 'corpus_fresh/fresh34k.npy rows [33000:34500] '
                         '(pure eval corpus)',
           'final_held_ce_fresh_bf16': log.get('final_held_ce'),
           'spikes': log['spikes'], 'diverged': log['diverged'],
           'final_penalty': log.get('final_penalty'),
           'train_curve_every200': log['train_loss'],
           'held100_fresh_curve': log['held_ce']}
    if extra:
        rec.update(extra)
    merge(jp, key, rec)
    return rec


# ---------------- Muon optimizer (Logan update three) ----------------
# From-scratch Muon per the nanoGPT-speedrun convention: nesterov momentum 0.95
# buffer, 5-iteration Newton-Schulz orthogonalization of the 2D update in
# bfloat16, scaled by sqrt(max(rows, cols) / min(rows, cols)). Applied ONLY to
# 2D hidden matrices; the tied embedding/unembedding and all sub-2D parameters
# stay on AdamW at the family lr.
def ns_orth(G, steps=5):
    """Newton-Schulz iteration approximately orthogonalizing G (2D)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                assert p.dim() == 2, "Muon is for 2D matrices only"
                g = p.grad
                st = self.state[p]
                if 'buf' not in st:
                    st['buf'] = torch.zeros_like(g)
                buf = st['buf']
                buf.mul_(group['momentum']).add_(g)
                u = g.add(buf, alpha=group['momentum']) \
                    if group['nesterov'] else buf
                u = ns_orth(u, group['ns_steps']).to(p.dtype)
                r, cdim = p.shape
                scale = math.sqrt(max(r, cdim) / min(r, cdim))
                p.add_(u, alpha=-group['lr'] * scale)


def muon_params_split(model):
    """Muon: 2D hidden matrices. AdamW: tied embedding (decay) + sub-2D."""
    mu, adamw_decay, adamw_nodecay = [], [], []
    for nm, p in model.named_parameters():
        if p.dim() >= 2 and not nm.startswith('wte'):
            mu.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)
    return mu, adamw_decay, adamw_nodecay


def train_muon(lr_muon, coeff, total_steps, log_every=200, save=True,
               factory=None, save_stem='qk_e0m', lr_adamw=None):
    """train_v8's loop with the Muon/AdamW split (same schedule, data order,
    logging, spike/divergence guards, checkpoint + heldloss conventions).
    NOTE (flagged in the JSONs): the group-lasso penalty stays IN THE LOSS
    exactly as in the AdamW arms so the comparison isolates the optimizer, but
    Newton-Schulz orthogonalization distorts the penalty's shrinkage direction;
    a decoupled proximal soft-threshold would be cleaner -- follow-up, not
    implemented here."""
    Q.gpu_guard(min_free=4500)
    model = (factory or make_e0b)()
    lr_a = get_lr() if lr_adamw is None else lr_adamw
    mu, dec, nod = muon_params_split(model)
    opt_m = Muon(mu, lr=lr_muon)
    opt_a = torch.optim.AdamW([{'params': dec, 'weight_decay': Q.WD},
                               {'params': nod, 'weight_decay': 0.0}],
                              lr=lr_a, betas=(0.9, 0.95))

    def fac(step):
        if step < Q.WARMUP:
            return (step + 1) / Q.WARMUP
        p = (step - Q.WARMUP) / max(1, total_steps - Q.WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))

    log = {'variant': model.variant, 'lr': lr_muon, 'lr_adamw': lr_a,
           'group_coeff': coeff, 'steps': total_steps, 'train_loss': [],
           'held_ce': [], 'spikes': 0, 'diverged': False}
    step, t0, run = 0, time.time(), None
    model.train()
    done = False
    for epoch in range(10 ** 6):
        order = Q.epoch_order(epoch)
        for i in range(Q.STEPS_PER_EPOCH):
            if step >= total_steps:
                done = True
                break
            f = fac(step)
            for gpg in opt_m.param_groups:
                gpg['lr'] = lr_muon * f
            for gpg in opt_a.param_groups:
                gpg['lr'] = lr_a * f
            seqs = Q.TRAIN[order[i * Q.BATCH:(i + 1) * Q.BATCH]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(seqs[:, :Q.T])
            ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                                 seqs[:, 1:Q.T + 1].reshape(-1))
            loss = ce + coeff * V8T.group_penalty(model) if coeff > 0 else ce
            l = ce.item()
            if not math.isfinite(l) or l > 30:
                log['diverged'] = True
                log['diverged_at'] = step
                print(f"  MUON DIVERGED at step {step} (loss {l})", flush=True)
                done = True
                break
            opt_m.zero_grad(set_to_none=True)
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
            opt_m.step()
            opt_a.step()
            run = l if run is None else 0.98 * run + 0.02 * l
            if l > run + 1.0:
                log['spikes'] += 1
            if step % log_every == 0:
                log['train_loss'].append([step, round(l, 4), round(run, 4)])
                print(f"  MUON[lr {lr_muon}/{lr_a}] step {step}/{total_steps} "
                      f"ce {l:.4f} (ema {run:.4f}) {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}"
                      f"MiB", flush=True)
            if step % 2000 == 0 and step > 0:
                hce, _ = Q.eval_held(model, n_seq=100)
                log['held_ce'].append([step, round(hce, 4)])
                print(f"  MUON step {step} held100 fresh CE {hce:.4f}",
                      flush=True)
            step += 1
        if done:
            break
    if log['diverged']:
        log['final_held_ce'] = float('inf')
        del model, opt_m, opt_a
        torch.cuda.empty_cache()
        return log
    ce100, _ = Q.eval_held(model, n_seq=100)
    log['held100_ce'] = ce100
    log['final_penalty'] = float(V8T.group_penalty(model).detach())
    if save:
        hce, pt = Q.eval_held(model, per_token=True)
        log['final_held_ce'] = hce
        print(f"== MUON FINAL held CE {hce:.5f} ({time.time() - t0:.0f}s, "
              f"{log['spikes']} spikes)", flush=True)
        torch.save({'state_dict': model.state_dict(),
                    'variant': model.variant,
                    'config': dict(D=Q.D, NH=Q.NH, HD=Q.HD, T=Q.T,
                                   BATCH=Q.BATCH, SEED=Q.SEED, lr=lr_muon,
                                   lr_adamw=lr_a, group_coeff=coeff,
                                   optimizer='muon', WARMUP=Q.WARMUP,
                                   steps=total_steps),
                    'log': log}, ckpath(save_stem))
        np.save(f'{QK}/{save_stem}_heldloss.npy', pt)
    else:
        log['final_held_ce'] = ce100
        print(f"  muon sweep lr {lr_muon}: held100 CE {ce100:.4f} "
              f"({log['spikes']} spikes)", flush=True)
    del model, opt_m, opt_a
    torch.cuda.empty_cache()
    return log


# ---------------- evaluation on an explicit dataset ----------------
@torch.no_grad()
def eval_ce(model, data, per_token=False, n_seq=None, batch=8):
    """bf16-autocast eval convention (matches Q.eval_held) on an explicit
    dataset (used for the OLD cooc held set)."""
    model.eval()
    hs = data if n_seq is None else data[:n_seq]
    tot, n, pts = 0.0, 0, []
    for i in range(0, len(hs), batch):
        b = hs[i:i + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(b[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        if per_token:
            pts.append(ce.cpu())
        tot += ce.sum().item()
        n += ce.numel()
    return tot / n, (torch.cat(pts).numpy() if per_token else None)


def oldheld_record(stem, factory, jp, key):
    """Evaluate the trained arm on the OLD cooc held set (models never saw any
    cooc data), save per-token losses for pairing among fresh arms."""
    if SMOKE:
        return None
    out = loadj(jp)
    if key in out:
        return out[key]
    m, _ = load_arm(stem, factory)
    ce, pt = eval_ce(m, OLD_HELD, per_token=True)
    np.save(f'{QK}/{stem}_oldheldloss.npy', pt)
    rec = {'old_held_ce_bf16': round(ce, 5),
           'note': 'cooc rows [5500:6000]; model trained on fresh corpus only'}
    merge(jp, key, rec)
    del m
    torch.cuda.empty_cache()
    return rec


def paired(file_a, file_b, nseq, label):
    """Paired per-token CE difference with sequence-clustered standard error."""
    la = np.load(jpath(file_a) if SMOKE else f'{QK}/{file_a}')
    lb = np.load(jpath(file_b) if SMOKE else f'{QK}/{file_b}')
    dd = la - lb
    ds = dd.reshape(nseq, Q.T).mean(1)
    return {'held_ce': float(la.mean()), f'{label}_held_ce': float(lb.mean()),
            f'minus_{label}': float(dd.mean()),
            f'minus_{label}_se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
            f'minus_{label}_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}


def paired_fresh(stem, jp, key):
    """Pair an arm's fresh heldloss vs both fresh controls."""
    if SMOKE:
        return
    for ctl_stem, label in ((E0A_STEM, 'e0a'), (E0B_STEM, 'e0b')):
        f_ctl = f'{ctl_stem}_heldloss.npy'
        f_arm = f'{stem}_heldloss.npy'
        if os.path.exists(f'{QK}/{f_ctl}') and os.path.exists(f'{QK}/{f_arm}'):
            merge(jp, f'{key}_minus_{label}_fresh',
                  paired(f_arm, f_ctl, len(Q.HELD), label))


# ---------------- probes ----------------
def weight_support(model, slot_of=None):
    """Frobenius support of consumer li's seven read matrices (readout: tied
    wte) on module k's write slot. slot_of maps module index -> slot index
    (default identity; E4 uses the shifted map)."""
    slot_of = slot_of or (lambda k: k)
    scores = {}
    for li in range(DEPTH + 1):
        mats = ([model.wte.weight] if li == DEPTH else
                [getattr(model.h[li], nm).weight for nm in READ_NAMES])
        row = {}
        for k in range(NGROUP):
            si = k + 1
            if si not in model.vis[li] or si > 2 * min(li, DEPTH):
                continue
            s = slot_of(k)
            dims = slice(SUB * s, SUB * (s + 1))
            row[si] = float(sum(M[:, dims].float().pow(2).sum().item()
                                for M in mats) ** 0.5)
        scores[li] = row
    return scores


def light_probe(model, slot_of=None):
    """Causal consumption graph (mean-ablation, old cooc held) + weight-support
    wiring agreement: the weight-vs-causal Spearman readability probe used by
    the family (qk_v13_run pattern)."""
    base, means = R2.base_and_means(model)
    if hasattr(model, 'embed_stream'):
        # base_and_means hardcodes stream 0 = full normed embedding; E4's
        # stream 0 is the masked token line, so recompute its ablation mean.
        with torch.no_grad():
            s = torch.zeros(means.shape[1], dtype=torch.float64,
                            device=means.device)
            n = 0
            for i in range(0, R2.ABL_N, 8):
                b = R2.HELD[i:i + 8]
                e = model.embed_stream(b[:, :R2.T])
                s += e.double().sum((0, 1))
                n += b.shape[0] * R2.T
            means[0] = (s / n).float()
    dce = R2.consumption_graph(model, base, means)
    scores = weight_support(model, slot_of)
    wp = [(li, si) for li in scores for si in scores[li]]
    sup = [scores[li][si] for li, si in wp]
    cau = [dce[li][si] for li, si in wp]
    eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
    top_sup = set(np.argsort(sup)[::-1][:10].tolist())
    top_cau = set(np.argsort(cau)[::-1][:10].tolist())
    totals = {}
    for j in range(DEPTH):
        totals[str(j)] = round(sum(dce[li].get(si, 0.0) for li in dce
                                   for si in (1 + 2 * j, 2 + 2 * j)
                                   if si in dce[li]), 5)
    pairs = sorted([(li, si, dce[li][si]) for li in dce for si in dce[li]],
                   key=lambda p: -p[2])
    r = {'base_ce_fp32_abl_oldheld': round(base, 5),
         'wiring_n_pairs': len(wp),
         'wiring_spearman_all': round(R2.spearman(sup, cau), 4),
         'wiring_n_effectual': len(eff),
         'wiring_spearman_effectual': round(R2.spearman(
             [sup[k] for k in eff], [cau[k] for k in eff]), 4),
         'wiring_top10_precision': len(top_sup & top_cau) / 10.0,
         'per_block_total_consumption': totals,
         'dead_blocks_below_0.001': [j for j in totals if totals[j] < 0.001],
         'consumption_top20': [
             {'consumer': ('readout' if li == DEPTH else f'block{li}'),
              'source': R2.stream_name(si), 'dce': round(v, 5)}
             for li, si, v in pairs[:20]],
         'consumption_matrix': {str(li): {str(si): round(v, 6)
                                          for si, v in dce[li].items()}
                                for li in dce},
         'weight_support_matrix': {str(li): {str(si): round(v, 3)
                                             for si, v in scores[li].items()}
                                   for li in scores}}
    print(f"light probe: Spearman all {r['wiring_spearman_all']} "
          f"effectual({len(eff)}) {r['wiring_spearman_effectual']} "
          f"top10 {r['wiring_top10_precision']}", flush=True)
    return r


def tok_probe(model):
    """Token-determined + linear-in-embedding standard probes (qk_v8_probe)."""
    return P8.standard_probes_guarded(model)


def probe_arm(stem, factory, jp, light_key, tok_key=None, slot_of=None):
    """Idempotent probe block for a trained arm."""
    if SMOKE:
        return
    out = loadj(jp)
    need_light = light_key not in out
    need_tok = tok_key is not None and tok_key not in out
    if not (need_light or need_tok):
        print(f"{stem}: probes already done -- skip", flush=True)
        return
    m, _ = load_arm(stem, factory)
    if need_light:
        merge(jp, light_key, light_probe(m, slot_of))
    if need_tok:
        merge(jp, tok_key, tok_probe(m))
    del m
    torch.cuda.empty_cache()
