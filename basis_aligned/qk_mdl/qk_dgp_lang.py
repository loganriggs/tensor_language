"""PLANTED-MODULAR-CONTENT DGP -- shared infrastructure (aggregate list #4;
predictions registered in qk_dgp_modular_predictions.json BEFORE this code was
written).  Experiments 5 and 6 import this module; keep it dependency-light.

THE GENERATIVE LAW (precise; seed 2026 fixes every table)
  Vocabulary V = 512.  Ground-truth tables:
    * selection class  c(t) = t // 64                      (8 classes, 64 tokens each)
    * content units    UNITS in R^{64 x 24}, 24 EXACTLY orthonormal columns
                       (QR of a seeded Gaussian).  "Orthonormal-ish" in the brief
                       is sharpened to exact orthonormality so that
                       <b(t), b(u)> equals the integer bundle-overlap count.
    * bundles          chi(t) in {0,1}^24 with |chi(t)| in {2, 3} (fair coin),
                       all 512 bundles distinct (rejection sampling);
                       b(t) = UNITS @ chi(t) in R^64.
    * permutation      sigma(k) = (k + 3) mod 8  -- a fixed DERANGEMENT of the
                       8 classes (no fixed point), so a token never attends to
                       its own class occurrence at its own position.
    * planted embedding (the semi-planted arm freezes wte to this table)
                       E(t) = [ onehot(c(t))  | dims 0..7
                                b(t)          | dims 8..71   (content block)
                                1.0           | dim 72       (always-on coordinate)
                                tail(t)       | dims 73..127 (iid N(0, 0.1^2)) ]
    * unigram floor    b_w = 0.5 * standardize_w( <r0, tail(w)> ), r0 a seeded
                       Gaussian over the tail coordinates -- AFFINE in E(w) by
                       construction, hence exactly representable by the tied
                       readout (a constant stream direction).

  Sequences (T = 128): t_0 ~ softmax(b).  For i >= 0, with k = c(t_i) and
      A_i = sum_{j <= i} 1[c(t_j) = sigma(k)] * chi(t_j)   in Z^24
  (per-unit counts over ALL earlier tokens of the attended class; j = i never
  contributes because sigma is a derangement):
      p(t_{i+1} = w | prefix) = softmax_w( GAMMA * <A_i, chi(w)> + b_w ),
  GAMMA = 1.  <A_i, chi(w)> is the bundle-overlap count summed over the
  attended positions.  When no attended-class token exists yet, A_i = 0 and
  the law IS the unigram floor softmax(b).

STATED DEVIATIONS from the brief's sketch (each with its reason):
  1. Selection is the SUM over all earlier tokens of class sigma(k), not the
     most recent one.  The family's pattern is an unnormalized no-softmax
     product of bilinear scores: it computes sums over selected positions and
     cannot compute an argmax/most-recent.  The sum IS the architecture's
     native selection operation; modularity is unchanged (the law still
     factors as [who: a class predicate] x [what: a 24-unit content code]).
  2. The unigram smoothing floor is an IN-SOFTMAX per-token bias b_w, not an
     epsilon-mixture outside the softmax.  An outside mixture makes the
     log-law non-affine in the sufficient statistics, so no softmax-readout
     model could be exactly calibrated to it; the in-softmax floor plays the
     same role (the no-context distribution is exactly softmax(b)) and all
     logits are bounded, so there is no zero-probability pathology to smooth.
  3. An always-on coordinate (dim 72) is added to the frozen layout: it gives
     pattern branch 2 a constant score, the MLP access to linear-in-stream
     terms, and the readout the b_w bias direction.  One coordinate.
  4. Units exactly orthonormal (see above).

WHY THE LAW IS REPRESENTABLE BY ONE BLOCK OF THIS FAMILY (honest statement).
  One block computes the law's sufficient statistics exactly up to two
  quantified small terms:
    * pattern: head h serves query-class h.  Branch-1 q/k vectors live in the
      SLOWEST rotary plane (frequency 10000^(-7/8) ~ 3.2e-4; phase drift over
      T = 128 is <= 0.041 rad, i.e. relative score drift <= 8.2e-4), with
      q1(t) an indicator of c(t) = h and k1(u) an indicator of
      c(u) = sigma(h); branch-2 is constant via the always-on coordinate.
      The product pattern equals alpha * 1[c(key) = sigma(c(query))] up to a
      relative error <= 1.7e-3.  EXACT position-invariance is impossible in
      this family -- rotary has no zero-frequency subspace -- and this bounded
      drift is the honest residue.
    * value/readout: v reads the content block, so attention accumulates
      alpha * UNITS @ A_i into the stream; the tied planted readout then gives
      <stream content part, E(w)> proportional to <A_i, chi(w)>, and a constant
      stream direction gives b_w.  The one remaining family constraint is the
      readout RMSNorm's per-position temperature, which is
      1 + O((signal / always-on component)^2) -- second order, shrinkable by
      scaling.  The OPERATIONAL representability check is the task-learned
      gate: trained held-out CE within 0.05 nats of the law's own log-loss.

WHY CONTENT PROVABLY FACTORS INTO THE 24 UNITS.  The conditional law depends
on the prefix ONLY through (c(t_i), A_i), and A_i is the per-unit count
vector -- a linear image of the attended bundles in the unit basis.  The
token-side dependence is <A_i, chi(w)>: a rank-24 bilinear interaction whose
factors ARE the planted units (chi has full column rank 24 over the
vocabulary).  Any model of the law must carry the 24 unit-count statistics,
and no finer content structure exists (bundles are the only token content).

MODEL: we REUSE tf_model.TinyBilin (vanilla variant, depth 1, width 128,
head_dim 16 -> 8 heads, hidden 512, T = 128, vocab 512) rather than
re-implementing.  Reason: reuse guarantees the trained object is the exact
architecture family of the tiny-models program (rotary, per-head qk RMSNorm,
no-softmax product-bilinear pattern, ungated bilinear MLP, tied readout,
30*tanh logit cap), with the fold machinery (fold_layer0_qk / fold_mlp /
fold_forward) and its validated identity gates available for free; a fresh
implementation risks silent drift from the family.

TRAINER: verbatim tiny-models defaults (tf_train.py): Muon lr 0.02 on 2D
hidden matrices (nesterov momentum 0.95, 5-step Newton-Schulz in bf16,
sqrt(max/min-dim) scaling), AdamW lr 0.004 betas (0.9, 0.95) weight-decay 0.1
on the tied embedding (decay group) and 0.0 on sub-2D params, warmup 250 then
cosine to 0, grad clip 1.0, batch 16, bf16 autocast on CUDA, fresh
single-epoch data (iid sampled sequences, one sequential pass, generator seed
stored).  Held evaluation for the task gate runs in full fp32 (no autocast):
the gate margin is 0.05 nats and bf16 evaluation noise is not free.
"""
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/tiny_full_interp')
import tf_model as TFM  # noqa: E402

# ---------------------------------------------------------------- constants
V = 512
NCLASS = 8
NUNITS = 24
DCONTENT = 64
WS = 128                 # width = stream width (vanilla) = 8 heads x 16
T = 128
GAMMA = 1.0
BW_STD = 0.5
TAIL_STD = 0.1
CONST_COORD = 72
CONTENT_SLICE = slice(8, 8 + DCONTENT)          # dims 8..71
CLASS_SLICE = slice(0, NCLASS)                  # dims 0..7
TAIL_SLICE = slice(73, WS)                      # dims 73..127
TABLE_SEED = 2026


class DGPTables:
    """All ground-truth tables, deterministic in TABLE_SEED (CPU float32).

    TWO VARIANTS (identical in everything except the bundle table CHI):
      * 'overlap' (the original REALISM ARM): bundles of 2-3 units, all 512
        distinct, drawn at random.  Its exact third moment is NOT a pure
        rank-24 object (co-occurrence cross-moments): the best rank-24 nonneg
        CP matches the units at mean cosine ~0.947 (23/24 at >= 0.9, one
        Perron-like mixture) -- calibration finding F0, measured, and the
        noiseless ceiling that variant is scored against.
      * 'identifiable' (the IDENTIFIABILITY ARM, added per the 2026-08-10
        amendment): bundle sizes 1-2 -- 75% singletons (units assigned
        round-robin) and 25% pairs, each specific pair used AT MOST ONCE
        (128 pairs of the 276 possible), singleton/pair status interleaved by
        t mod 4 so bundle size is EXACTLY balanced across the 8 selection
        classes (64 mod 4 == 0).  STATED TRADES vs the overlap variant:
        smaller bundles (overlap counts in {0,1,2} instead of {0..3}) and
        bundles are no longer all distinct (24 + 128 distinct bundle types
        over 512 tokens -- synonym tokens).  MEASURED acceptance (the
        analytic control re-checks it every run): the frozen CP machinery on
        the exact third moment matches the planted units at mean cosine
        0.993, min 0.990, 24/24 at >= 0.99."""

    def __init__(self, seed=TABLE_SEED, variant='overlap'):
        assert variant in ('overlap', 'identifiable')
        g = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.variant = variant
        # 24 exactly-orthonormal content units in R^64
        Q, _ = torch.linalg.qr(torch.randn(DCONTENT, NUNITS, generator=g))
        self.UNITS = Q.contiguous()                       # (64, 24)
        # classes and derangement
        self.CLS = torch.arange(V) // (V // NCLASS)       # (V,) long
        self.SIGMA = (torch.arange(NCLASS) + 3) % NCLASS  # derangement
        chi = torch.zeros(V, NUNITS)
        if variant == 'overlap':
            # distinct bundles of 2-3 units
            seen = set()
            for t in range(V):
                while True:
                    sz = 2 + int(torch.randint(0, 2, (1,), generator=g))
                    sup = tuple(sorted(
                        torch.randperm(NUNITS, generator=g)[:sz].tolist()))
                    if sup not in seen:
                        break
                seen.add(sup)
                chi[t, list(sup)] = 1.0
        else:
            # sizes 1-2, pair-once, class-balanced (see class docstring)
            pairs = [(a, b) for a in range(NUNITS) for b in range(a + 1, NUNITS)]
            perm = torch.randperm(len(pairs), generator=g)
            n_single = 0
            n_pair = 0
            for t in range(V):
                if t % 4 != 3:                            # 3 of 4: singleton
                    chi[t, n_single % NUNITS] = 1.0
                    n_single += 1
                else:                                     # 1 of 4: a fresh pair
                    a, b = pairs[perm[n_pair % len(pairs)]]
                    chi[t, a] = chi[t, b] = 1.0
                    n_pair += 1
        self.CHI = chi                                    # (V, 24) 0/1
        self.B = chi @ self.UNITS.T                       # (V, 64) content bundles
        # planted embedding table
        E = torch.zeros(V, WS)
        E[torch.arange(V), self.CLS] = 1.0
        E[:, CONTENT_SLICE] = self.B
        E[:, CONST_COORD] = 1.0
        tail = torch.randn(V, WS - 73, generator=g) * TAIL_STD
        E[:, TAIL_SLICE] = tail
        self.E = E                                        # (V, 128)
        # unigram floor bias: affine in the tail (hence in E) by construction
        r0 = torch.randn(WS - 73, generator=g)
        z = tail @ r0
        self.b = BW_STD * (z - z.mean()) / z.std()        # (V,)

    def step_logits(self, S, t):
        """Law logits for the NEXT token given running per-class unit counts
        S (B, 8, 24) that already include position i's token t (B,)."""
        A = S[torch.arange(len(t)), self.SIGMA[self.CLS[t]]]      # (B, 24)
        return GAMMA * (A @ self.CHI.T) + self.b                  # (B, V)


@torch.no_grad()
def sample_seqs(tabs, n, seed, T_=T, batch=4096):
    """iid sequences from the law.  CPU, deterministic in `seed` (single
    generator consumed sequentially across batches).  Returns (n, T) long."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for start in range(0, n, batch):
        Bn = min(batch, n - start)
        seq = torch.empty(Bn, T_, dtype=torch.long)
        S = torch.zeros(Bn, NCLASS, NUNITS)
        p0 = torch.softmax(tabs.b, 0).expand(Bn, V)
        t = torch.multinomial(p0, 1, generator=g).squeeze(1)
        seq[:, 0] = t
        ar = torch.arange(Bn)
        S[ar, tabs.CLS[t]] += tabs.CHI[t]
        for i in range(1, T_):
            p = torch.softmax(tabs.step_logits(S, t), 1)
            t = torch.multinomial(p, 1, generator=g).squeeze(1)
            seq[:, i] = t
            S[ar, tabs.CLS[t]] += tabs.CHI[t]
        out.append(seq)
    return torch.cat(out)


@torch.no_grad()
def true_scores(tabs, seqs, batch=2048):
    """Exact per-position law scores on given sequences.
    Returns (logloss, entropy), each (n, T-1): position i scores the
    prediction of seqs[:, i+1] from the prefix -- the same targets a model
    scores when fed seqs[:, :-1].  Entropy is EXACT per prefix (the Monte
    Carlo is only over the prefix distribution)."""
    lls, ents = [], []
    for start in range(0, len(seqs), batch):
        sq = seqs[start:start + batch]
        Bn, T_ = sq.shape
        ar = torch.arange(Bn)
        S = torch.zeros(Bn, NCLASS, NUNITS)
        ll = torch.empty(Bn, T_ - 1)
        en = torch.empty(Bn, T_ - 1)
        for i in range(T_ - 1):
            t = sq[:, i]
            S[ar, tabs.CLS[t]] += tabs.CHI[t]
            logp = torch.log_softmax(tabs.step_logits(S, t), 1)
            ll[:, i] = -logp[ar, sq[:, i + 1]]
            en[:, i] = -(logp.exp() * logp).sum(1)
        lls.append(ll)
        ents.append(en)
    return torch.cat(lls), torch.cat(ents)


def entropy_reference(tabs, held):
    """Task-gate references on the held set: mean exact conditional entropy
    (the DGP entropy estimate, MC over prefixes) and the law's own mean
    log-loss on the SAME tokens the model is scored on (the paired, tighter
    reference)."""
    ll, en = true_scores(tabs, held)
    return {'entropy_mc_mean': float(en.mean()),
            'entropy_by_pos_first8': [round(float(x), 4) for x in en.mean(0)[:8]],
            'entropy_by_pos_last4': [round(float(x), 4) for x in en.mean(0)[-4:]],
            'true_law_logloss_mean': float(ll.mean()),
            'n_seqs': int(held.shape[0])}


# ---------------------------------------------------------------- optimizer
# Muon + param split: VERBATIM ports of ns_orth / Muon / muon_params_split in
# ../tiny_full_interp/tf_train.py (themselves transcribed from
# ../qk_mdl/qk_e_common.py) -- copied rather than imported because tf_train
# executes corpus I/O paths at import time.
def ns_orth(G, steps=5):
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
                assert p.dim() == 2, 'Muon is for 2D matrices only'
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
                p.add_(u, alpha=-group['lr'] * math.sqrt(max(r, cdim)
                                                         / min(r, cdim)))


def muon_params_split(model):
    """tf_train.muon_params_split, plus a requires_grad filter so the frozen
    planted embedding is simply absent from every optimizer group."""
    excl = tuple(getattr(model, 'muon_exclude', ()))
    mu, dec, nod = [], [], []
    for nm, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if excl and any(nm.startswith(x) or f'.{x}' in nm for x in excl):
            nod.append(p)
        elif p.dim() == 2 and not nm.startswith('wte'):
            mu.append(p)
        elif p.dim() >= 2:
            dec.append(p)
        else:
            nod.append(p)
    return mu, dec, nod


# ---------------------------------------------------------------- model/arms
def make_arm_model(tabs, arm, seed=0, device='cpu'):
    """arm 'semi': wte FROZEN to the planted table.  arm 'learned': free wte
    at the family's standard init (normal std 0.02)."""
    assert arm in ('semi', 'learned')
    cfg = TFM.TFConfig(depth=1, width=WS, vocab=V, tok='bpe', seed=seed,
                       variant='vanilla', T=T)
    model = TFM.make_model(cfg, device)
    if arm == 'semi':
        with torch.no_grad():
            model.wte.weight.copy_(tabs.E.to(device))
        model.wte.weight.requires_grad_(False)
    return model


# tiny-models trainer defaults (tf_train.py), stated and reused verbatim
MUON_LR = 0.02
ADAMW_LR = 0.004
WD = 0.1
WARMUP = 250
GRAD_CLIP = 1.0
BATCH = 16


def train_arm(model, train_seqs, held_seqs, steps, dev, log_every=200,
              name='arm'):
    """Single sequential pass over iid `train_seqs` (fresh single-epoch by
    construction); mirrors tf_train.train_cell.  Returns the log dict."""
    assert len(train_seqs) >= steps * BATCH, 'not enough fresh sequences'
    mu, dec, nod = muon_params_split(model)
    opt_m = Muon(mu, lr=MUON_LR)
    groups = [{'params': dec, 'weight_decay': WD},
              {'params': nod, 'weight_decay': 0.0}]
    groups = [gr for gr in groups if gr['params']]
    opt_a = torch.optim.AdamW(groups, lr=ADAMW_LR, betas=(0.9, 0.95)) \
        if groups else None

    def fac(step):
        if step < WARMUP:
            return (step + 1) / WARMUP
        p = (step - WARMUP) / max(1, steps - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))

    log = {'lr_muon': MUON_LR, 'lr_adamw': ADAMW_LR, 'steps': steps,
           'batch': BATCH, 'warmup': WARMUP, 'grad_clip': GRAD_CLIP,
           'weight_decay': WD, 'train_curve_every%d' % log_every: [],
           'held100_curve': [], 'spikes': 0, 'diverged': False}
    run, t0 = None, time.time()
    model.train()
    for step in range(steps):
        f = fac(step)
        for g in opt_m.param_groups:
            g['lr'] = MUON_LR * f
        if opt_a is not None:
            for g in opt_a.param_groups:
                g['lr'] = ADAMW_LR * f
        seqs = train_seqs[step * BATCH:(step + 1) * BATCH].to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
            logits = model(seqs[:, :-1])
        ce = F.cross_entropy(logits.float().reshape(-1, V),
                             seqs[:, 1:].reshape(-1))
        l = ce.item()
        if not math.isfinite(l) or l > 30:
            log['diverged'], log['diverged_at'] = True, step
            break
        opt_m.zero_grad(set_to_none=True)
        if opt_a is not None:
            opt_a.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt_m.step()
        if opt_a is not None:
            opt_a.step()
        run = l if run is None else 0.98 * run + 0.02 * l
        if l > (run or l) + 1.0:
            log['spikes'] += 1
        if step % log_every == 0:
            log['train_curve_every%d' % log_every].append(
                [step, round(l, 4), round(run, 4)])
            print(f'  {name} step {step}/{steps} ce {l:.4f} (ema {run:.4f}) '
                  f'{time.time() - t0:.0f}s', flush=True)
        if step > 0 and step % max(1, steps // 8) == 0:
            hce = eval_ce(model, held_seqs[:100], dev)
            log['held100_curve'].append([step, round(hce, 4)])
            model.train()
    log['wall_seconds'] = round(time.time() - t0, 1)
    if not log['diverged']:
        log['final_held_ce'] = round(eval_ce(model, held_seqs, dev), 5)
    else:
        log['final_held_ce'] = float('inf')
    return log


@torch.no_grad()
def eval_ce(model, seqs, dev, batch=32):
    """Full-fp32 held CE (no autocast): the task gate margin is 0.05 nats."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(seqs), batch):
        b = seqs[i:i + batch].to(dev)
        logits = model(b[:, :-1])
        ce = F.cross_entropy(logits.float().reshape(-1, V),
                             b[:, 1:].reshape(-1), reduction='sum')
        tot += float(ce)
        n += b[:, 1:].numel()
    return tot / n
