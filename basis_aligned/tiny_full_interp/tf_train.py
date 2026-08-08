"""Fresh single-epoch training for the tiny-full-interpretation grid, plus the
matched baselines that any cost quote depends on.

PROTOCOL (inherited from ../qk_mdl/AGENT_BRIEF.md, non-negotiable)
  - ONE pass over the train split.  Never a second visit to any sequence.
    15,000 steps x batch 16 = 240,000 sequences = the whole train split.
  - Data order epoch_order(0) with DATA_SEED 1234, IDENTICAL across every cell,
    so per-token differences between cells are paired.
  - Muon on the 2D hidden matrices, AdamW on the tied embedding and everything
    sub-2D (ns_orth / Muon / muon_params_split transcribed verbatim from
    ../qk_mdl/qk_e_common.py).  Predicate parameters are routed to AdamW-no-
    decay via the model's muon_exclude, exactly as E22 does.
  - MATCHED-OPTIMIZER BASELINES.  The parent program understated every tax by
    0.094 nats for a week by comparing Muon arms to an AdamW baseline; the
    bigram/unigram baselines here are optimizer-free (closed form), and the
    same-architecture comparisons are all Muon-vs-Muon.
  - The lr sweep runs on the SPARE split, never on train, so no model has ever
    touched a train sequence before its single epoch.
  - REGISTERED PREDICTIONS are written into the results JSON BEFORE training.

Baselines (GRID.md requires both before any cost can be quoted):
  unigram floor   -- best constant predictor fitted on train, scored on held
  bigram table    -- closed-form P(next|prev) with unigram-backoff Dirichlet
                     smoothing, alpha tuned on the estimation split, scored on
                     held.  CONTROL: the bigram must beat the unigram floor.

Usage
  python tf_train.py baselines [--vocab 4096]
  python tf_train.py cell --variant vanilla --depth 1 --width 32 --seed 0
  TF_SMOKE=1 python tf_train.py cell --depth 1 --width 32   # tiny CPU smoke
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import tf_corpus
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get('TF_SMOKE') == '1'
DEV = 'cpu' if (SMOKE or not torch.cuda.is_available()) else 'cuda'

STEPS = 15000
BATCH = 16
T = 512
DATA_SEED = 1234
WARMUP = 250
GRAD_CLIP = 1.0
WD = 0.1
MUON_LRS = (0.01, 0.02, 0.04)
ADAMW_LR = 0.004                 # the parent's family embedding lr
SWEEP_STEPS = 400
HELD_EVAL_N = 1500               # held sequences for the final paired eval
LOG_EVERY = 200

if SMOKE:
    STEPS, BATCH, T, SWEEP_STEPS, HELD_EVAL_N, WARMUP = 30, 2, 64, 4, 8, 2
    LOG_EVERY = 10

REGISTERED = {
    'written_before_training': True,
    'ladder_rungs_1_2': 'the exact fold and the materialized tables pass their '
                        'identity gates at EVERY width and depth in the grid; '
                        'if a rung fails it fails at the LARGEST width first',
    'width_effect': 'held CE improves monotonically with width at fixed depth, '
                    'with diminishing returns by width 128',
    'depth_effect': 'depth 2 beats depth 1 at every width, and the depth-2 gain '
                    'GROWS with width (composition needs capacity to be worth '
                    'having)',
    'baseline_gap': 'every trained cell beats the closed-form bigram table; the '
                    'width-32 depth-1 cell beats it by less than 0.5 nats',
    'attn_table_rank': 'each layer-0 V x V score table has numerical rank equal '
                       'to its algebraic bound (head_dim = 16) at delta = 0, so '
                       'no head wastes its key subspace',
    'seed_convergence': 'held CE across seeds at a fixed cell spreads by less '
                        'than 0.02 nats, but the materialized tables do NOT '
                        'match across seeds beyond a permutation',
}


# ---------------------------------------------------------------- Muon
# Transcribed VERBATIM from ../qk_mdl/qk_e_common.py (nanoGPT-speedrun
# convention: nesterov momentum 0.95, 5-step Newton-Schulz in bf16, scaled by
# sqrt(max(rows, cols) / min(rows, cols))).
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
    """Muon: 2D hidden matrices.  AdamW: tied embedding (decay) + sub-2D +
    anything the model lists in muon_exclude (predicate params: pred_prof is 3D
    and pred_b/pred_c are scalar mixtures -- orthogonalizing either is
    nonsense)."""
    excl = tuple(getattr(model, 'muon_exclude', ()))
    mu, dec, nod = [], [], []
    for nm, p in model.named_parameters():
        if excl and any(nm.startswith(x) or f'.{x}' in nm for x in excl):
            nod.append(p)
        elif p.dim() == 2 and not nm.startswith('wte'):
            mu.append(p)
        elif p.dim() >= 2:
            dec.append(p)
        else:
            nod.append(p)
    return mu, dec, nod


# ---------------------------------------------------------------- data
class Corpus:
    def __init__(self, vocab, tok='bpe'):
        self.V, self.tok = vocab, tok
        self.man = json.load(
            open(f'{tf_corpus.corpus_root(vocab, tok)}/MANIFEST.json'))
        need = STEPS * BATCH
        self.train = torch.from_numpy(
            tf_corpus.load_split(vocab, 'train', None if SMOKE else need,
                                 tok=tok)).to(DEV)
        self.held = torch.from_numpy(
            tf_corpus.load_split(vocab, 'held', HELD_EVAL_N, tok=tok)).to(DEV)
        self.ntr = len(self.train)

    def order(self, epoch=0):
        g = torch.Generator().manual_seed(DATA_SEED + epoch)
        return torch.randperm(self.ntr, generator=g)

    def unk_rates(self):
        s = self.man['splits_stats']
        unk_id = self.man.get('unk_id', 0)
        held_t = self.held[:, 1:T + 1]
        out = {'tokenizer': self.tok,
               'train_per_token': s['train']['unk_rate_per_token'],
               'held_per_token': s['held']['unk_rate_per_token'],
               'held_per_seq_mean': s['held']['unk_rate_per_seq_mean']}
        # tok='bpe' has no UNK symbol at all (unk_id -1); id 0 there is EOS, so
        # counting id-0 targets would silently report the EOS rate as an UNK rate.
        out['held_eval_target_unk_frac'] = (
            0.0 if unk_id is None or unk_id < 0
            else float((held_t == unk_id).float().mean()))
        if unk_id is not None and unk_id < 0:
            out['note'] = ('byte-level BPE: UNK impossible by construction; '
                           'id 0 is <|endoftext|>')
            out['held_eos_frac'] = float((held_t == 0).float().mean())
        return out


@torch.no_grad()
def eval_held(model, corpus, n_seq=None, per_token=False, batch=8):
    model.eval()
    hs = corpus.held if n_seq is None else corpus.held[:n_seq]
    tot, n, pts = 0.0, 0, []
    for i in range(0, len(hs), batch):
        b = hs[i:i + batch]
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            logits = model(b[:, :T])
        ce = F.cross_entropy(logits.float().reshape(-1, corpus.V),
                             b[:, 1:T + 1].reshape(-1), reduction='none')
        if per_token:
            pts.append(ce.cpu())
        tot += ce.sum().item()
        n += ce.numel()
    model.train()
    return tot / n, (torch.cat(pts).numpy() if per_token else None)


# ---------------------------------------------------------------- baselines
def baselines(vocab=8192, tok='bpe', force=False):
    """Closed-form unigram floor and bigram table.  No training, no optimizer,
    so they are matched by construction.  CONTROL: bigram must beat unigram."""
    tag = f'b{vocab}' if tok == 'bpe' else f'v{vocab}'
    jp = f'{HERE}/tf_baselines_{tag}.json'
    if os.path.exists(jp) and not force:
        print('baselines already computed -- skip', flush=True)
        return json.load(open(jp))
    V = vocab
    t0 = time.time()
    tr = tf_corpus.load_split(V, 'train', 20000 if SMOKE else None, tok=tok)
    est = tf_corpus.load_split(V, 'est', 2000 if SMOKE else None, tok=tok)
    hl = tf_corpus.load_split(V, 'held', 1000 if SMOKE else None, tok=tok)
    print(f'baselines: train {tr.shape} est {est.shape} held {hl.shape}',
          flush=True)

    # ---- unigram, fitted on the train split's TARGET positions ----
    uni = np.bincount(tr[:, 1:].reshape(-1), minlength=V).astype(np.float64)
    p_uni = (uni + 1.0) / (uni.sum() + V)            # Laplace, avoids -inf
    logp_uni = np.log(p_uni)
    ce_uni_held = float(-logp_uni[hl[:, 1:].reshape(-1)].mean())
    ce_uni_est = float(-logp_uni[est[:, 1:].reshape(-1)].mean())

    # ---- bigram counts on train ----
    prev = tr[:, :-1].reshape(-1)
    nxt = tr[:, 1:].reshape(-1)
    C = np.zeros((V, V), dtype=np.float32)
    step = 2_000_000
    for a in range(0, len(prev), step):
        np.add.at(C, (prev[a:a + step], nxt[a:a + step]), 1.0)
    rowsum = C.sum(1)

    def bigram_ce(data, alpha):
        """Dirichlet smoothing with unigram backoff:
        P(j|i) = (C[i,j] + alpha*P_uni[j]) / (rowsum[i] + alpha)."""
        p, n = 0.0, 0
        pv = data[:, :-1].reshape(-1)
        nv = data[:, 1:].reshape(-1)
        for a in range(0, len(pv), step):
            i, j = pv[a:a + step], nv[a:a + step]
            num = C[i, j] + alpha * p_uni[j]
            p += float(np.log(num / (rowsum[i] + alpha)).sum())
            n += len(i)
        return -p / n

    alphas = [0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
              10000.0]
    sweep = {str(a): round(bigram_ce(est, a), 5) for a in alphas}
    best_alpha = float(min(sweep, key=lambda k: sweep[k]))
    at_edge = best_alpha in (alphas[0], alphas[-1])
    if at_edge:
        print(f'WARNING: smoothing alpha {best_alpha} at a grid edge -- widen',
              flush=True)
    ce_bi_held = bigram_ce(hl, best_alpha)

    res = {'vocab': V, 'tokenizer': tok,
           'bits_per_byte_rule': 'these are nats PER TOKEN and are comparable '
                                 'only within this tokenizer; see '
                                 'tf_tokenizer_compare.json for the '
                                 'cross-tokenizer bits/byte table',
           'unigram_floor_held_ce': round(ce_uni_held, 5),
           'unigram_floor_est_ce': round(ce_uni_est, 5),
           'bigram_held_ce': round(ce_bi_held, 5),
           'bigram_alpha_sweep_on_est': sweep,
           'bigram_alpha_chosen': best_alpha,
           'bigram_alpha_at_grid_edge': bool(at_edge),
           'bigram_beats_unigram_nats': round(ce_uni_held - ce_bi_held, 5),
           'bigram_nonzero_cells': int((C > 0).sum()),
           'bigram_row_coverage': float((rowsum > 0).mean()),
           'fit': 'counts on the train split; alpha tuned on the estimation '
                  'split; scored on held.  Both are closed form -- no optimizer,'
                  ' so they are matched baselines by construction.',
           'seconds': round(time.time() - t0, 1)}
    # THE BITS/BYTE RULE, applied mechanically so no cross-tokenizer comparison
    # can be made from this file without it (README.md).
    res['bytes_per_token'] = round(tf_corpus.bytes_per_token(V, tok), 4)
    res['unigram_floor_bits_per_byte'] = round(
        tf_corpus.bits_per_byte(ce_uni_held, V, tok), 5)
    res['bigram_bits_per_byte'] = round(
        tf_corpus.bits_per_byte(ce_bi_held, V, tok), 5)
    rep = tf_corpus.unk_repair_bpb(V, tok)
    if rep:
        res['unk_repair_bits_per_byte'] = rep
        res['bigram_bits_per_byte_honest'] = round(
            res['bigram_bits_per_byte'] + rep, 5)
        res['lossy'] = True
    res['control_bigram_beats_unigram'] = bool(ce_bi_held < ce_uni_held)
    assert res['control_bigram_beats_unigram'], \
        f'CONTROL FAILED: bigram {ce_bi_held} did not beat unigram {ce_uni_held}'
    json.dump(res, open(jp, 'w'), indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k != 'bigram_alpha_sweep_on_est'}, indent=2), flush=True)
    return res


# ---------------------------------------------------------------- training
def train_cell(cfg, corpus, lr_muon=None, steps=STEPS, save=True,
               log_every=LOG_EVERY, stem=None):
    torch.cuda.reset_peak_memory_stats() if DEV == 'cuda' else None
    model = M.make_model(cfg, DEV)
    mu, dec, nod = muon_params_split(model)
    lr_muon = lr_muon if lr_muon is not None else 0.02
    opt_m = Muon(mu, lr=lr_muon)
    opt_a = torch.optim.AdamW([{'params': dec, 'weight_decay': WD},
                               {'params': nod, 'weight_decay': 0.0}],
                              lr=ADAMW_LR, betas=(0.9, 0.95))

    def fac(step):
        if step < WARMUP:
            return (step + 1) / WARMUP
        p = (step - WARMUP) / max(1, steps - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))

    log = {'lr_muon': lr_muon, 'lr_adamw': ADAMW_LR, 'steps': steps,
           'batch': BATCH, 'train_curve_every200': [], 'held100_curve': [],
           'spikes': 0, 'diverged': False}
    order = corpus.order(0)
    run, t0 = None, time.time()
    model.train()
    for step in range(steps):
        f = fac(step)
        for g in opt_m.param_groups:
            g['lr'] = lr_muon * f
        for g in opt_a.param_groups:
            g['lr'] = ADAMW_LR * f
        sl = order[step * BATCH:(step + 1) * BATCH]
        assert len(sl) == BATCH, 'train stream exhausted -- corpus too small'
        seqs = corpus.train[sl]
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            logits = model(seqs[:, :T])
        ce = F.cross_entropy(logits.float().reshape(-1, corpus.V),
                             seqs[:, 1:T + 1].reshape(-1))
        loss = ce + cfg.group_coeff * model.group_penalty() \
            if cfg.group_coeff > 0 else ce
        aux = model.pop_aux_loss()
        if aux is not None:
            loss = loss + aux
        l = ce.item()
        if not math.isfinite(l) or l > 30:
            log['diverged'], log['diverged_at'] = True, step
            print(f'  DIVERGED at step {step} (ce {l})', flush=True)
            break
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt_m.step()
        opt_a.step()
        run = l if run is None else 0.98 * run + 0.02 * l
        if l > run + 1.0:
            log['spikes'] += 1
        if step % log_every == 0:
            log['train_curve_every200'].append([step, round(l, 4), round(run, 4)])
            print(f'  {stem or cfg.stem()} step {step}/{steps} ce {l:.4f} '
                  f'(ema {run:.4f}) {time.time() - t0:.0f}s', flush=True)
        if step > 0 and step % max(1, steps // 8) == 0:
            hce, _ = eval_held(model, corpus, n_seq=100)
            log['held100_curve'].append([step, round(hce, 4)])
            print(f'  held100 CE {hce:.4f} at step {step}', flush=True)
    dt = time.time() - t0
    log['wall_seconds'] = round(dt, 1)
    log['sec_per_step'] = round(dt / max(1, steps), 5)
    log['peak_mib'] = (round(torch.cuda.max_memory_allocated() / 2 ** 20)
                       if DEV == 'cuda' else 0)
    if log['diverged']:
        log['final_held_ce'] = float('inf')
        return log, model
    hce, pt = eval_held(model, corpus, per_token=True)
    log['final_held_ce'] = round(hce, 5)
    log['n_params'] = model.n_params()
    log['body_params'] = model.body_params()
    print(f'== {stem or cfg.stem()} FINAL held CE {hce:.5f} '
          f'({dt:.0f}s, {log["spikes"]} spikes, {log["peak_mib"]} MiB)',
          flush=True)
    if save:
        st = stem or cfg.stem()
        torch.save({'state_dict': model.state_dict(),
                    'cfg': vars(cfg), 'log': log}, f'{HERE}/{st}.pt')
        np.save(f'{HERE}/{st}_heldloss.npy', pt)
    return log, model


def lr_sweep(cfg, corpus, jp):
    """Short Muon-lr sweep on the SPARE split (never train), cached in the
    cell JSON.  Winner at a grid edge -> widen (AGENT_BRIEF)."""
    out = json.load(open(jp)) if os.path.exists(jp) else {}
    if 'lrsweep' in out:
        return out['lrsweep']['chosen']
    spare = torch.from_numpy(
        tf_corpus.load_split(cfg.vocab, 'spare',
                             SWEEP_STEPS * BATCH * len(MUON_LRS),
                             tok=cfg.tok)).to(DEV)

    class SweepCorpus(Corpus):
        def __init__(s):
            s.V, s.man, s.tok = corpus.V, corpus.man, corpus.tok
            s.train, s.held = spare, corpus.held[:100]
            s.ntr = len(spare)

    sc = SweepCorpus()
    res = {}
    for lr in MUON_LRS:
        print(f'-- lr sweep {lr} x {SWEEP_STEPS} steps (spare split)', flush=True)
        lg, m = train_cell(cfg, sc, lr_muon=lr, steps=SWEEP_STEPS, save=False,
                           log_every=max(1, SWEEP_STEPS // 4),
                           stem=f'sweep_lr{lr}')
        res[str(lr)] = {'held100_ce': (None if lg['diverged']
                                       else lg['final_held_ce']),
                        'diverged': lg['diverged'], 'spikes': lg['spikes']}
        del m
        if DEV == 'cuda':
            torch.cuda.empty_cache()
    ok = {k: v for k, v in res.items() if v['held100_ce'] is not None}
    chosen = float(min(ok, key=lambda k: ok[k]['held100_ce'])) if ok \
        else float(MUON_LRS[1])
    at_edge = chosen in (MUON_LRS[0], MUON_LRS[-1])
    out['lrsweep'] = {'results': res, 'chosen': chosen, 'grid': list(MUON_LRS),
                      'winner_at_grid_edge': at_edge,
                      'split': 'spare (disjoint from train/held/est)'}
    json.dump(out, open(jp, 'w'), indent=2)
    if at_edge:
        print(f'WARNING: lr winner {chosen} at a grid edge -- widen the grid',
              flush=True)
    return chosen


def run_cell(variant='vanilla', depth=1, width=32, seed=0, vocab=8192,
             tok='bpe', do_sweep=True, slot=0, suffix='', n_slots=0,
             group_coeff=None, lr=None):
    """`slot` and `suffix` exist for MATCHED-PARAMETER CONTROLS only.

    solve_slot reinvests a small decoder's savings into a WIDER STREAM, and the
    embedding is V x stream_width, so `bandwidth`/`predicate`/`codebook` carry
    more embedding parameters than `vanilla`/`slots`/`shrink` at the same
    compute width.  Forcing `slot` to width/n_slots pins the stream (and hence
    the embedding) to the vanilla value, which separates the small-decoder
    mechanism from the extra parameters it buys.  Such a cell needs its own
    checkpoint name, hence `suffix`; the checkpoint stores the full cfg, so
    nothing downstream has to parse it back out of the stem."""
    kw = {} if group_coeff is None else {'group_coeff': group_coeff}
    cfg = M.TFConfig(depth=depth, width=width, vocab=vocab, tok=tok, seed=seed,
                     variant=variant, T=T, slot=slot, n_slots=n_slots, **kw)
    if cfg.small_dec and not cfg.slot:
        cfg.slot = M.solve_slot(cfg)[0]
    stem = cfg.stem() + suffix
    jp = f'{HERE}/{stem}.json'
    out = json.load(open(jp)) if os.path.exists(jp) else {}
    if out.get('run', {}).get('final_held_ce') is not None \
            and os.path.exists(f'{HERE}/{stem}.pt'):
        print(f'{stem}: already trained -- skip', flush=True)
        return out
    corpus = Corpus(vocab, tok)
    # registered predictions FIRST, before any training or analysis
    out.setdefault('registered_predictions', REGISTERED)
    out['config'] = dict(vars(cfg))
    out['config']['stream_width'] = cfg.stream_width
    out['config']['n_heads'] = cfg.n_heads
    out['config']['hidden'] = cfg.hidden
    out['corpus'] = {
        'vocab': vocab, 'tokenizer': tok, 'seq_len': T + 1, 'unk': corpus.unk_rates(),
        'train_split_rows': corpus.ntr, 'data_order_id': f'epoch_order(0)'
                                                         f'@seed{DATA_SEED}',
        'single_epoch_arithmetic': f'{STEPS} steps x batch {BATCH} = '
                                   f'{STEPS * BATCH} sequences = the entire '
                                   f'train split, each visited exactly once',
        'held': 'held split rows [0:%d] (pure eval, never trained)' % HELD_EVAL_N}
    json.dump(out, open(jp, 'w'), indent=2)
    if lr is None:
        lr = lr_sweep(cfg, corpus, jp) if (do_sweep and not SMOKE) else 0.02
    out = json.load(open(jp))
    log, model = train_cell(cfg, corpus, lr_muon=lr, steps=STEPS, stem=stem)
    out['run'] = log
    json.dump(out, open(jp, 'w'), indent=2)
    # baseline gap, if the baselines exist
    bp = f'{HERE}/tf_baselines_' + \
         (f'b{vocab}' if tok == 'bpe' else f'v{vocab}') + '.json'
    if os.path.exists(bp) and not log['diverged']:
        b = json.load(open(bp))
        out['vs_baselines'] = {
            'unigram_floor': b['unigram_floor_held_ce'],
            'bigram': b['bigram_held_ce'],
            'model_minus_bigram': round(log['final_held_ce']
                                        - b['bigram_held_ce'], 5),
            'model_minus_unigram': round(log['final_held_ce']
                                         - b['unigram_floor_held_ce'], 5),
            'model_bits_per_byte': round(
                tf_corpus.bits_per_byte(log['final_held_ce'], vocab, tok), 5),
            'bigram_bits_per_byte': b.get('bigram_bits_per_byte'),
            'unigram_floor_bits_per_byte':
                b.get('unigram_floor_bits_per_byte'),
            'bits_per_byte_rule': 'the ONLY quantity comparable across '
                                  'tokenizers; nats/token above are '
                                  'within-tokenizer only'}
        rep = tf_corpus.unk_repair_bpb(vocab, tok)
        if rep:
            out['vs_baselines']['unk_repair_bits_per_byte'] = rep
            out['vs_baselines']['model_bits_per_byte_honest'] = round(
                out['vs_baselines']['model_bits_per_byte'] + rep, 5)
        json.dump(out, open(jp, 'w'), indent=2)
    del model
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['baselines', 'cell'])
    ap.add_argument('--variant', default='vanilla')
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vocab', type=int, default=8192)
    ap.add_argument('--tok', default='bpe', choices=['bpe', 'trunc'],
                    help="'bpe' = trained byte-level BPE (primary, zero UNK); "
                         "'trunc' = top-K GPT-2 truncation (comparison arm)")
    ap.add_argument('--no-sweep', action='store_true')
    ap.add_argument('--slot', type=int, default=0,
                    help='force the slot width (matched-parameter controls only)')
    ap.add_argument('--suffix', default='',
                    help='checkpoint-name suffix for a matched-parameter control')
    ap.add_argument('--n-slots', type=int, default=0,
                    help='override the slot count (mechanism-decomposition arms)')
    ap.add_argument('--group-coeff', type=float, default=None,
                    help='override the in-loss group lasso coefficient')
    ap.add_argument('--lr', type=float, default=None,
                    help='force the Muon lr (learning-rate robustness arms)')
    a = ap.parse_args()
    if a.cmd == 'baselines':
        baselines(a.vocab, a.tok)
    else:
        run_cell(a.variant, a.depth, a.width, a.seed, a.vocab, a.tok,
                 do_sweep=not a.no_sweep, slot=a.slot, suffix=a.suffix,
                 n_slots=a.n_slots, group_coeff=a.group_coeff, lr=a.lr)
