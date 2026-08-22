"""CAUSAL CAPSTONE: grammar steers STRONGLY, the mirror of content/topic's WEAK steering (§868) — same method,
apples-to-apples. The two-machine claim says grammar is low-rank and strongly causal while content is
high-dimensional and only weakly steerable by one direction. §868 showed topic steering gives a small (+0.042)
specific gain. Here run the IDENTICAL steering protocol on the GRAMMATICAL CLASS at a front layer and show it
is much STRONGER, confirming the asymmetry on one consistent pipeline.

Method: build each class's mean-deviation direction from a front layer's residual (block READ_L output, where
class is written, §851), add it (alpha sweep) at that layer, and measure the logit gain on that class's tokens
vs other classes' tokens (diagonal vs off-diagonal of the steer x class matrix). Controls: shuffled-class-label
steering (null); off-diagonal is the within-experiment specificity control; direct comparison to §868 topic
(own-gain 0.042).

REGISTERED PREDICTIONS:
  (0) SANITY: classes have coherent token sets; shuffled-class steering ~ 0 gain;
  (a) GRAMMAR STEERS STRONGLY + SPECIFICALLY: mean own-class logit gain is LARGE (>> the §868 topic 0.042, and
      >> off-diagonal) -> a single class direction moves prediction hard, confirming grammar is low-rank and
      strongly causal, the mirror image of the weakly-steerable high-dimensional topic;
  (b) report the diagonal/off-diagonal and the grammar-vs-topic ratio plainly."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_steer_results.json'
NEVAL = 220; SEQ = 256; READ_L = 2; ALPHAS = [8.0, 16.0, 32.0]; NDISTINCT = 40
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
ST = {'on': False, 'vec': None, 'alpha': 0.0}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def mk_hook():
    def hook(mo, i_, o_):
        if not ST['on']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D)
        v2 = v + ST['alpha'] * ST['vec'].to(v.dtype)
        return (v2.reshape(sh),) + tuple(o_[1:]) if isinstance(o_, tuple) else v2.reshape(sh)
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_resid(rows):
    outs = []; seqs = []
    def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[READ_L].register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :SEQ].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hh.remove()
    return torch.cat(outs, 0), np.concatenate(seqs, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    R, S = capture_resid(rows); toks = S.reshape(-1); T = S.shape[1]
    tgt = np.full_like(S, -1); tgt[:, :-1] = S[:, 1:]; tgt = tgt.reshape(-1)
    clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    g = R.mean(0, keepdim=True)
    # class-mean-deviation steering directions (at the current token position's residual)
    dirs = {};
    for c in range(len(CLASSES)):
        mk = clslab == c
        if mk.sum() < 20: continue
        v = R[torch.tensor(mk, device=DEV)].mean(0) - g[0]; dirs[c] = v / (v.norm() + 1e-9)
    # distinctive NEXT tokens per class (what a class predicts next) for the logit readout
    base = Counter(tgt[tgt >= 0]); Nn = int((tgt >= 0).sum()); dtok = {}
    for c in dirs:
        mk = clslab == c; nc = Counter(tgt[mk][tgt[mk] >= 0]); nj = int((tgt[mk] >= 0).sum()); sc = []
        for t, cc in nc.items():
            if cc < 4: continue
            sc.append(((cc/nj)/((base.get(t, 0)+1)/Nn), t))
        sc.sort(reverse=True); dtok[c] = [t for _, t in sc[:NDISTINCT]]
    cids = [c for c in dirs if dtok.get(c)]
    idxb = rows[:32, :SEQ].to(DEV)[:, :-1].contiguous()
    hh = m.transformer.h[READ_L].register_forward_hook(mk_hook())
    ST['on'] = False; base_logits = forward_logits(idxb).float().reshape(-1, int(m.lm_head.weight.shape[0]))
    def mean_logit_on(lg, ts): return float(lg[:, torch.tensor(ts, device=DEV)].mean())
    best = None
    for A in ALPHAS:
        mat = {}
        for a in cids:
            ST['on'] = True; ST['vec'] = dirs[a]; ST['alpha'] = A
            lg = forward_logits(idxb).float().reshape(-1, base_logits.shape[-1]); ST['on'] = False
            mat[a] = {b: round(mean_logit_on(lg, dtok[b]) - mean_logit_on(base_logits, dtok[b]), 3) for b in cids}
        dg = float(np.mean([mat[a][a] for a in cids])); og = float(np.mean([mat[a][b] for a in cids for b in cids if b != a]))
        print(f"alpha {A}: mean diagonal (own-class gain) {dg:.3f} | off-diag {og:.3f}", flush=True)
        if best is None or (dg - og) > best[3]: best = (A, mat, dg, dg - og, og)
    # shuffled-class null at best alpha
    A, mat, dg, _, og = best
    rng = np.random.RandomState(0); sh = clslab.copy(); rng.shuffle(sh)
    shdirs = {}
    for c in range(len(CLASSES)):
        mk = sh == c
        if mk.sum() < 20: continue
        v = R[torch.tensor(mk, device=DEV)].mean(0) - g[0]; shdirs[c] = v/(v.norm()+1e-9)
    shgain = []
    for a in cids:
        if a not in shdirs: continue
        ST['on'] = True; ST['vec'] = shdirs[a]; ST['alpha'] = A
        lg = forward_logits(idxb).float().reshape(-1, base_logits.shape[-1]); ST['on'] = False
        shgain.append(mean_logit_on(lg, dtok[a]) - mean_logit_on(base_logits, dtok[a]))
    hh.remove()
    out = {'read_layer': READ_L, 'best_alpha': A, 'n_classes': len(cids),
           'class_names': {c: CLASSES[c] for c in cids},
           'mean_diagonal': round(dg, 3), 'mean_offdiag': round(og, 3),
           'shuffled_class_null_gain': round(float(np.mean(shgain)), 3) if shgain else None,
           'topic_reference_own_gain_868': 0.042,
           'grammar_over_topic_ratio': round(dg / 0.042, 1),
           'matrix': {str(a): {str(b): mat[a][b] for b in cids} for a in cids}, 'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_grammar_strong'] = bool(dg > og + 0.3 and dg > 5 * 0.042 and (out['shuffled_class_null_gain'] is None or dg > 3 * abs(out['shuffled_class_null_gain'])))
    json.dump(out, open(OUT, 'w'), indent=1)
    for a in cids: print(f"steer->{CLASSES[a]:>6}: own-class gain {mat[a][a]} | mean other {round(np.mean([mat[a][b] for b in cids if b!=a]),3)}", flush=True)
    print(f"\nGRAMMAR mean diagonal {out['mean_diagonal']} vs off-diag {out['mean_offdiag']} | shuffled-class null {out['shuffled_class_null_gain']}", flush=True)
    print(f"grammar own-gain {out['mean_diagonal']} vs TOPIC own-gain 0.042 (§868) -> {out['grammar_over_topic_ratio']}x", flush=True)
    print(f"(a) grammar steers strongly + specifically: {out['pred_a_grammar_strong']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
