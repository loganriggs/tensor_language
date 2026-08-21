"""MLP17 CLASS DIRECTIONS -- the SUPERVISED version, after 617 showed
the unsupervised PCA basis of mlp17's output is not interpretable. For
each token class, find the output direction that best predicts it, and
test whether those supervised readout directions are low-rank,
mutually distinct, and capture the readout.

617 lesson: PCA orders output directions by variance/magnitude, not by
what discriminates token classes, so the top-8 PCA directions are not
interpretable. The functional/readout directions must be found with
supervision. For each token class C, the class-readout direction is
    dC = mean(output | next token in C) - mean(output | not in C)
(the direction mlp17's output moves along when the next token is in
class C). This is the empirical readout direction for C.

Tests:
  DISTINCTNESS: are the class directions mutually distinct (low
    pairwise cosine), i.e. does mlp17 use a separate readout channel
    per class, or are they collinear (one shared magnitude axis)?
  RANK: do the ~10 class directions span a low-rank subspace (their
    own SVD), and does that subspace match mlp17's rank-8 output?
  READOUT VALIDITY: does projecting a position's output onto dC
    actually predict whether the next token is in C, out of sample?

REGISTERED PREDICTIONS:
  (0) POPULATED: >= 6 token classes have >= 50 positions each -- VOIDS
      on failure;
  (a) READOUT VALIDITY: for the newline class (the cleanest readout,
      615), the projection onto d_newline separates newline-target
      from other positions with AUC >= 0.8 out of sample (fit on one
      data half, test on the other) -- the supervised direction is a
      real readout, unlike the PCA directions;
  (b) DISTINCTNESS: the class directions are mutually more distinct
      than random directions -- median pairwise |cosine| among the
      class directions is LOWER than among matched random directions?
      No -- classes overlap; instead report the median pairwise
      |cosine| and whether any pair is near-collinear (|cos|>0.8,
      a shared axis);
  (c) EFFECTIVE RANK: the class directions' own effective rank (how
      many independent channels the ~10 classes use). Report it
      against mlp17's rank-8 output;
  NULL: projecting onto a RANDOM direction separates newline-target
      positions at AUC ~0.5 (chance) -- the supervised direction's
      AUC is real, not what any direction achieves."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
LJ = 17
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_class_directions_results.json'
NFRESH = 96

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}


def fine_class(s):
    t = s.strip().lower()
    if not t:
        return 'space'
    if t in DET:
        return 'determiner'
    if t in PREP:
        return 'preposition'
    if t in PRON:
        return 'pronoun'
    if t[0].isdigit():
        return 'digit'
    if all(not c.isalnum() for c in t):
        return 'punct'
    if s.strip()[:1].isupper():
        return 'capitalized'
    if s.startswith(' '):
        return 'space_word'
    return 'subword'


def classify(t):
    s = cl.d1(int(t))
    if chr(10) in s:
        return 'newline'
    return fine_class(s)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos = labels == 1
    npos = pos.sum()
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return (ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    mlp = m.transformer.h[LJ].mlp

    cap = []
    hk = mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0)                   # (Npos, D)
    Npos = O.shape[0]
    nxt = fresh[:, 1:257].reshape(-1)
    cls = np.array([classify(t) for t in nxt.tolist()])

    classes = [c for c in set(cls.tolist())
               if (cls == c).sum() >= 50]
    print(f'{Npos} positions; {len(classes)} classes with >=50: {sorted(classes)}',
          flush=True)
    p0 = len(classes) >= 6
    if not p0:
        json.dump({'void': 'too few classes', 'classes': sorted(classes)},
                   open(OUT, 'w'), indent=1)
        return

    # train/test split
    rng = np.random.default_rng(0)
    idx = rng.permutation(Npos)
    tr, te = idx[:Npos // 2], idx[Npos // 2:]
    Otr, Ote = O[tr].numpy(), O[te].numpy()
    cls_tr, cls_te = cls[tr], cls[te]

    # class-readout directions (fit on train)
    gmean = Otr.mean(0)
    dirs = {}
    for c in classes:
        m_c = Otr[cls_tr == c].mean(0)
        d = m_c - gmean
        d = d / (np.linalg.norm(d) + 1e-9)
        dirs[c] = d

    # (a) newline readout validity, out of sample
    def cls_auc(c, direction, Om, clsm):
        s = Om @ direction
        lab = (clsm == c).astype(float)
        return auc(s, lab)
    aucs = {c: round(float(cls_auc(c, dirs[c], Ote, cls_te)), 3)
            for c in classes}
    nl_auc = aucs.get('newline')
    pa = nl_auc is not None and nl_auc >= 0.8
    print(f'(a) newline readout AUC out-of-sample {nl_auc}: '
          f'{"HELD" if pa else "FAILED"}', flush=True)
    print(f'    all class AUCs: {dict(sorted(aucs.items(), key=lambda kv:-kv[1]))}',
          flush=True)

    # (b) pairwise distinctness
    dl = list(classes)
    M = np.stack([dirs[c] for c in dl])     # (K, D)
    cosM = M @ M.T
    iu = np.triu_indices(len(dl), 1)
    med_cos = float(np.median(np.abs(cosM[iu])))
    near_collinear = [(dl[i], dl[j], round(float(cosM[i, j]), 2))
                      for i, j in zip(*iu) if abs(cosM[i, j]) > 0.8]
    print(f'(b) median pairwise |cosine| among class directions {med_cos:.3f}; '
          f'near-collinear pairs (>0.8): {near_collinear}', flush=True)

    # (c) effective rank of the class directions
    s = np.linalg.svd(M, compute_uv=False)
    p = s / s.sum()
    erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    print(f'(c) effective rank of {len(dl)} class directions: {erank:.2f} '
          f'(mlp17 output rank-8)', flush=True)

    # NULL: random direction newline AUC
    g = np.random.default_rng(1)
    rand_aucs = []
    for _ in range(5):
        r = g.standard_normal(D)
        r /= np.linalg.norm(r)
        rand_aucs.append(round(float(cls_auc('newline', r, Ote, cls_te)), 3))
    null_ok = nl_auc is not None and nl_auc > 0.7 and np.mean(
        [abs(a - 0.5) for a in rand_aucs]) < 0.15
    print(f'NULL: random-direction newline AUCs {rand_aucs} (~0.5): '
          f'{"ok" if null_ok else "CHECK"}', flush=True)

    out = {'n_positions': Npos, 'classes': sorted(classes), 'pred_0': bool(p0),
           'class_aucs': aucs, 'newline_auc': nl_auc, 'pred_a': bool(pa),
           'median_pairwise_cos': med_cos, 'near_collinear': near_collinear,
           'class_dirs_erank': erank, 'random_newline_aucs': rand_aucs,
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
