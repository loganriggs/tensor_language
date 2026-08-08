"""Are the atoms an INTERPRETATION or just a code?

The parent program's section 4 answered this by dumping, for each atom, the
tokens whose folded rows use it most, and found the atoms semantic rather than
merely morphological.  Same test here, on the same object: the joint dictionary
over the token's whole folded query/key signature, learned under the ported
context-expected OV objective.

A code becomes an explanation only if the atoms name something.  This file
produces the evidence for or against, plus two quantitative summaries that do
not depend on reading the dump: (1) atom usage concentration, and (2) whether
an atom's token set is predictable from cheap surface features (leading space,
case, digits, punctuation, length) -- if it is, the atoms are morphological; if
it is not, they are either semantic or spectral.
"""
import argparse
import json
import os
import re

import numpy as np
import torch

import tf_corpus
import tf_dict_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
log = L.log


def tok_class(s):
    if s is None:
        return 'none'
    t = s
    lead = t.startswith(' ') or t.startswith('Ġ')
    core = t.replace('Ġ', ' ').strip()
    if core == '':
        return 'space/newline'
    if re.fullmatch(r'\d+', core):
        return 'digits'
    if not re.search(r'[A-Za-z]', core):
        return 'punct/symbol'
    if core[0].isupper():
        return ('lead-cap' if lead else 'cap')
    return ('lead-word' if lead else 'suffix')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='tf_vanilla_d1_w128_b8192_s0')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--show', type=int, default=24)
    a = ap.parse_args()

    D = L.FoldDesc(a.stem)
    X = L.build_X(D)
    q, cnt = L.unigram_q(D)
    Mv, Ms = L.ctx_metrics(D, q, verbose=False)
    Msel = L.metric_at(Mv, Ms, 16, blend=0.8)
    Dic, idx, coef = L.dict_learn(X, Msel, a.n, a.k, iters=a.iters)
    R = L.sparse_recon(Dic, idx, coef)
    s = D.score_fold(L.X_to_FT(D, R), n_seq=256)
    vocab = tf_corpus.load_vocab(D.V, tok=D.cfg.tok)
    dec = vocab['decoded']
    lab = [dec[i] if i < len(dec) else '' for i in range(D.V)]
    cls = [tok_class(t) for t in lab]

    use = torch.bincount(idx.reshape(-1), minlength=a.n).float()
    order = torch.argsort(use, descending=True)
    # purity: the most common surface class among an atom's top-32 users
    pur, ent = [], []
    for at in range(a.n):
        rows = (idx == at).any(1).nonzero().squeeze(1)
        if len(rows) == 0:
            continue
        w = torch.zeros(len(rows), device=X.device)
        for j in range(a.k):
            m = idx[rows, j] == at
            w[m] = coef[rows, j][m].abs()
        top = rows[torch.argsort(w, descending=True)[:32]]
        cs = [cls[int(t)] for t in top]
        p = max(cs.count(c) for c in set(cs)) / len(cs)
        pur.append(p)
    # a random-atom null for purity: same set sizes, tokens drawn at random
    g = np.random.default_rng(0)
    null = []
    for _ in range(len(pur)):
        cs = [cls[int(i)] for i in g.integers(0, D.V, 32)]
        null.append(max(cs.count(c) for c in set(cs)) / len(cs))

    out = {'stem': a.stem, 'n': a.n, 'k': a.k, 'score': s,
           'atom_usage_top10_share': float(use.sort(descending=True).values[:10]
                                           .sum() / use.sum()),
           'dead_atoms': int((use == 0).sum()),
           'surface_purity_mean': float(np.mean(pur)),
           'surface_purity_null_mean': float(np.mean(null))}
    lines = [f'# Atoms of the folded-signature dictionary '
             f'(n={a.n}, k={a.k}, context-OV objective)', '',
             f'Model `{a.stem}`, held CE {s["ce"]:.4f} '
             f'(model {s["ce_model"]:.4f}), KL {s["kl"]:.4f}.', '',
             f'Atom usage: top-10 atoms carry '
             f'{out["atom_usage_top10_share"]*100:.1f}% of all uses; '
             f'{out["dead_atoms"]} atoms are unused.', '',
             f'Surface-class purity of the top-32 users of an atom: '
             f'**{out["surface_purity_mean"]:.2f}** versus a random-token null '
             f'of {out["surface_purity_null_mean"]:.2f}.', '']
    for at in order[:a.show].tolist():
        rows = (idx == at).any(1).nonzero().squeeze(1)
        if len(rows) == 0:
            continue
        w = torch.zeros(len(rows), device=X.device)
        for j in range(a.k):
            m = idx[rows, j] == at
            w[m] = coef[rows, j][m].abs()
        top = rows[torch.argsort(w, descending=True)[:18]]
        toks = ' '.join(repr(lab[int(t)]) for t in top)
        lines.append(f'- **atom {at}** (used by {len(rows)} tokens): {toks}')
    open(f'{HERE}/tf_dict_atoms.md', 'w').write('\n'.join(lines) + '\n')
    json.dump(out, open(f'{HERE}/{a.stem}_dict_atoms.json', 'w'), indent=1)
    log('wrote tf_dict_atoms.md', out)


if __name__ == '__main__':
    main()
