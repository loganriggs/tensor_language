"""Marginal-energy private spaces include shared/private interactions.

V1 picks eigendirections of (I-P) Q (I-P). This variant instead picks leading
eigendirections of (I-P) Q^2 (I-P), then diagonalizes the retained private core.
The latter measures all outputs of Q on a private input, including shared ones.
"""
import torch
from ll1_joint_parent_graph_v1 import execute, factors, price


@torch.no_grad()
def build(a, s, c, readers, nodes):
    groups, pairs, conditions, memberships = [], [], [], []
    for g in range(len(a)):
        ids = [i for i, node in enumerate(nodes) if g in node['consumers']]
        k, rank = len(ids), a.shape[1]
        if k > rank:
            raise ValueError('More parents than rank')
        rr = readers[ids]
        if k:
            u, change = torch.linalg.qr(rr.T, mode='reduced')
            condition = float(torch.linalg.cond(change))
            if condition > 1e6:
                raise ValueError('Dependent shared readers')
            inv = torch.linalg.inv(change)
            residual_a = a[g] - (a[g] @ u) @ u.T
            oldspan = torch.linalg.qr(a[g].T, mode='reduced').Q
            membership = float(torch.linalg.svdvals(oldspan.T @ u).min().square())
        else:
            u = a.new_zeros(a.shape[-1], 0); inv = a.new_zeros(0, 0)
            residual_a = a[g]; condition = 1.; membership = 1.
        conditions.append(condition); memberships.append(membership)
        private_span, small_a = torch.linalg.qr(residual_a.T, mode='reduced')
        # Q^2 = A.T diag(s) (A A.T) diag(s) A; no dense d-by-d Q required.
        weighted_gram = s[g][:, None] * (a[g] @ a[g].T) * s[g][None]
        marginal = small_a @ weighted_gram @ small_a.T
        values, vectors = torch.linalg.eigh(marginal)
        selected = values.argsort(descending=True)[:rank-k]
        private = (private_span @ vectors[:, selected]).T
        av = a[g] @ private.T
        private_core = (av.T * s[g][None]) @ av
        lam, rotate = torch.linalg.eigh(private_core)
        private = rotate.T @ private
        au, av = a[g] @ u, a[g] @ private.T
        shared = inv @ ((au.T * s[g][None]) @ au) @ inv.T
        cross = inv @ ((au.T * s[g][None]) @ av)
        pair_ids, coeff = [], []
        for i in range(k):
            for j in range(i, k):
                pair = (ids[i], ids[j])
                if pair not in pairs:
                    pairs.append(pair)
                pair_ids.append(pairs.index(pair))
                coeff.append(shared[i, j]*(1 if i == j else 2))
        groups.append(dict(parent_ids=torch.tensor(ids, dtype=torch.long),
                           pair_ids=torch.tensor(pair_ids, dtype=torch.long),
                           shared_coeff=torch.stack(coeff) if coeff else a.new_zeros(0),
                           private=private, lam=lam, cross=2*cross, writer=c[g]))
    return dict(readers=readers, pairs=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
                groups=groups, parent_conditions=conditions, minimum_joint_memberships=memberships)
