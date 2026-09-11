"""Metric-corrected thin SVD distinguishes branch rank from consumer count.

A: exact/truncated CP and executor identities <=1e-9.
B: >=ceil(n/2) parents require >=2 branches for95% coefficient energy.
C: >=ceil(n/4) parents have sigma2/sigma1>=.5.
No semantic or behavioral interpretation of the canonical branches is implied.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from shared_parent_intervention_v1 import banks, group_parent, value, concatenate
from structured_branch_amplitudes_v1 import inner


def node_factors(graph, parent):
    u = graph['readers'][parent]
    u = u/u.norm()
    writers, partners = [], []
    for index, group in enumerate(graph['groups']):
        if parent not in group['parent_ids'].tolist():
            continue
        a, b, w = group_parent(graph, index, parent)
        c = group['writer']
        weights = (w.T@c)/(c@c)
        au, bu = a@u, b@u
        partner = (weights[:, None]*(bu[:, None]*a+au[:, None]*b)).sum(0)
        partner -= (weights*au*bu).sum()*u
        writers.append(c)
        partners.append(partner)
    c, p = torch.stack(writers, 1), torch.stack(partners)
    transformed = p + (math.sqrt(2)-1)*(p@u)[:, None]*u[None]
    qo, ro = torch.linalg.qr(c, mode='reduced')
    qi, ri = torch.linalg.qr(transformed.T, mode='reduced')
    left, singular, right = torch.linalg.svd(ro@ri.T, full_matrices=False)
    output = (qo@left)*singular
    partner = qi@right.T
    partner += (2**-.5-1)*u[:, None]*(u@partner)[None]
    return u, output, partner, singular


def main(label):
    assert label in ('spectral', 'native')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    root = Path(__file__).parent
    source = root/f'SHARED_READER_JOINT_FIT_V1_{label.upper()}_GRAPH.pt'
    out = root/f'SHARED_NODE_CANONICAL_BRANCHES_V1_{label.upper()}.json'
    artifact = root/f'SHARED_NODE_CANONICAL_BRANCHES_V1_{label.upper()}.pt'
    assert not out.exists() and not artifact.exists()
    graph = torch.load(source, weights_only=True, map_location='cpu')
    nodes = banks(graph)[2]
    torch.manual_seed(4201)
    x = torch.randn(13, graph['readers'].shape[1])
    rows, factors, checks = [], [], []
    for parent, node in enumerate(nodes):
        u, c, p, s = node_factors(graph, parent)
        bank = (u.expand(len(s), -1), p.T, c)
        energy = inner(node, node)
        replay = float((value(node, x)-value(bank, x)).norm()/value(node, x).norm())
        norm_error = float(abs(.5*s.square().sum()-energy)/energy)
        checks.extend([replay, norm_error])
        rank_rows = []
        for rank in range(1, len(s)+1):
            reduced = (bank[0][:rank], bank[1][:rank], bank[2][:, :rank])
            error = concatenate(node, (reduced[0], reduced[1], -reduced[2]))
            measured = inner(error, error)/energy
            predicted = s[rank:].square().sum()/s.square().sum()
            identity = float(abs(measured-predicted))
            checks.append(identity)
            rank_rows.append(dict(rank=rank, capture=float(1-predicted),
                                  squared_relative_error=float(measured), identity_error=identity))
        rank95 = next(r['rank'] for r in rank_rows if r['capture'] >= .95)
        rows.append(dict(parent=parent, consumers=len(s), rank95=rank95,
                         second_over_first=float(s[1]/s[0]) if len(s)>1 else 0.,
                         rank_results=rank_rows, executor_replay=replay, energy_replay=norm_error,
                         exact_standalone_floats=u.numel()+c.numel()+p.numel(),
                         exact_variable_products=len(s)))
        factors.append(dict(reader=u, writers=c, partners=p, singular_values=s))
    need_two = sum(r['rank95']>=2 for r in rows)
    strong = sum(r['second_over_first']>=.5 for r in rows)
    torch.save(dict(nodes=factors, output_whitener=graph['output_whitener']), artifact)
    result = dict(pred_a=max(checks)<=1e-9, pred_b=need_two>=math.ceil(len(rows)/2),
                  pred_c=strong>=math.ceil(len(rows)/4), label=label,
                  nodes_needing_two_at_95=need_two, nodes_with_strong_second_branch=strong,
                  maximum_replay=max(checks), rows=rows,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  artifact=dict(path=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                  scope='Exact standalone node factorization; independently executing all node-removal '
                        'banks double-counts shared-pair terms. It is not a replacement of the whole DAG. '
                        'Common background/U and adapters remain; branches have no behavioral labels.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))
    print(json.dumps([dict(parent=r['parent'], consumers=r['consumers'],rank95=r['rank95'],
                          rank1_capture=r['rank_results'][0]['capture'],ratio=r['second_over_first']) for r in rows], indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
