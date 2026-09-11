"""Reuse registered parent-intervention checks on a completed shared graph.

A executor/CP <=1e-9; B at least one shared-pair correction >=1% joint energy;
C every parent has at least two consumers with removal energy >=1% group energy.
Before/after function agreement and matched capture/cost are descriptive here;
the four-arm aggregate keeps its original protocol and remains pending.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

from shared_parent_intervention_v1_audit import audit
from shared_parent_intervention_v1 import banks
from structured_branch_amplitudes_v1 import inner
from ll1_joint_parent_graph_v2 import factors
from ll1_group_matching_v1 import group_inner


def main(label):
    assert label in ('spectral', 'native')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    root = Path(__file__).parent
    stem = f'SHARED_READER_JOINT_FIT_V1_{label.upper()}'
    receipt = json.loads((root/f'{stem}_GRAPH.json').read_text())
    original_receipt = json.loads((root/f'{stem}_ORIGINAL.json').read_text())
    path = Path(receipt['artifact']['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['artifact']['sha256']
    graph = torch.load(path, weights_only=True, map_location='cpu')
    before_path = root/'LL1_JOINT_CORE_SOLVE_V1_GRAPHS.pt'
    before = torch.load(before_path, weights_only=True, map_location='cpu')[label]
    original = torch.load(original_receipt['artifact']['path'], weights_only=True, map_location='cpu')
    result = audit(graph, label)
    first, second = banks(before)[2], banks(graph)[2]
    assert len(first) == len(second)
    rows = []
    for i, (old, new) in enumerate(zip(first, second)):
        eo, en, cross = inner(old, old), inner(new, new), inner(old, new)
        reader_cosine = float(abs(before['readers'][i]@graph['readers'][i]) /
                              before['readers'][i].norm()/graph['readers'][i].norm())
        rows.append(dict(parent=i, reader_abs_cosine=reader_cosine,
                         removal_function_cosine=float(cross/(eo*en).sqrt()),
                         relative_removal_function_change=float(((eo+en-2*cross)/eo).clamp_min(0).sqrt()),
                         removal_energy_ratio=float(en/eo)))
    a, b = factors(original), factors(graph)
    aa, bb, ab = group_inner(a, a), group_inner(b, b), group_inner(a, b)
    result.update(parent_changes=rows,
                  whole_function_cosine_to_matched_original=float(ab.sum()/(aa.sum()*bb.sum()).sqrt()),
                  relative_function_difference_to_original=float(((aa.sum()+bb.sum()-2*ab.sum())/aa.sum()).clamp_min(0).sqrt()),
                  capture=receipt['capture'], original_capture=original_receipt['capture'],
                  capture_gap=original_receipt['capture']-receipt['capture'],
                  price=receipt['price'],
                  both_locally_converged=bool(receipt['pred_b'] and original_receipt['pred_b']),
                  sources=dict(fitted=receipt['artifact'], before_sha256=hashlib.sha256(before_path.read_bytes()).hexdigest()),
                  scope='Intervention accounting and reuse in a fitted approximate DAG; '
                        'before/after fitting is not independent-start identification. '
                        'No native behavioral, sufficiency, OOD, or selectivity claim.')
    out = root/f'SHARED_READER_POSTFIT_INTERFACE_V1_{label.upper()}.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('parents','pairs','sources')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
