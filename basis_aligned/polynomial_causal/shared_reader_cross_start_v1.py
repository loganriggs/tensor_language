"""Score the pre-frozen shared-node correspondence after both fits finish.

A existing native/spectral executor audits <=1e-9.
B >=4 fixed matches have signed function cosine >=.8 and reader abs cosine >=.95.
C >=4 such matches have >=2 effective consumers on both sides.
Later signed/absolute rematching is descriptive only.
"""
import hashlib
import json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from shared_parent_intervention_v1 import banks
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    root = Path(__file__).parent
    output = root/'SHARED_READER_CROSS_START_V1_RESULT.json'
    assert not output.exists()
    frozen_path = root/'SHARED_READER_CROSS_START_MATCHING_V1_FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    graphs, audits, sources = [], [], {}
    for label in ('SPECTRAL','NATIVE'):
        path = root/f'SHARED_READER_JOINT_FIT_V1_{label}_GRAPH.pt'
        receipt = json.loads(path.with_suffix('.json').read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == receipt['artifact']['sha256']
        sources[str(path)] = digest
        graphs.append(torch.load(path, weights_only=True, map_location='cpu'))
        audits.append(json.loads((root/f'SHARED_READER_POSTFIT_INTERFACE_V1_{label}.json').read_text()))
    left, right = [banks(graph)[2] for graph in graphs]
    energies = [torch.stack([inner(node,node) for node in nodes]) for nodes in (left,right)]
    cross = torch.stack([torch.stack([inner(a,b) for b in right]) for a in left])
    cosines = cross/(energies[0][:,None]*energies[1][None]).sqrt()
    readers = [F['readers']/F['readers'].norm(dim=1,keepdim=True) for F in graphs]
    reader_cosines = (readers[0]@readers[1].T).abs()
    def record(i,j):
        cosine, reader = float(cosines[i,j]), float(reader_cosines[i,j])
        counts = [audits[0]['parents'][i]['consumers_above_one_percent'],
                  audits[1]['parents'][j]['consumers_above_one_percent']]
        return dict(spectral_parent=int(i), native_parent=int(j), function_cosine=cosine,
            reader_abs_cosine=reader, effective_consumer_counts=counts,
            stable=cosine>=.8 and reader>=.95,
            stable_and_reused=cosine>=.8 and reader>=.95 and min(counts)>=2)
    rows = [record(r['spectral_parent'],r['native_parent']) for r in frozen['matching']]
    stable = sum(r['stable'] for r in rows)
    reused = sum(r['stable_and_reused'] for r in rows)
    descriptive = {}
    for label, matrix in [('signed',cosines),('absolute',cosines.abs())]:
        ii,jj = linear_sum_assignment(-matrix.numpy())
        descriptive[label] = [record(int(i),int(j)) for i,j in zip(ii,jj)]
    valid = all(a['pred_a'] and a['maximum_executor_replay']<=1e-9 for a in audits)
    result = dict(pred_a=valid, pred_b=valid and stable>=4, pred_c=valid and reused>=4,
        fixed_stable_matches=stable, fixed_stable_reused_matches=reused, fixed_matches=rows,
        descriptive_rematching=descriptive, function_cosines=cosines.tolist(),
        reader_abs_cosines=reader_cosines.tolist(), sources=sources,
        frozen_matching_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        scope='Frozen correspondence of fitted approximate graph-node removals; '
              'different topologies and unconverged fits limit interpretation. No behavioral identification.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('function_cosines','reader_abs_cosines','descriptive_rematching','sources')},indent=2))


if __name__ == '__main__': main()
