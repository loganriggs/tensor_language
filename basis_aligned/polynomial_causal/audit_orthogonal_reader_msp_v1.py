"""CPU planted recovery for the full-dimensional shared-reader dictionary.
Registered on AGENT_BOARD before execution. No native or corpus data.
"""
import hashlib
import json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from orthogonal_reader_msp_v1 import fit, polar


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(657)
    d, n = 24, 4096
    truth = polar(torch.randn(d,d))
    coefficients = torch.randn(d,n)*(torch.rand(d,n)<.15)
    y = truth @ coefficients
    reports = []
    for seed in (0,937,1902):
        a, result = fit(y,seed,max_steps=2000,max_seconds=60)
        similarities = (a@truth).abs()
        rows,cols = linear_sum_assignment(-similarities.numpy())
        matched = similarities[rows,cols]
        result.update(seed=seed,mean_matched_cosine=float(matched.mean()),minimum_matched_cosine=float(matched.min()))
        reports.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    output=dict(predictions={
        'pred_a_orthogonal_monotone':all(r['orthogonality_error']<=1e-10 for r in reports),
        'pred_b_all_converged':all(r['converged'] for r in reports),
        'pred_c_dictionary_recovery':all(r['mean_matched_cosine']>=.99 for r in reports)},
        starts=reports,dimension=d,signals=n,sparsity_probability=.15,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('orthogonal_reader_msp_v1.py'))},
        scope='Synthetic sparse signal recovery only; finite native weight vectors need separate controls.')
    with Path(__file__).with_name('ORTHOGONAL_READER_MSP_V1_CONTROL.json').open('x') as f:
        json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output['predictions']))


if __name__=='__main__':
    main()
