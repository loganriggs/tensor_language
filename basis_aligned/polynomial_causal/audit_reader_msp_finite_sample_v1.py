"""Sparse recovery and dense null with eight weight-like vectors/dimension."""
import hashlib
import json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from orthogonal_reader_msp_v1 import fit, polar, topk_energy


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(700)
    d = 24
    truth = polar(torch.randn(d,d))
    reports = []
    for name,probability in [('sparse',.15),('dense',1.)]:
        def draw(n):
            coefficients = torch.randn(d,n)*(torch.rand(d,n)<probability)
            y = truth@coefficients
            return y/y.norm(dim=0,keepdim=True).clamp_min(1e-30)
        train,test = draw(8*d),draw(4096)
        baseline = topk_energy(torch.eye(d),test,3)
        for seed in (0,937,1902):
            a,result = fit(train,seed,max_steps=2000,max_seconds=60)
            similarity = (a@truth).abs()
            i,j = linear_sum_assignment(-similarity.numpy())
            train_capture,test_capture = topk_energy(a,train,3),topk_energy(a,test,3)
            result.update(case=name,seed=seed,mean_matched_cosine=float(similarity[i,j].mean()),
                train_top3_energy=train_capture,test_top3_energy=test_capture,
                baseline_test_top3_energy=baseline,test_gain=test_capture-baseline,
                train_test_gap=train_capture-test_capture)
            reports.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    result = dict(predictions={
        'pred_a_all_converged':all(r['converged'] and r['orthogonality_error']<=1e-10 for r in reports),
        'pred_b_sparse_recovered':all(r['mean_matched_cosine']>=.95 for r in reports if r['case']=='sparse'),
        'pred_c_dense_no_transfer':all(r['test_gain']<=.02 for r in reports if r['case']=='dense'),
        'pred_d_dense_training_optimism':all(r['train_test_gap']>=.05 for r in reports if r['case']=='dense')},
        starts=reports,dimension=d,training_signals=8*d,test_signals=4096,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Synthetic normalized weight vectors; test vectors never fitted. No native or activation data.')
    with Path(__file__).with_name('READER_MSP_FINITE_SAMPLE_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['predictions']))


if __name__=='__main__':
    main()
