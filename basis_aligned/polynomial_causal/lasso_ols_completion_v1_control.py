"""Compare Schur OLS completion with existing OLS and feature permutations."""
import itertools,json
from pathlib import Path
import torch
from lasso_ols_completion_v1 import complete,encode
from shared_dictionary_ols_v1 import select

@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(817)
    b=torch.randn(16,40);x=torch.randn(40);gram=b@b.T;rhs=x@b.T;initial=[3,9]
    residual_gram=gram-gram[:,initial]@torch.linalg.solve(gram[initial][:,initial],gram[initial])
    residual_cross=rhs-rhs[initial]@torch.linalg.solve(gram[initial][:,initial],gram[initial])
    expected,gains=select(residual_gram,residual_cross[None],5)
    observed,actual_gains=complete(gram,rhs,initial,7)
    gain_error=max(abs(a-bb) for a,bb in zip(gains,actual_gains))
    target=torch.tensor([[.8,.4,.3,.2]]);target/=target.norm();functions=[];reports=[]
    for permutation in itertools.permutations(range(4)):
        basis=torch.eye(4)[torch.tensor(permutation)]
        ids,values,report=encode(basis,target,k=2,penalty=.5)
        functions.append(torch.einsum('nk,nkd->nd',values,basis[ids]));reports.append(report)
    variation=max(float((a-functions[0]).abs().max()) for a in functions)
    capture=1-float((functions[0]-target).square().sum())
    result=dict(predictions=dict(pred_a_existing_ols=observed==initial+expected and gain_error<=1e-10,
        pred_b_permutation=variation<=1e-12,pred_c_normal=all(r['support_ls_normal_residual']<=1e-10 for r in reports),
        pred_d_capture=abs(capture-80/93)<=1e-12),gain_error=gain_error,permutation_error=variation,capture=capture,
        scope='Greedy OLS completion fixes this zero-padding counterexample; no global sparse recovery claim.')
    with Path(__file__).with_name('LASSO_OLS_COMPLETION_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True);assert all(result['predictions'].values())

if __name__=='__main__':main()
