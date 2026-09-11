"""Feature relabeling can change top-k-plus-LS when Lasso codes have zero ties."""
import itertools,json
from pathlib import Path
import torch
from lasso_reader_encoding_v1 import encode
from quadratic_token_dictionary_v1 import conditional
from two_support_oracle_v1 import encode as oracle

@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    x=torch.tensor([[.8,.4,.3,.2]]);x/=x.norm()
    rows=[];raw_outputs=[];oracle_outputs=[]
    for permutation in itertools.permutations(range(4)):
        order=torch.tensor(permutation);b=torch.eye(4)[order]
        raw,report=conditional(torch.zeros(1,4),b@b.T,x@b.T,.5,'codes',5000,1e-9)
        ids,values,encoding=encode(b,x,k=2,penalty=.5)
        after=torch.einsum('nk,nkd->nd',values,b[ids])
        exact,exact_report=oracle(b.numpy(),x.numpy())
        raw_outputs.append(raw@b);oracle_outputs.append(torch.from_numpy(exact)@b)
        rows.append(dict(permutation=list(permutation),support_original_ids=order[ids[0]].tolist(),
            selected_raw_codes=raw.gather(1,ids).flatten().tolist(),
            lasso_converged=report['converged'],maximum_lasso_kkt=encoding['maximum_code_kkt'],
            support_normal=encoding['support_ls_normal_residual'],capture=1-float((after-x).square().sum()),
            oracle_capture=exact_report['capture']))
    raw_error=max(float((r-raw_outputs[0]).abs().max()) for r in raw_outputs)
    oracle_error=max(float((r-oracle_outputs[0]).abs().max()) for r in oracle_outputs)
    low=min(r['capture'] for r in rows);high=max(r['capture'] for r in rows)
    result=dict(predictions=dict(
        pred_a_lasso_invariance=raw_error<=1e-12 and all(r['lasso_converged'] and r['maximum_lasso_kkt']<=1e-8 for r in rows),
        pred_b_zero_padding=all(any(abs(v)<=1e-14 for v in r['selected_raw_codes']) for r in rows),
        pred_c_capture_variation=high-low>=.1,pred_d_oracle_invariance=oracle_error<=1e-12),
        raw_lasso_physical_error=raw_error,oracle_physical_error=oracle_error,
        minimum_capture=low,maximum_capture=high,capture_range=high-low,rows=rows,
        scope='Synthetic exact dictionary permutation, same two-term price. Reveals zero-tie padding artifact in existing encoder; native incidence/magnitude not measured. Live L1 source and encoding unchanged.')
    with Path(__file__).with_name('LASSO_PADDING_PERMUTATION_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
