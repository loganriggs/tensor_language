"""Noniterative finite-sample fourth-moment baseline on native and null weights."""
import json
from pathlib import Path
import time
import torch
from native_reader_msp_generalization_v1 import P,CK
from orthogonal_reader_msp_v1 import polar

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    started=time.perf_counter()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    native=torch.cat([sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right')])
    native/=native.norm(dim=1,keepdim=True)
    generator=torch.Generator().manual_seed(811)
    null=torch.randn(native.shape,generator=generator);null/=null.norm(dim=1,keepdim=True)
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))
    train=torch.cat((order[:3072],order[:3072]+4608))
    test=torch.cat((order[3072:],order[3072:]+4608))
    selected=torch.randperm(len(train),generator=torch.Generator().manual_seed(812))[:1152]
    rows=[]
    for name,y in [('native',native),('isotropic_gaussian',null)]:
        basis=polar(y[train[selected]])
        scores={}
        for split,ids in [('train',train),('test',test)]:
            z=y[ids]@basis.T
            scores[split]=dict(fourth_moment=float(z.pow(4).sum()/len(ids)),
                top128_capture=float(z.square().topk(128,dim=1).values.sum()/len(ids)))
        rows.append(dict(name=name,scores=scores,orthogonality_error=float((basis@basis.T-torch.eye(1152)).norm())))
    expected=3/(1152+2)
    predictions=dict(pred_a_orthogonal=all(row['orthogonality_error']<=1e-10 for row in rows),
        pred_b_quartic_gap=all(row['scores']['train']['fourth_moment']>=10*row['scores']['test']['fourth_moment'] for row in rows),
        pred_c_gaussian_exact=abs(rows[1]['scores']['test']['fourth_moment']/expected-1)<=.10,
        pred_d_sparse_overfit=rows[1]['scores']['train']['top128_capture']-rows[1]['scores']['test']['top128_capture']>=.05)
    result=dict(predictions=predictions,arms=rows,gaussian_expected_fourth_moment=expected,
        seconds=time.perf_counter()-started,iterative_fitting=False,
        scope='One sample-polar finite-weight baseline per matrix, not a distributional confidence interval or no-structure theorem')
    (P/'SAMPLE_POLAR_READER_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
