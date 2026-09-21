"""Reader-rank lower bounds for local Jacobian fidelity at fixed anchors.
For Fhat(x)=g(B^T x), every Jacobian row lies in span(B). For summed teacher
Gram G=sum_a J_a^T J_a, rank-r loss is at least sum_{i>r}lambda_i(G).
The unconstrained derivative projection need not integrate to an allowed g.
"""
import torch


def analyze(grams,ranks,thresholds=(.1,.05,.01)):
    bases=[];records=[]
    for panel,g in enumerate(grams):
        e,q=torch.linalg.eigh((g+g.T)/2);e=e.flip(0).clamp_min(0);q=q.flip(1);total=e.sum();tails=torch.cat([e.flip(0).cumsum(0).flip(0),e.new_zeros(1)])/total
        floor={str(r):float(tails[r].sqrt()) for r in ranks}
        needed={str(t):int(torch.nonzero(tails<=t*t)[0,0]) for t in thresholds}
        records.append(dict(panel=panel,floors=floor,necessary_reader_counts=needed,trace=float(total)));bases.append(q)
    transfer=[]
    for source,q in enumerate(bases):
        for target,g in enumerate(grams):
            if source==target:continue
            errors={str(r):float(((g.trace()-(q[:,:r]*(g@q[:,:r])).sum())/g.trace()).clamp_min(0).sqrt()) for r in ranks}
            transfer.append(dict(source_panel=source,target_panel=target,frozen_basis_errors=errors))
    return dict(panels=records,basis_transfer=transfer),bases


def controls():
    torch.manual_seed(5000);torch.set_default_dtype(torch.float64)
    j=torch.randn(70,35);rot=torch.linalg.qr(torch.randn(35,35)).Q;grams=[j.T@j,rot.T@(j.T@j)@rot];result,q=analyze(grams,[8,16,32]);err=[]
    for i,g in enumerate(grams):
        jj=j if i==0 else j@rot
        for rank in [8,16,32]:
            basis=q[i][:,:rank];actual=float((jj-(jj@basis)@basis.T).norm()/jj.norm());err.append(abs(actual-result['panels'][i]['floors'][str(rank)]))
    assert max(err)<1e-12
    return dict(maximum_projection_error=max(err),panels=2,input_width=35,ranks=[8,16,32])
if __name__=='__main__':
    import json
    from pathlib import Path
    result=controls();Path(__file__).with_name('QUARTIC_READER_RANK_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
