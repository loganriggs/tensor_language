"""Exact-metric sparse paired product blocks for K(Jz,y), weights only."""
from pathlib import Path
import json,time,signal
import torch
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    signal.alarm(240);torch.set_num_threads(2);tic=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    J=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['mixed_map'].double()
    L,R,D=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    A,B=L@J,R@J;m,d=L.shape
    gram=(A@A.T)*(R@R.T)+(B@B.T)*(L@L.T)
    cross=(A@B.T)*(R@L.T);gram+=cross+cross.T
    gram*=D.T@D
    ref=json.loads((P/'INTERACTION_INPUT_MODE_COMPRESSION_V1_RESULT.json').read_text())['total_tensor_squared_norm']
    total=float(gram.sum());assert abs(total-ref)/ref<1e-10
    scales=gram.diag().clamp_min(1e-300).sqrt()
    kernel=gram/scales[:,None]/scales[None,:]
    target=gram.sum(1)/scales
    residual=target.clone();diagonal=kernel.diag().clone()
    columns=torch.zeros(m,512,dtype=torch.float64);chosen=[];captures=[];captured=0.;cells=[]

    def refit(indices):
        idx=torch.tensor(indices);K=kernel[idx][:,idx];b=target[idx]
        alpha=torch.linalg.solve(K,b)
        error=total-2*float(alpha@b)+float(alpha@K@alpha)
        return dict(relative_error=(max(0,error)/total)**.5,
                    stationarity=float((K@alpha-b).norm()/b.norm()),
                    weights=(alpha/scales[idx]).tolist())

    for step in range(512):
        score=residual.square()/diagonal.clamp_min(1e-14)
        score[diagonal<1e-12]=-1
        if chosen:score[torch.tensor(chosen)]=-1
        pivot=int(score.argmax());assert score[pivot]>=0
        column=kernel[:,pivot].clone()
        if step:column-=columns[:,:step]@columns[pivot,:step]
        column/=diagonal[pivot].sqrt()
        amplitude=float(residual[pivot]/diagonal[pivot].sqrt())
        captured+=amplitude**2;residual-=column*amplitude
        diagonal=(diagonal-column.square()).clamp_min(0)
        columns[:,step]=column;chosen.append(pivot);captures.append(captured)
        if step+1 in (16,32,64,128,256,512):
            fitted=refit(chosen)
            reference_indices=torch.argsort(scales,descending=True)[:step+1].tolist()
            baseline=refit(reference_indices)
            incremental_error=(max(0,total-captured)/total)**.5
            assert abs(incremental_error-fitted['relative_error'])<1e-8
            cells.append(dict(blocks=step+1,scalar_products=2*(step+1),
                              selected_weight_scalars=J.numel()+(3*d+1)*(step+1),
                              original_weight_scalars=J.numel()+3*d*m,
                              relative_error=fitted['relative_error'],
                              coefficient_stationarity=fitted['stationarity'],
                              topnorm_support_refit_error=baseline['relative_error']))
            print(json.dumps(cells[-1]),flush=True)
    fitted=refit(chosen)
    offdiag=kernel.abs().clone();offdiag.fill_diagonal_(0)
    row_sums=torch.topk(offdiag,511,dim=1).values.cumsum(1)
    correlations=torch.sort(target.square(),descending=True).values.cumsum(0)
    bounds=[]
    for size in (16,32,64,128,256,512):
        lower=float(kernel.diag().min()-row_sums[:,size-2].max())
        capture_upper=float(correlations[size-1])/lower if lower>0 else None
        bounds.append(dict(blocks=size,restricted_gram_eigenvalue_lower=lower,
                           any_support_relative_error_lower=(max(0,1-capture_upper/total))**.5 if capture_upper is not None else None))
    result=dict(any_support_bounds=bounds,cells=cells,indices=chosen,coefficients=fitted['weights'],
                gram_total_relative_check=abs(total-ref)/ref,seconds=time.perf_counter()-tic,
                scope='Weights-only exact coefficient metric for paired native-channel blocks in K(Jz,y). '
                'Greedy orthogonal least squares maximizes next exact residual reduction; support search '
                'is not globally optimal. Coefficient refits are solved linear systems, not evidence '
                'against learned readers, alternative blocks or DAG topology. J is shared once; '
                'selected L/R/D rows plus coefficients charged. Normalization/other response terms '
                'and native behavior are outside this restricted operator screen.')
    (P/'INTERACTION_PRODUCT_NODES_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print('seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
