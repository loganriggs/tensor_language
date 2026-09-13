"""Joint error-budget comparison: compact complement + sparse interaction core.
A actual coefficient error and orthogonal partition replay<=1e-10.
B >=5%byte improvement over best entry-only baseline at any tolerance.
C a B-winning candidate removes>=10%core residual dimensions.
Fixed budgets0/.1/.25/.5/.75/1, tolerances2/5/10%, native/output-head frames.
No learned frame, behavioral, runtime or whole-model claim.
"""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);start=time.perf_counter();t,ids=build()
    matrix=t.movedim(1,0).reshape(1152,-1).contiguous();total=float(t.square().sum())
    eig,q=torch.linalg.eigh(matrix@matrix.T);cumulative=eig.clamp_min(0).cumsum(0)
    bases=[]
    for axis in (0,2):
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1)
        bases.append(torch.linalg.eigh(flat@flat.T)[1].flip(1))
    rows=[];max_replay=0.
    for tol in (.02,.05,.1):
        for share in (0.,.1,.25,.5,.75,1.):
            k=int(torch.searchsorted(cumulative,share*tol**2*total,right=True))
            if k:
                packed,tau=torch.geqrf(q[:,:k].contiguous())
                rotated=torch.ormqr(packed,tau,matrix,left=True,transpose=True)
            else:rotated=matrix
            dropped=float(rotated[:k].square().sum());core=rotated[k:].reshape(1152-k,12,128)
            for frame in ('native','output_head'):
                x=core if frame=='native' else torch.einsum('op,ioa,ac->ipc',bases[0],core,bases[1])
                values,order=x.flatten().square().sort(descending=True)
                keep=min(x.numel(),int(torch.searchsorted(values.cumsum(0),(1-tol**2)*total))+1)
                sparse=torch.zeros_like(x).flatten();sparse[order[:keep]]=x.flatten()[order[:keep]];sparse=sparse.reshape_as(x)
                error_prediction=((dropped+float((x-sparse).square().sum()))/total)**.5
                recovered=sparse if frame=='native' else torch.einsum('op,ipc,ac->ioa',bases[0],sparse,bases[1])
                padded=torch.zeros_like(matrix);padded[k:]=recovered.reshape(1152-k,-1)
                replay=torch.ormqr(packed,tau,padded,left=True,transpose=False) if k else padded
                actual=float((replay-matrix).norm()/matrix.norm())
                diff=abs(actual-error_prediction);max_replay=max(max_replay,diff)
                assert diff<=1e-10 and actual<=tol+1e-10
                reflectors=1152*k-k*(k-1)//2
                other=0 if frame=='native' else sum(b.numel() for b in bases)
                cost=4*(reflectors+other+keep)+(x.numel()+7)//8
                rows.append(dict(tolerance=tol,projection_budget_fraction=share,frame=frame,discarded=k,
                                 retained=1152-k,entries=keep,adapter_scalars=reflectors+other,bytes=cost,
                                 total_error=actual,projection_error=(dropped/total)**.5))
    comparisons=[]
    for tol in (.02,.05,.1):
        eligible=[r for r in rows if r['tolerance']==tol]
        baseline=min((r for r in eligible if r['discarded']==0),key=lambda r:r['bytes'])
        combined=min((r for r in eligible if r['discarded']>0),key=lambda r:r['bytes'])
        comparisons.append(dict(tolerance=tol,entry_only=baseline,best_nonzero_projection=combined,
                                relative_bytes_gain=1-combined['bytes']/baseline['bytes']))
    winners=[c for c in comparisons if c['relative_bytes_gain']>=.05]
    result=dict(pred_a=True,pred_b=bool(winners),pred_c=any(c['best_nonzero_projection']['discarded']>=.1*1152 for c in winners),
                max_error_partition_difference=max_replay,comparisons=comparisons,cells=rows,seconds=time.perf_counter()-start,
                scope='Fixed weight-only spectral complement and exact core-entry supports. Errors partition orthogonally, one global coefficient budget. Reflector and frame adapters plus bitmap priced; no arbitrary dense residual adapter. Core-node reduction still reads all native input coordinates. No native behavior/runtime/global topology-optimum claim.')
    (P/'INTERACTION_COMPLEMENT_SPARSE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':main()
