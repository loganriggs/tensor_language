"""Exact pairwise fourth-power maximization with finite Givens circuit pricing."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent

def rotate_pair(x,y):
    xx=x.square();yy=y.square()
    c=(xx.square()+yy.square()-6*xx*yy).sum(-1)/4
    s=(x*y*(xx-yy)).sum(-1)
    angle=torch.atan2(s,c)/4
    co=angle.cos()[:,None];si=angle.sin()[:,None]
    return co*x+si*y,-si*x+co*y,angle

def main():
    torch.set_num_threads(2);start=time.perf_counter();gen=torch.Generator().manual_seed(61327)
    planted=torch.randn(8,64,generator=gen,dtype=torch.float64);theta=torch.linspace(-.6,.6,8)[:,None]
    x=planted*theta.cos();y=planted*theta.sin();a,b,_=rotate_pair(x,y)
    recovery=float(torch.minimum(a.square().sum(-1),b.square().sum(-1)).sum().sqrt()/planted.norm());assert recovery<1e-10
    t,ids=build();bases=[]
    for axis in [0,2]:
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1);bases.append(torch.linalg.eigh(flat@flat.T)[1])
    core=torch.einsum('op,oia,ab->pib',bases[0],t,bases[1])
    initial=core.movedim(1,0).reshape(1152,-1).contiguous()
    scale=initial.square().mean().sqrt();initial=initial/scale
    n=initial.numel();total=float(initial.square().sum());adapter=sum(q.numel() for q in bases)
    def score(x,rounds):
        cumulative=x.flatten().square().sort(descending=True).values.cumsum(0);cells=[]
        for tol in [.02,.05,.1]:
            k=int(torch.searchsorted(cumulative,(1-tol*tol)*total))+1
            price=4*adapter+4*k+(n+7)//8+12*576*rounds
            cells.append(dict(tolerance=tol,retained_entries=k,bytes=price,bytes_over_dense=price/(4*n)))
        return cells
    trajectories=[];maxnorm=0.;worstdecrease=0.
    for seed in [61327,61328,61329]:
        gen=torch.Generator().manual_seed(seed);x=initial.clone();history=[];checkpoints=[]
        objective=float(x.pow(4).sum());checkpoints.append(dict(round=0,objective=objective,curves=score(x,0)))
        for step in range(1,41):
            perm=torch.randperm(1152,generator=gen);left,right=perm[::2],perm[1::2]
            a,b,angles=rotate_pair(x[left],x[right]);x[left]=a;x[right]=b
            after=float(x.pow(4).sum());change=(after-objective)/objective
            worstdecrease=min(worstdecrease,change);assert change>=-1e-10
            normerror=abs(float(x.square().sum())/total-1);maxnorm=max(maxnorm,normerror);assert normerror<1e-10
            history.append(change);objective=after
            if step in [1,5,10,20,40]:checkpoints.append(dict(round=step,objective=objective,curves=score(x,step)))
        trajectories.append(dict(seed=seed,checkpoints=checkpoints,objective_relative_changes=history))
        print(seed,[(s['round'],[round(c['bytes_over_dense'],5) for c in s['curves']]) for s in checkpoints],flush=True)
    baseline={c['tolerance']:c['bytes'] for c in trajectories[0]['checkpoints'][0]['curves']}
    gains={str(tol):max(1-c['bytes']/baseline[tol] for tr in trajectories for step in tr['checkpoints'] for c in step['curves'] if c['tolerance']==tol) for tol in [.02,.05,.1]}
    out=dict(pred_a=True,pred_b=gains['0.1']>=.05,pred_c=gains['0.02']>=.05,planted_error=recovery,max_norm_error=maxnorm,worst_relative_objective_decrease=worstdecrease,trajectories=trajectories,best_relative_byte_savings=gains,seconds=time.perf_counter()-start,scope='Weights-only fourth-power surrogate with three finite pairing schedules; no stationarity/global recovery or behavioral claim. All rotation costs included; no dense residual adapter.')
    (P/'INTERACTION_GIVENS_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    print({k:v for k,v in out.items() if k!='trajectories'})
if __name__=='__main__':main()
