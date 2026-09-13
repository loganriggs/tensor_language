"""Direct L0-penalized reconstruction pair search; includes rotation storage charge.

Registered before execution: three schedules, 10 rounds, 17 angles including zero;
A norm and accepted-step capped-loss controls <=1e-10. B >=5% actual total-byte
improvement at 10% coefficient error in any declared checkpoint1,5,10. No claim
of global/coordinate convergence for finite angle grid or pairing schedule.
"""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);start=time.perf_counter();t,ids=build();bases=[]
    for axis in [0,2]:
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1);bases.append(torch.linalg.eigh(flat@flat.T)[1])
    core=torch.einsum('op,oia,ab->pib',bases[0],t,bases[1]);initial=core.movedim(1,0).reshape(1152,-1).contiguous()
    initial=initial/initial.square().mean().sqrt();total=float(initial.square().sum());n=initial.numel();adapter=sum(b.numel() for b in bases)
    def score(x,rotations):
        values=x.flatten().square().sort(descending=True).values;k=int(torch.searchsorted(values.cumsum(0),.99*total))+1
        price=4*adapter+4*k+(n+7)//8+12*rotations
        return dict(entries=k,rotations=rotations,bytes=price,bytes_over_dense=price/(4*n)),float(values[k-1])
    base,_=score(initial,0);results=[];normerror=0.;worst=0.
    for seed in [61330,61331,61332]:
        gen=torch.Generator().manual_seed(seed);x=initial.clone();rotations=0;history=[]
        for step in range(1,11):
            _,threshold=score(x,rotations)
            perm=torch.randperm(1152,generator=gen);left,right=perm[::2],perm[1::2];a,b=x[left],x[right]
            before=(a.square().clamp_max(threshold)+b.square().clamp_max(threshold)).sum(-1)
            best=before.clone();best_angle=torch.zeros(576,dtype=x.dtype)
            for angle in torch.linspace(-torch.pi/4,torch.pi/4,17,dtype=x.dtype):
                co,si=angle.cos(),angle.sin();aa=co*a+si*b;bb=-si*a+co*b
                val=(aa.square().clamp_max(threshold)+bb.square().clamp_max(threshold)).sum(-1)
                improved=val<best;best[improved]=val[improved];best_angle[improved]=angle
            accepted=(before-best)>3*threshold
            angle=torch.where(accepted,best_angle,0)[:,None];co,si=angle.cos(),angle.sin()
            x[left]=co*a+si*b;x[right]=-si*a+co*b;rotations+=int(accepted.sum())
            after=(x[left].square().clamp_max(threshold)+x[right].square().clamp_max(threshold)).sum(-1)
            decrease=float(((after-before)/before.clamp_min(1e-30)).max());worst=max(worst,decrease);assert decrease<=1e-10
            normerror=max(normerror,abs(float(x.square().sum())/total-1));assert normerror<1e-10
            cell,_=score(x,rotations);history.append(dict(round=step,accepted=int(accepted.sum()),**cell))
        results.append(dict(seed=seed,history=history));print(seed,history[-1],flush=True)
    gain=max(1-c['bytes']/base['bytes'] for r in results for c in r['history'] if c['round'] in [1,5,10])
    out=dict(pred_a=True,pred_b=gain>=.05,baseline=base,results=results,best_relative_bytes_gain=gain,max_norm_error=normerror,max_capped_objective_increase=worst,seconds=time.perf_counter()-start,scope='Finite-grid greedy L0 reconstruction tradeoff, weights only; variable threshold rounds do not share one global capped objective. No stationarity or behavioral claim.')
    (P/'INTERACTION_GIVENS_CAPPED_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='results'})
if __name__=='__main__':main()
