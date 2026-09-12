"""Exhaustive fixed-reader support exchange on the saved mixed stationary miss.

A initial replay<=1e-8 and every coefficient step nonincreasing;
B final error<=1e-4; C final gradient<=1e-6. Three support/refit cycles.
"""
from pathlib import Path
from itertools import combinations
import json,time
import torch
from quadratic_product_core_v1 import gram,pairs
from quartic_manifold_cg_v1 import tangent
from quartic_manifold_lbfgs_v1 import fit

P=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
s=torch.load(P/'QUARTIC_MIXED_RECOVERY_V1_WORST.pt',weights_only=True)
b,n=s['b'],s['n'];tb,tn,ta=s['trueb'],s['truen'],s['true_mixing'];count=len(b)
allpairs=pairs(2*count);candidatepairs=pairs(count)
lookup={tuple(pair):i for i,pair in enumerate(allpairs.T.tolist())}
candidateids=torch.tensor([lookup[tuple(pair)] for pair in candidatepairs.T.tolist()])
trueids=torch.tensor([lookup[tuple(v+count for v in pair)] for pair in s['edges'].T.tolist()])
oldids=torch.tensor([int(((candidatepairs==pair[:,None]).all(0)).nonzero()[0]) for pair in s['edges'].T])
def matrices(b,n):
    k=gram(torch.cat([b,tb]),torch.cat([n,tn]))
    return k[candidateids][:,candidateids],k[candidateids][:,trueids]@ta,float((ta*(k[trueids][:,trueids]@ta)).sum())
k,c,energy=matrices(b,n)
def value(k,c,ids):
    sub=k[ids][:,ids];cross=c[ids];a=torch.linalg.solve(sub,cross)
    return (energy-(a*cross).sum())/energy
initial=float(value(k,c,oldids))
receipt=json.loads((P/'QUARTIC_MIXED_RECOVERY_V1_CONTROL.json').read_text())
expected=max(r['final_error'] for r in receipt['rows'])
replay=abs(max(initial,0.)**.5-expected)
supports=[torch.tensor(z) for z in combinations(range(len(candidateids)),len(oldids))]
rows=[];valid=True;start=time.perf_counter();last=initial
for cycle in range(3):
    k,c,_=matrices(b,n)
    scores=[float(value(k,c,ids)) for ids in supports]
    ids=supports[min(range(len(scores)),key=scores.__getitem__)]
    chosen=float(value(k,c,ids));valid&=chosen<=last+1e-10
    def evaluate(b,n,divisor,gradient=False):
        if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
        with torch.set_grad_enabled(gradient):
            k,c,_=matrices(b,n);sub=k[ids][:,ids];cross=c[ids]
            with torch.no_grad():a=torch.linalg.solve(sub,cross)
            loss=(energy+(a*(sub@a)).sum()-2*(a*cross).sum())/divisor
            if gradient:
                gb,gn=tangent(b,n,*torch.autograd.grad(loss,(b,n)))
                return float(loss.detach()),a.detach(),gb.detach(),gn.detach()
            return float(loss),a
    b,n,a,h,reason=fit(b,n,evaluate,energy,max_steps=2000,max_seconds=30)
    last=h[-1]['objective'];valid&=last<=chosen+1e-10
    row=dict(cycle=cycle,edges=candidatepairs[:,ids].tolist(),selected_error=max(chosen,0.)**.5,
             final_error=max(last,0.)**.5,gradient=h[-1]['projected_gradient_norm'],
             iterations=h[-1]['iteration'],termination=reason,seconds=h[-1]['seconds'])
    rows.append(row);print(json.dumps(row),flush=True)
    if row['final_error']<=1e-4:break
result=dict(pred_a=replay<=1e-8 and valid,pred_b=rows[-1]['final_error']<=1e-4,
            pred_c=rows[-1]['gradient']<=1e-6,initial_error=max(initial,0.)**.5,
            initial_replay_error=replay,supports_per_cycle=len(supports),rows=rows,
            wall_seconds=time.perf_counter()-start,scope='Exact fixed-reader support search on one small failed case; joint refits remain nonconvex. No native or global guarantee.')
out=P/'QUARTIC_MIXED_SUPPORT_V1_CONTROL.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
