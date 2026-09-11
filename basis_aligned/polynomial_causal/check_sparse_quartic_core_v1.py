"""A coefficient/allfactor-gradient/denseprojection<=1e-9 B two nearstart planted recoveries<=1e-6.
Cold starts descriptive; optimizer convergence is not global recovery.
"""
import json
import itertools
from pathlib import Path
import torch
from composed_quartic_contraction_v1 import contract
from sparse_quartic_core_v1 import coefficients,indices,fit_fixed,execute


def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(11501)
    p=Path(__file__).parent;out=p/'SPARSE_QUARTIC_CORE_V1_CONTROL.json';assert not out.exists()
    d=4;w=[torch.randn(*shape,requires_grad=True) for shape in [(5,d),(5,d),(3,5),(4,3),(4,3),(2,4)]]
    bank=torch.randn(d,3,requires_grad=True);terms,m=indices(3);c=coefficients(w,bank,terms,m,.7)
    xx=bank.T[terms.T];ref=contract(xx,*w,.7).T*m;errors=[rel(c,ref)]
    upstream=torch.randn_like(c);g0=torch.autograd.grad((c*upstream).sum(),w+[bank],retain_graph=True)
    g1=torch.autograd.grad((ref*upstream).sum(),w+[bank]);errors.extend(rel(a,b) for a,b in zip(g0,g1))
    basis=torch.linalg.qr(torch.randn(d,d)).Q;terms,m=indices(d);w=[a.detach() for a in w]
    c=coefficients(w,basis,terms,m,.7)
    ordered=torch.tensor(list(itertools.product(range(d),repeat=4)))
    dense=contract(torch.eye(d)[ordered],*w,.7)
    errors.append(abs(float(c.square().sum()/dense.square().sum())-1))
    x=torch.randn(9,d);errors.append(rel(execute(basis,terms,m,c,x),contract(x[:,None].expand(-1,4,-1),*w,.7)))
    selected=c.square().sum(0).topk(7).indices
    directions=torch.eye(d)[ordered]@basis;pred=torch.zeros_like(dense)
    for e in selected.tolist():
        permutations=set(itertools.permutations(terms[:,e].tolist()))
        feature=sum(torch.stack([directions[:,i,t[i]] for i in range(4)],-1).prod(-1) for t in permutations)/m[e]
        pred+=feature[:,None]*c[:,e]
    errors.append(abs(float((dense-pred).square().sum()-(dense.square().sum()-c[:,selected].square().sum()))/float(dense.square().sum()))
    eye=torch.eye(6);planted=[eye[[0,2]],eye[[1,3]],torch.eye(2),torch.eye(2),torch.eye(2),torch.eye(2)]
    terms,m=indices(4);true=eye[:,:4];total=coefficients(planted,true,terms,m).square().sum();runs=[]
    for kind in ['near','cold']:
        for seed in [11502,11503]:
            torch.manual_seed(seed);bank=torch.linalg.qr(true+.03*torch.randn_like(true) if kind=='near' else torch.randn_like(true)).Q
            history=[];stable=0
            for cycle in range(8):
                c=coefficients(planted,bank,terms,m);support=c.square().sum(0).topk(2).indices
                bank,r=fit_fixed(planted,bank,terms[:,support],m[support],total,seconds=4,tolerance=1e-8)
                new=coefficients(planted,bank,terms,m).square().sum(0).topk(2).indices
                same=torch.equal(new.sort().values,support.sort().values);stable=stable+1 if same else 0;history.append(r)
                if stable>=2 and r['converged']:break
            captured=float(coefficients(planted,bank,terms[:,support],m[support]).square().sum()/total)
            runs.append(dict(kind=kind,seed=seed,capture=captured,recovered=abs(1-captured)<=1e-6,converged=stable>=2 and r['converged'],cycles=history))
    result=dict(pred_a=max(errors)<=1e-9,pred_b=all(r['recovered'] for r in runs if r['kind']=='near'),maximum_identity_error=max(errors),runs=runs)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='cycles'} for r in runs],indent=2));assert result['pred_a'] and result['pred_b']


if __name__=='__main__':main()
