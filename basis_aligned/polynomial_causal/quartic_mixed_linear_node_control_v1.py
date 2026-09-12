"""A quadratic node with no self edge is conditionally linear in its matrix.

A original replay<=1e-10 and full-matrix least squares descends;
B rank2 projection plus output refit reduces error>=10%; C final error<=1e-4.
"""
from pathlib import Path
import json,math
import torch
from mixed_quartic_objective_v1 import make_objective
from quartic_manifold_lbfgs_v1 import fit

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
s=torch.load(P/'QUARTIC_MIXED_RECOVERY_V1_WORST.pt',weights_only=True)
b,n,edges=s['b'],s['n'],s['edges'];node=1
assert not any(i==node and j==node for i,j in edges.T.tolist())
def quadratics(b,n):return torch.einsum('kdi,ki,kei->kde',b,n,b)
def product(q,g):
    a=torch.einsum('ab,cd->abcd',q,g)
    return (a+a.permute(2,3,0,1)+a.permute(0,2,1,3)+a.permute(2,0,3,1)+a.permute(0,2,3,1)+a.permute(2,0,1,3))/6
def atoms(q):return torch.stack([product(q[i],q[j]) for i,j in edges.T])
target=torch.einsum('em,eabcd->mabcd',s['true_mixing'],atoms(quadratics(s['trueb'],s['truen'])))
energy=float(target.square().sum())
def oracle(slots):return torch.einsum('na,nb,nc,nd,mabcd->nm',*slots.unbind(1),target)
evaluate=make_objective(edges,oracle,energy)
initial,mix=evaluate(b,n,energy);q=quadratics(b,n)
expected=max(r['final_error'] for r in json.loads((P/'QUARTIC_MIXED_RECOVERY_V1_CONTROL.json').read_text())['rows'])
replay=abs(max(initial,0.)**.5-expected)
remaining=target.clone();g=torch.zeros(2,7,7)
for e,(i,j) in enumerate(edges.T.tolist()):
    if i==node:g+=mix[e,:,None,None]*q[j]
    elif j==node:g+=mix[e,:,None,None]*q[i]
    else:remaining-=mix[e,:,None,None,None,None]*product(q[i],q[j])
basis=[]
for i in range(7):
    for j in range(i,7):
        z=torch.zeros(7,7)
        if i==j:z[i,j]=1
        else:z[i,j]=z[j,i]=1/math.sqrt(2)
        basis.append(z)
basis=torch.stack(basis)
design=torch.stack([torch.stack([product(z,gm) for gm in g]).flatten() for z in basis],1)
solution=torch.linalg.lstsq(design,remaining.flatten(),driver='gelsd')
fullq=torch.einsum('k,kij->ij',solution.solution,basis)
fullerror=float((design@solution.solution-remaining.flatten()).norm()/math.sqrt(energy))
values,vectors=torch.linalg.eigh(fullq);ids=values.abs().argsort()[-2:]
b=b.clone();n=n.clone();b[node]=vectors[:,ids];n[node]=values[ids]/values[ids].norm()
projected,_=evaluate(b,n,energy)
b,n,mix,h,reason=fit(b,n,evaluate,energy,max_steps=2000,max_seconds=30)
final=max(0.,h[-1]['objective'])**.5
result=dict(pred_a=replay<=1e-10 and fullerror<=expected+1e-10,
            pred_b=max(projected,0.)**.5<=.9*expected,pred_c=final<=1e-4,
            original_replay_error=replay,initial_error=expected,full_matrix_error=fullerror,
            design_rank=int(solution.rank),rank2_refitted_error=max(projected,0.)**.5,
            final_error=final,gradient=h[-1]['projected_gradient_norm'],iterations=h[-1]['iteration'],termination=reason,
            scope='One stationary mixed-graph toy miss; exact conditional linear solve before heuristic rank projection and joint nonconvex refit. No native recovery theorem.')
out=P/'QUARTIC_MIXED_LINEAR_NODE_V1_CONTROL.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
