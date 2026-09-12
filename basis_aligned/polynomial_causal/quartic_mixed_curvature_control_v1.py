"""Finite-difference tangent Hessian on the saved mixed-graph stationary miss.

A symmetry<=1e-5 and step comparison<=1e-4; B negative eigenvalue<-1e-5;
C negative-curvature probe plus refit improves error>=10%. No formal certificate.
"""
from pathlib import Path
import json
import torch
from quartic_selected_edges_v1 import gram,target_cross
from quartic_manifold_cg_v1 import tangent,retract
from quartic_manifold_lbfgs_v1 import fit

P=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
s=torch.load(P/'QUARTIC_MIXED_RECOVERY_V1_WORST.pt',weights_only=True)
b,n,edges=s['b'],s['n'],s['edges'];tb,tn,ta=s['trueb'],s['truen'],s['true_mixing']
def oracle(slots):
    reads=torch.einsum('nsd,kdr->nskr',slots,tb)
    out=torch.zeros(len(slots),2)
    def bil(a,c):return (reads[:,a]*reads[:,c]*tn).sum(-1)
    for (a,c),(e,f) in [((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]:
        first,second=bil(a,c),bil(e,f)
        values=(first[:,edges[0]]*second[:,edges[1]]+first[:,edges[1]]*second[:,edges[0]])/6
        out+=values@ta
    return out
energy=float((ta*(gram(tb,tn,edges)@ta)).sum())
def evaluate(b,n,divisor,gradient=False):
    if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
    with torch.set_grad_enabled(gradient):
        k=gram(b,n,edges);c=target_cross(b,n,edges,oracle)
        with torch.no_grad():mix=torch.linalg.solve(k,c)
        loss=(energy+(mix*(k@mix)).sum()-2*(mix*c).sum())/divisor
        if gradient:
            gb,gn=tangent(b,n,*torch.autograd.grad(loss,(b,n)))
            return float(loss.detach()),mix.detach(),gb.detach(),gn.detach()
        return float(loss),mix
initial,_,gb,gn=evaluate(b,n,energy,True)
def flatten(x,y):return torch.cat([x.flatten(),y.flatten()])
def split(v):return v[:b.numel()].reshape_as(b),v[b.numel():].reshape_as(n)
dimension=b.numel()+n.numel()
projector=torch.stack([flatten(*tangent(b,n,*split(e))) for e in torch.eye(dimension)],1)
values,basis=torch.linalg.eigh(projector);basis=basis[:,values>.5]
matrices=[];symmetry=[]
for eps in (1e-4,5e-5):
    columns=[]
    for vector in basis.T:
        db,dn=split(vector);grads=[]
        for sign in (-1,1):
            bb,nn=retract(b,n,db,dn,sign*eps)
            _,_,g1,g2=evaluate(bb,nn,energy,True)
            grads.append(flatten(*tangent(b,n,g1,g2)))
        columns.append(basis.T@((grads[1]-grads[0])/(2*eps)))
    h=torch.stack(columns,1)
    symmetry.append(float((h-h.T).norm()/h.norm()))
    matrices.append((h+h.T)/2)
agreement=float((matrices[0]-matrices[1]).norm()/matrices[1].norm())
eigen,vectors=torch.linalg.eigh(matrices[1]);negative=float(eigen[0])<-1e-5
final=initial;probe=initial;history=[]
if negative:
    db,dn=split(basis@vectors[:,0]);best=(initial,b,n)
    for step in (-.5,-.1,-.01,.01,.1,.5):
        bb,nn=retract(b,n,db,dn,step);value,_=evaluate(bb,nn,energy)
        if value<best[0]:best=(value,bb,nn)
    probe,bb,nn=best
    bb,nn,_,history,reason=fit(bb,nn,evaluate,energy,max_steps=2000,max_seconds=30)
    final=history[-1]['objective']
result=dict(pred_a=max(symmetry)<=1e-5 and agreement<=1e-4,
            pred_b=negative,pred_c=negative and max(final,0.)**.5<=.9*max(initial,0.)**.5,
            tangent_dimension=basis.shape[1],initial_error=max(initial,0.)**.5,
            gradient_norm=float(flatten(gb,gn).norm()),symmetry_errors=symmetry,
            step_agreement=agreement,minimum_eigenvalue=float(eigen[0]),maximum_eigenvalue=float(eigen[-1]),
            eigenvalues=eigen.tolist(),after_probe_error=max(probe,0.)**.5,final_error=max(final,0.)**.5,
            scope='Numerical local-curvature diagnosis on one planted miss, not a rigorous minimum/global certificate or native result.')
out=P/'QUARTIC_MIXED_CURVATURE_V1_CONTROL.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
