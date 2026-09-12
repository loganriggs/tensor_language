"""Known mixed-graph recovery, two near and two independent initializations.

A dense target agreement<=1e-10, constraints<=1e-10 and descent;
B every coefficient error<=1e-4; C every projected gradient<=1e-6.
2000 updates or60CPU seconds per start; misses are retained.
"""
from pathlib import Path
import json,time
import torch
from quartic_selected_edges_v1 import gram,target_cross
from quartic_manifold_cg_v1 import tangent
from quartic_manifold_lbfgs_v1 import fit

P=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(91931)
edges=torch.tensor([[0,0,1,2],[0,1,2,2]])
tb=torch.linalg.qr(torch.randn(3,7,2),mode='reduced')[0]
tn=torch.randn(3,2);tn/=tn.norm(dim=-1,keepdim=True)
ta=torch.randn(4,2)
q=torch.einsum('kdi,ki,kei->kde',tb,tn,tb)
atoms=[]
for i,j in edges.T:
    a=torch.einsum('ab,cd->abcd',q[i],q[j])
    atoms.append((a+a.permute(2,3,0,1)+a.permute(0,2,1,3)+a.permute(2,0,3,1)+a.permute(0,2,3,1)+a.permute(2,0,1,3))/6)
target=torch.einsum('em,eabcd->mabcd',ta,torch.stack(atoms))
energy=float(target.square().sum())
def oracle(slots):return torch.einsum('na,nb,nc,nd,mabcd->nm',*slots.unbind(1),target)
dense_gram=torch.stack(atoms).flatten(1)@torch.stack(atoms).flatten(1).T
identity=float((dense_gram-gram(tb,tn,edges)).norm()/dense_gram.norm())
cross=target_cross(tb,tn,edges,oracle)
identity=max(identity,float((cross-dense_gram@ta).norm()/(dense_gram@ta).norm()))
assert identity<=1e-10
def evaluate(b,n,divisor,gradient=False):
    if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
    with torch.set_grad_enabled(gradient):
        k=gram(b,n,edges);c=target_cross(b,n,edges,oracle)
        with torch.no_grad():mix=torch.linalg.solve(k,c)
        loss=(energy+(mix*(k@mix)).sum()-2*(mix*c).sum())/divisor
        if gradient:
            gb,gn=tangent(b,n,*torch.autograd.grad(loss,(b,n)))
            return float(loss),mix.detach(),gb.detach(),gn.detach()
        return float(loss),mix
rows=[];worst=None;start=time.perf_counter()
for kind,seed in [('near',91932),('near',91933),('independent',91934),('independent',91935)]:
    torch.manual_seed(seed)
    b=tb+.02*torch.randn_like(tb) if kind=='near' else torch.randn_like(tb)
    b=torch.linalg.qr(b,mode='reduced')[0]
    n=tn+.02*torch.randn_like(tn) if kind=='near' else torch.randn_like(tn)
    n/=n.norm(dim=-1,keepdim=True)
    initial,_=evaluate(b,n,energy)
    b,n,mix,history,reason=fit(b,n,evaluate,energy,max_steps=2000,max_seconds=60)
    error=max(0.,history[-1]['objective'])**.5
    constraint=max(float((b.transpose(-1,-2)@b-torch.eye(2)).abs().max()),float((n.norm(dim=-1)-1).abs().max()))
    row=dict(kind=kind,seed=seed,initial_error=max(0.,initial)**.5,final_error=error,
             gradient=history[-1]['projected_gradient_norm'],iterations=history[-1]['iteration'],
             seconds=history[-1]['seconds'],termination=reason,constraint_error=constraint,
             monotone=all(y['objective']<=x['objective']+1e-10 for x,y in zip(history,history[1:])))
    rows.append(row);print(json.dumps(row),flush=True)
    if worst is None or error>worst[0]:worst=(error,dict(b=b,n=n,mixing=mix,trueb=tb,truen=tn,true_mixing=ta,edges=edges))
result=dict(pred_a=identity<=1e-10 and all(r['constraint_error']<=1e-10 and r['monotone'] for r in rows),
            pred_b=all(r['final_error']<=1e-4 for r in rows),pred_c=all(r['gradient']<=1e-6 for r in rows),
            identity_error=identity,rows=rows,wall_seconds=time.perf_counter()-start,
            scope='Small fixed-graph realizable target; near and independent recovery separate from global/native guarantees.')
out=P/'QUARTIC_MIXED_RECOVERY_V1_CONTROL.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
if not result['pred_b']:
    artifact=P/'QUARTIC_MIXED_RECOVERY_V1_WORST.pt';assert not artifact.exists();torch.save(worst[1],artifact)
print(json.dumps(result,indent=2));assert result['pred_a']
