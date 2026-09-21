"""Convex local constraints for changing only component-three second-read weights."""
import json
from pathlib import Path
import torch
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
from conditional_source_constraints import ConditionalSourceConstraints
P=Path(__file__).resolve().parent

def build(extra_direction=None):
 torch.set_num_threads(2)
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);graph=torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True);bundle=expand(graph);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];metric=ConditionalSourceConstraints(d,H,base)
 V=graph['pairs']['2']['private_reader'][:,-32:];V=V/V.norm(dim=0)
 if extra_direction is not None:V=torch.cat((V,extra_direction[:,None]/extra_direction.norm()),1)
 atoms=torch.einsum('ik,jk->kij',V,V);z,h,s=metric.z,metric.h,metric.s;n=V.shape[1]
 centered=(z-d['mu'])@V;psi=centered.square()-torch.einsum('ik,ij,jk->k',V,d['old_covariance'],V);gradpsi=2*centered[:,None,:]*V[None]
 G=torch.zeros(8,n,n,dtype=H.dtype);b=torch.zeros(8,n,dtype=H.dtype);c=metric.ratios_full(H).detach();pair=d['pairs'][2]
 for k,(M,name) in enumerate(((metric.I,'native_error'),(metric.S,'covariance_error'))):
  F=(M@atoms@M).flatten(1).T;res=(M@(H[5]-metric.true[5])@M).flatten();den=3*(M@metric.true[4:6]@M).square().sum()*(1.1*base[name])**2;G[k]=F.T@F/den;b[k]=F.T@res/den
 delta=metric.true-H;linear=2*delta@d['mu'];bias=torch.einsum('ij,oij->o',d['old_covariance'],delta)-torch.einsum('i,oij,j->o',d['mu'],delta,d['mu']);reads=torch.einsum('ni,oij,nj->no',z,H,z)+z@linear.T+bias;qa_grad=2*z@H[4]+linear[4];a=(h@pair['a']-.5*reads[:,4])/s-pair['alpha'];values,jacs=metric.responses(H);truth=pair['truth'][d['indices']]
 designs=[(4,values[2]-truth,(a/s)[:,None]*psi,(truth-truth.mean()).norm()*min(.15,1.1*base['per_mode_errors'][2])),(7,jacs[2]-metric.truth_jac[2],(a/s)[:,None,None]*gradpsi-.5*qa_grad[:,:,None]*psi[:,None,:]/s[:,None,None].square(),metric.truth_jac[2].norm()*1.1*base['euclidean_jacobian_errors'][2])]
 for k,res,F,den in designs:
  F=F.reshape(-1,n)/den;res=res.flatten()/den;G[k]=F.T@F;b[k]=F.T@res
 checks=[]
 for scale in (0.,.001,.01):
  x=torch.linspace(-scale,scale,n,dtype=H.dtype);trial=H.clone();trial[5]+=torch.einsum('k,kij->ij',x,atoms);pred=torch.einsum('i,kij,j->k',x,G,x)+2*b@x+c;actual=metric.ratios_full(trial);checks.append(float((pred-actual).abs().max()));assert checks[-1]<1e-8
 return dict(G=G,b=b,c=c,V=V,atoms=atoms,H=H,data=d,graph=graph,metric=metric,checks=checks)
if __name__=='__main__':
 torch.set_grad_enabled(False);problem=build();(P/'ROBUST_FIXED_READ_CONTROL_V1.json').write_text(json.dumps(dict(quadratic_replay=problem['checks'],variables=32,constraints=8,scope='Exact frozen-direction conditional subproblem, not global graph infeasibility.'),indent=2)+'\n');print(problem['checks'])
