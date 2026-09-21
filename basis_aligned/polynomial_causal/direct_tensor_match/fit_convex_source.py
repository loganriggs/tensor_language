"""Conditional second-read refitting under eight original fidelity constraints."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from convex_quadratic_minimax import solve
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
from pairwise_graph_assessment import Assessment
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
plan=json.loads((P/'CONVEX_SOURCE_PLAN_V1.json').read_text());base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);audit=Assessment(data);parents=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)
ids=data['indices'];z=data['z'][ids];h=data['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();rows=[]
def ratios(result):
 return np.square([result['native_error']/(1.1*base['native_error']),result['covariance_error']/(1.1*base['covariance_error'])]+[v/min(.15,1.1*b) for v,b in zip(result['per_mode_errors'],base['per_mode_errors'])]+[v/(1.1*b) for v,b in zip(result['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])])
def make_graph(parent,banks,allocation,x):
 graph=copy.deepcopy(parent);offset=0
 for j,n in enumerate(allocation):
  V,lam=banks[j];p=graph['pairs'][str(j)];width=len(p['shared_indices'])+len(p['private_indices']);idx=torch.arange(width,width+n)
  if n:
   p['private_reader']=torch.cat([p['private_reader'],V[:,:n]],1);p['private_indices']=torch.cat([p['private_indices'],idx]);p['product_indices']=torch.cat([p['product_indices'],torch.stack([idx,idx,torch.zeros_like(idx)])],1);weights=lam[:n]*torch.as_tensor(x[offset:offset+n]);p['product_weights']=torch.cat([p['product_weights'],torch.stack([torch.zeros_like(weights),weights],1)])
  offset+=n
 return audit.correct(graph)
for geometry in plan['geometries']:
 parent=parents[geometry+'_inherited'];bundle=expand(parent);A,inv=(audit.S,data['inverse_root']) if geometry=='calibration_shaped' else (audit.I,audit.I);banks=[];pair_data=[]
 for j in range(3):
  H=decode(bundle[str(j)]);true=audit.Q[2*j:2*j+2];e,U=torch.linalg.eigh(A@(true[1]-H[1])@A);order=e.abs().argsort(descending=True)[:14];V=inv@U[:,order];lam=e[order];banks.append((V,lam));p=bundle[str(j)]
  lin=torch.stack([p['a_linear'],p['b_linear']]);bias=torch.stack([p['a_bias'],p['b_bias']]);reads=torch.einsum('ni,oij,nj->no',z,H,z)+z@lin.T+bias
  grad=2*torch.einsum('oij,nj->noi',H,z)+lin[None];pair=data['pairs'][j];aa=(h@pair['a']-.5*reads[:,0])/s-pair['alpha'];bb=reads[:,1]/s-pair['beta'];val=aa*bb
  exact=torch.einsum('ni,oij,nj->no',z,true,z);truegrad=2*torch.einsum('oij,nj->noi',true,z);ta=(h@pair['a']-.5*exact[:,0])/s-pair['alpha'];tb=exact[:,1]/s-pair['beta'];J=-.5*(tb/s)[:,None]*truegrad[:,0]+(ta/s)[:,None]*truegrad[:,1];jac=-.5*(bb/s)[:,None]*grad[:,0]+(aa/s)[:,None]*grad[:,1]
  centered=(z-data['mu'])@V;psi=(centered.square()-torch.einsum('ik,ij,jk->k',V,data['old_covariance'],V))*lam
  value_design=aa[:,None]/s[:,None]*psi
  jac_design=(aa/s)[:,None,None]*2*centered[:,None,:]*V[None]*lam-.5*grad[:,0,:,None]*psi[:,None,:]/s[:,None,None].square()
  truth=pair['truth'][ids];value_den=(truth-truth.mean()).norm()*min(.15,1.1*base['per_mode_errors'][j]);jac_den=J.norm()*(1.1*base['euclidean_jacobian_errors'][j])
  pair_data.append(dict(H=H,true=true,value_residual=(val-truth)/value_den,value_design=value_design/value_den,jac_residual=(jac-J).flatten()/jac_den,jac_design=jac_design.reshape(-1,14)/jac_den))
 for allocation in plan['allocations']:
  total=sum(allocation);G=torch.zeros(8,total,total,dtype=torch.float64);b=torch.zeros(8,total,dtype=torch.float64);c=torch.zeros(8,dtype=torch.float64);offset=0
  for j,n in enumerate(allocation):
   D=pair_data[j];V,lam=banks[j];sl=slice(offset,offset+n)
   for k,(M,name) in enumerate(((audit.I,'native_error'),(audit.S,'covariance_error'))):
    T=M@D['true']@M;E=M@(D['H']-D['true'])@M;den=3*T.square().sum()*(1.1*base[name])**2;B=M@V[:,:n];atoms=torch.einsum('ik,jk,k->ijk',B,B,lam[:n]).reshape(-1,n) if n else torch.empty(M.numel(),0,dtype=M.dtype)
    G[k,sl,sl]=atoms.T@atoms/den;b[k,sl]=atoms.T@E[1].flatten()/den;c[k]+=E.square().sum()/den
   for index,kind in ((2+j,'value'),(5+j,'jac')):
    F=D[kind+'_design'][:,:n];r=D[kind+'_residual'];G[index,sl,sl]=F.T@F;b[index,sl]=F.T@r;c[index]=r.square().sum()
   offset+=n
  Gn,bn,cn=G.numpy(),b.numpy(),c.numpy()
  def predicted(x):return np.einsum('i,kij,j->k',x,Gn,x)+2*bn@x+cn
  # Independent native executor at a nontrivial fixed amplitude vector before optimizing.
  control=np.linspace(-.3,.7,total);control_result=audit.assess(make_graph(parent,banks,allocation,control));replay=float(np.max(np.abs(predicted(control)-ratios(control_result))));assert replay<1e-8
  result=solve(Gn,bn,cn,initial=np.ones(total));graph=make_graph(parent,banks,allocation,result['x']);scores=audit.assess(graph);final_replay=float(np.max(np.abs(predicted(result['x'])-ratios(scores))));assert final_replay<1e-8
  assert scores['source_total_multiplications']==1047648+1155*total<=1064448
  row=dict(geometry=geometry,allocation=allocation,solver=result,control_replay=replay,final_replay=final_replay,fidelity_pass=bool(ratios(scores).max()<=1),**scores);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='solver'}),flush=True)
  (P/'CONVEX_SOURCE_PARTIAL_V1.json').write_text(json.dumps(rows,indent=2)+'\n')
primary=rows[0];out=dict(plan=plan,records=rows,predictions=dict(instrument=True,fidelity=primary['fidelity_pass'],cost=True),seconds=time.monotonic()-start)
(P/'CONVEX_SOURCE_V1.json').write_text(json.dumps(out,indent=2)+'\n')
