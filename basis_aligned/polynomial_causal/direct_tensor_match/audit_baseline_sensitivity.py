"""Matched-program conditional derivatives under explicit perturbation metrics."""
from pathlib import Path
import json,torch
from quadratic_pair_blocks import products
P=Path(__file__).resolve().parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
base=torch.load(P/'MULTIMODE_PAIR_BASELINES_V1.pt',weights_only=True)
fit=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text())
p=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[fit['winner']]
z=d['z'][d['indices']];h=d['h'][d['indices']];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
root=torch.linalg.inv(d['inverse_root'])
def decode(pair):
 reader=pair['shared_reader'];i,j,kind=pair['product_indices'].long();u=reader[:,i];v=reader[:,j]
 L=torch.where((kind==1)[None],u+v,u);R=torch.where((kind==1)[None],u-v,v)
 raw=torch.einsum('ir,ro,jr->oij',L,pair['product_weights'],R)
 return (raw+raw.transpose(-1,-2))/2,torch.stack([pair['a_linear'],pair['b_linear']],1),torch.stack([pair['a_bias'],pair['b_bias']])
def calculate(Q,linear,bias,j):
 reads=torch.einsum('ni,oij,nj->no',z,Q,z)+z@linear+bias
 grad=2*torch.einsum('oij,nj->noi',Q,z)+linear.T[None]
 pair=d['pairs'][j];u=(h@pair['a']-.5*reads[:,0])/scale-pair['alpha'];v=reads[:,1]/scale-pair['beta']
 return u*v,-.5*(v/scale)[:,None]*grad[:,0]+(u/scale)[:,None]*grad[:,1]
s=p['shared_mixed'];raw=torch.einsum('ir,ro,jr->oij',s['left_reader'],s['product_weights'],s['right_reader']);Qs=(raw+raw.transpose(-1,-2))/2
records=[];checks=[]
for j in range(3):
 B,lin,bias=decode(base[str(j)])
 # Direct factor-program autograd independently validates decoded Jacobian.
 zz=z[:2].clone().requires_grad_(True);bb=base[str(j)]
 reads=products(zz@bb['shared_reader'],bb['product_indices'])@bb['product_weights']+zz@lin+bias
 val=((h[:2]@bb['h_reader']-.5*reads[:,0])/scale[:2]-bb['alpha'])*(reads[:,1]/scale[:2]-bb['beta'])
 auto=torch.autograd.grad(val.sum(),zz)[0].detach()
 with torch.no_grad():
  v0,J0=calculate(torch.stack(d['pairs'][j]['Qs']),torch.zeros_like(lin),torch.zeros_like(bias),j)
  vb,Jb=calculate(B,lin,bias,j)
  if j<2:vg,Jg=calculate(Qs[2*j:2*j+2],s['source_linear'][:,2*j:2*j+2],s['source_bias'][2*j:2*j+2],j)
  else:vg,Jg=calculate(*decode(p['private_pair']),j)
  checks.append(float((auto-Jb[:2]).norm()/auto.norm()))
  def tangent(J):return J-(J*z).sum(-1,keepdim=True)/z.square().sum(-1,keepdim=True)*z
  transforms={'euclidean':lambda x:x,'covariance':lambda x:x@root,'sphere_tangent':tangent}
  for name,transform in transforms.items():
   ref=transform(J0);b=float((transform(Jb)-ref).norm()/ref.norm());g=float((transform(Jg)-ref).norm()/ref.norm())
   records.append(dict(component=j+1,metric=name,baseline_error=b,graph_error=g,ratio=g/b,relative_pass=g<=1.10*b,absolute_pass=g<=.15))
  print('component',j+1,records[-3:])
out=dict(records=records,autograd_replay_errors=checks,predictions=dict(pred_a_replay=max(checks)<1e-8,pred_b_relative=all(r['relative_pass'] for r in records),pred_c_absolute=all(r['absolute_pass'] for r in records)),scope='448 previously opened states; h fixed. Covariance metric corresponds to perturbations with covariance root@root.T. Sphere projection removes radial perturbations, using Euclidean tangent geometry. Does not include downstream dependence h(z) or actual causal interventions.')
(P/'BASELINE_SENSITIVITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions']);print('replay',checks)
