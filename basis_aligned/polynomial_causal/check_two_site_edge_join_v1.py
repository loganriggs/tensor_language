"""Full-sequence oracle for existing mixed-edge core plus normalized MLP join."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from attention_mixed_edge_core import compile_core,mixed

torch.manual_seed(2501);torch.set_num_threads(2);b,t,d,h=2,5,8,2;e=d//h;dt=torch.float64;eps=torch.finfo(torch.float32).eps
w={name:torch.randn(d,d,dtype=dt)/d**.5 for name in ['q','k','q2','k2','v','o']};x=torch.randn(b,t,d,dtype=dt);first=torch.randn_like(x);da=torch.randn(b,d,dtype=dt)*.2;db=torch.randn(b,d,dtype=dt)*.2;pa=torch.tensor([4,1]);pb=torch.tensor([1,3]);batch=torch.arange(b);phase=torch.randn(t,e//2,dtype=dt);cos=phase.cos().bfloat16().double();sin=phase.sin().bfloat16().double();mixture=.4

def attention(state):
 z=F.rms_norm(state,(d,),eps=eps)
 def branch(name):
  v=F.rms_norm((z@w[name].T).reshape(b,t,h,e),(e,),eps=eps);a,c=v.chunk(2,-1)
  return torch.cat((a*cos[None,:,None]+c*sin[None,:,None],-a*sin[None,:,None]+c*cos[None,:,None]),-1)
 q,k,q2,k2=[branch(name) for name in ['q','k','q2','k2']];v=(1-mixture)*(z@w['v'].T).reshape(b,t,h,e)+mixture*first.reshape(b,t,h,e)
 pattern=(torch.einsum('bthe,bshe->bhts',q,k)/e*torch.einsum('bthe,bshe->bhts',q2,k2)/e).tril()
 return torch.einsum('bhts,bshe->bthe',pattern,v).reshape(b,t,d)@w['o'].T
states=[];writes=[]
for a,c in [(0,0),(1,0),(0,1),(1,1)]:
 z=x.clone();z[batch,pa]+=a*da;z[batch,pb]+=c*db;states.append(z);writes.append(attention(z))
later=torch.maximum(pa,pb);earlier=torch.minimum(pa,pb);qd=torch.where((pa>pb)[:,None],da,db);kd=torch.where((pa>pb)[:,None],db,da)
core=compile_core(w,x[batch,later],qd,x[batch,earlier],kd,torch.eye(d,dtype=dt)[None].expand(b,-1,-1),first[batch,earlier],mixture,h,cos[later],sin[later],cos[earlier],sin[earlier],eps,eps)
edge=mixed(core,1.,1.);expected=writes[3]-writes[1]-writes[2]+writes[0];compiled=torch.zeros_like(expected);compiled[batch,later]=edge
edge_error=float((compiled-expected).abs().max());off=expected.clone();off[batch,later]=0
left,right,down=[torch.randn(d,d,dtype=dt)/d**.5 for _ in range(3)]
def mlp(z):
 n=F.rms_norm(z,(d,),eps=eps);return z+((n@left.T)*(n@right.T))@down.T
post=[s+a for s,a in zip(states,writes)];add=post[1]+post[2]-post[0];joined=add+compiled
mlp_error=float((mlp(joined)-mlp(post[3])).abs().max());omitted=float((mlp(add)-mlp(post[3])).norm());assert edge_error<1e-12 and mlp_error<1e-11 and omitted>1e-5
result=dict(edge_max_error=edge_error,off_later_query_max=float(off.abs().max()),corrected_mlp_max_error=mlp_error,omitted_edge_mlp_error=omitted,coefficients_per_input=sum(v.numel() for v in core.values())//b,scope='Synthetic full-sequence normalized attention+MLP identity, both causal orders; native selective-source joint installation remains pending')
p=Path(__file__).with_name('TWO_SITE_EDGE_JOIN_CPU_V1.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(result)
