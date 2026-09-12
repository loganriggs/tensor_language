"""Frozen four-corner scalar9 audit: complete joint-QK routing versus value.
pred_a native replay and exact factorial identity <=1e-5relative.
pred_b EACHtemplate value-only delta reproduces full scalar delta <=10%relative.
pred_c EACHtemplate mixed routing/value term norm <=10%full scalar delta.
No factor fitting or task split between QK1/QK2; both belong to joint routing.
"""
import json,time,torch
import torch.nn.functional as F
from pathlib import Path
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_PRODUCER_JOINT_QK_VALUE_V1_RESULT.json';assert not out.exists()
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];values=torch.zeros(4,48,22,dtype=torch.float64)
 def parts(current,tokens):
  width=128;n=current.shape[1]
  projected=[F.rms_norm(F.linear(current,p[name][1]),(width,),eps=torch.finfo(torch.float32).eps) for name in ('q1','k1','q2','k2')]
  inv=1/(10000**(torch.arange(0,width,2,dtype=torch.float32)/width));angles=torch.outer(torch.arange(n,dtype=torch.float32),inv);c=angles.cos().bfloat16();s=angles.sin().bfloat16()
  def rot(x):
   a,b=x.chunk(2,-1);return torch.cat([a*c+b*s,-a*s+b*c],-1).double()
  q,k,q2,k2=[rot(x) for x in projected];g=(q@k.transpose(-1,-2)/width)*(q2@k2.transpose(-1,-2)/width);g=g.masked_fill(~torch.ones(n,n,dtype=torch.bool).tril(),0)
  v=current.double()@p['current_value_readers'][1]+p['first_token_values'][tokens,1]
  return g,v
 for i,row in enumerate(rows):
  n=len(row['ids']);tokens=torch.tensor([row['ids']]);x=[F.rms_norm(cache['r9'][a,i,:n],(1152,))[None] for a in range(2)];g0,v0=parts(x[0],tokens);g1,v1=parts(x[1],tokens)
  for arm,(g,v) in enumerate(((g0,v0),(g1,v1),(g0,v1),(g1,v0))):values[arm,i,:n]=(g@v[...,None])[0,:,0]
 cells=[]
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];delta=values[1,ix]-values[0,ix];dv=values[2,ix]-values[0,ix];dq=values[3,ix]-values[0,ix];mix=delta-dv-dq
  # Independent product-difference identity is algebraic; native endpoints independently cached.
  cells.append(dict(family=family,native_replay=rel(values[0,ix],cache['a9'][0,ix]),changed_replay=rel(values[1,ix],cache['a9'][1,ix]),factorial_replay=rel(dv+dq+mix,delta),value_only_error=rel(dv,delta),routing_only_error=rel(dq,delta),mixed_to_full_norm=float(mix.norm()/delta.norm()),value_to_full_norm=float(dv.norm()/delta.norm()),routing_to_full_norm=float(dq.norm()/delta.norm()),routing_value_cosine=float(F.cosine_similarity(dq.flatten(),dv.flatten(),dim=0))))
 A=all(max(c['native_replay'],c['changed_replay'],c['factorial_replay'])<=1e-5 for c in cells)
 result={'pred_a':A,'pred_b':A and all(c['value_only_error']<=.1 for c in cells),'pred_c':A and all(c['mixed_to_full_norm']<=.1 for c in cells),'cells':cells,'arms':['native','joint_changed','value_changed','routing_changed'],'seconds':time.perf_counter()-tic,'scope':'Four-corner conditional scalar9 decomposition after component8 removal; QK1*QK2 kept together. Reused native states, no factor fitting or final-effect adoption.'}
 torch.save(dict(scalars=values),P/'SCALAR_PRODUCER_JOINT_QK_VALUE_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
