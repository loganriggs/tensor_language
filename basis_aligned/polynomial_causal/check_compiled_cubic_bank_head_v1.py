"""Independent full-polynomial head, mixed-bank and causal-sum controls."""
from pathlib import Path
import itertools,json,torch
import torch.nn.functional as F
from compiled_cubic_bank_head_v1 import compile_head,execute_pairs,execute_sequence
from folded_normalized_router_v1 import rotary,EPS
from shared_cubic_source_projection_v1 import cross_factors
from cubic_source_mixture_execute_v1 import execute as reference
from cubic_cluster_coordinates_v1 import encode,components
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(9122044);d=2;w=4;o=3
 q1,k1,q2,k2=[torch.randn(w,d) for _ in range(4)];v=torch.randn(w,2*d);output=torch.randn(o,w)
 # Every symmetric cubic monomial in current+first input: exact source span.
 eye=torch.eye(2*d);full=torch.stack([eye[list(z)] for z in itertools.combinations_with_replacement(range(2*d),3)]);p=compile_head(full,torch.eye(len(full)),q1,k1,q2,k2,v,output)
 q=torch.randn(12,d);s=torch.randn(12,2*d);errors={}
 for t,j in [(0,0),(7,2),(31,17)]:
  rt,rs=rotary(t,w),rotary(j,w);scores=[]
  for qm,km in [(q1,k1),(q2,k2)]:
   qa=F.rms_norm(q@qm.T,(w,),eps=EPS)@rt.T;ks=F.rms_norm(s[:,:d]@km.T,(w,),eps=EPS)@rs.T;scores.append((qa*ks).sum(-1)/w)
  expected=(scores[0]*scores[1])[:,None]*(s@v.T@output.T);actual=execute_pairs(q,s,rt.T@rs,p);errors[f'full_basis_{t}_{j}']=float((actual-expected).norm()/expected.norm())
 # An arbitrary mixed three-feature cluster, compared to the prior generic executor.
 a=torch.randn(3,3,2*d);theta,chart=encode(a,[0,1,2]);c,m=components(theta,chart);p=compile_head(c,m,q1,k1,q2,k2,v,output);r=rotary(11,w).T@rotary(4,w)
 ka=torch.cat((r@k1,torch.zeros_like(k1)),1);kb=torch.cat((r@k2,torch.zeros_like(k2)),1);factors=cross_factors(c,q1[None],ka[None],q2[None],kb[None],v[None],output[:,None]);expected=reference(q,s,c,m,factors).sum(2)[:,0]
 norms=torch.ones(len(q))
 for qm,km in [(q1,k1),(q2,k2)]:norms*=((q@qm.T).square().mean(-1)+EPS)*((s[:,:d]@km.T).square().mean(-1)+EPS)
 expected/=w*w*norms.sqrt()[:,None];actual=execute_pairs(q,s,r,p);errors['mixed_reference']=float((actual-expected).norm()/expected.norm())
 # Folding a consumer into the output commutes with this physical source projection.
 C=torch.randn(2,o);folded=compile_head(c,m,q1,k1,q2,k2,v,C@output);read=execute_pairs(q,s,r,folded);errors['consumer_commutation']=float((read-actual@C.T).norm()/read.norm())
 current=torch.randn(2,4,d);first=torch.randn_like(current);seq=execute_sequence(current,first,p);manual=torch.zeros_like(seq)
 for b in range(2):
  for t in range(4):
   for j in range(t+1):manual[b,t]+=execute_pairs(current[b,t:t+1],torch.cat((current[b,j:j+1],first[b,j:j+1]),-1),rotary(t,w).T@rotary(j,w),p)[0]
 errors['causal_sequence']=float((seq-manual).norm()/manual.norm());result=dict(errors=errors,pred_a=max(errors.values())<=1e-10,scope='FP64 independent full source-polynomial basis vs normalized attention; mixed-bank prior executor; consumer folding; arbitrary actual RoPE positions and causal sequence sum. Native FP32/text replay remains separate.')
 out=P/'COMPILED_CUBIC_BANK_HEAD_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result);assert result['pred_a']
if __name__=='__main__':main()
