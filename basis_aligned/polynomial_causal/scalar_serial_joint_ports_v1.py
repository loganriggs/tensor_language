"""Independent multilinear joint-query/key/value expansion of serial head9 response.
A: native scalar-change replay<=1e-5 and expansion/product identity<=1e-10.
B: source-only (joint keys plus values) change error<=.1 EACH family.
C: query-only change error>=.25 EACH family. No fitting or new behavioral verdict.
"""
import json,time,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);tic=time.perf_counter()
 out=P/'SCALAR_SERIAL_JOINT_PORTS_V1_RESULT.json';assert not out.exists()
 cp=P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt'
 pp=P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt'
 rp=P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json'
 c=torch.load(cp,map_location='cpu',weights_only=True);p=torch.load(pp,map_location='cpu',weights_only=True)
 rows=json.loads(rp.read_text())['rows'];terms=torch.zeros(8,48,22,dtype=torch.float64);endpoints=torch.zeros(2,48,22,dtype=torch.float64)
 product_errors=[];expansion_errors=[]
 for i,row in enumerate(rows):
  n=len(row['ids']);ids=torch.tensor(row['ids']);x=F.rms_norm(c['r9'][:,i,:n],(1152,),eps=torch.finfo(torch.float32).eps)
  raw=[F.rms_norm(F.linear(x,p[k][1]),(128,),eps=torch.finfo(torch.float32).eps) for k in ('q1','k1','q2','k2')]
  inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));ang=torch.outer(torch.arange(n,dtype=torch.float32),inv);co=ang.cos().bfloat16();si=ang.sin().bfloat16()
  def rotate(t):
   a,b=t.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1).double()
  q,k,q2,k2=[rotate(t) for t in raw]
  Q=(q[..., :,None]*q2[...,None,:]).flatten(-2);K=(k[..., :,None]*k2[...,None,:]).flatten(-2)
  V=x.double()@p['current_value_readers'][1]+p['first_token_values'][ids,1]
  mask=torch.ones(n,n,dtype=torch.bool).tril()
  g=(Q@K.transpose(-1,-2)/128**2).masked_fill(~mask,0)
  direct=((q@k.transpose(-1,-2)/128)*(q2@k2.transpose(-1,-2)/128)).masked_fill(~mask,0)
  product_errors.append(float((g-direct).norm()/direct.norm().clamp_min(1e-30)))
  endpoints[:,i,:n]=(g@V[...,None])[...,0]
  for b in range(8):
   qt=Q[1]-Q[0] if b&1 else Q[0];kt=K[1]-K[0] if b&2 else K[0];vt=V[1]-V[0] if b&4 else V[0]
   terms[b,i,:n]=((qt@kt.T/128**2).masked_fill(~mask,0)@vt)
  delta=endpoints[1,i,:n]-endpoints[0,i,:n]
  expansion_errors.append(float((terms[1:,i,:n].sum(0)-delta).norm()/delta.norm().clamp_min(1e-30)))
 def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
 cells=[]
 for fam in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==fam];truth=c['a9'][1,ix]-c['a9'][0,ix];ts=terms[:,ix];delta=endpoints[1,ix]-endpoints[0,ix]
  source=ts[2]+ts[4]+ts[6];query=ts[1];cross=ts[3]+ts[5]+ts[7]
  cells.append(dict(family=fam,native_delta_replay=rel(delta,truth),source_only_error=rel(source,truth),query_only_error=rel(query,truth),query_source_interaction_fraction=float(cross.norm()/delta.norm()),term_to_full_norm=[float(t.norm()/delta.norm()) for t in ts[1:]],query_source_cosine=float(F.cosine_similarity(query.flatten(),source.flatten(),dim=0)),source_to_full_norm=float(source.norm()/delta.norm()),query_to_full_norm=float(query.norm()/delta.norm()),change_norm=float(delta.norm())))
 A=max(product_errors+expansion_errors)<=1e-10 and all(x['native_delta_replay']<=1e-5 for x in cells)
 result=dict(pred_a=A,pred_b=A and all(x['source_only_error']<=.1 for x in cells),pred_c=A and all(x['query_only_error']>=.25 for x in cells),max_joint_product_error=max(product_errors),max_expansion_error=max(expansion_errors),cells=cells,term_masks={'1':'delta joint query','2':'delta joint key','3':'delta query x delta key','4':'delta value','5':'delta query x delta value','6':'delta key x delta value','7':'delta query x delta key x delta value'},seconds=time.perf_counter()-tic,source_shas={str(t):hashlib.sha256(t.read_bytes()).hexdigest() for t in (cp,pp,rp,Path(__file__))},scope='Independent exact joint-query/key/value finite expansion on reused head9 native states after component8 removal. Joint feature ports need not be realizable as raw state edits. No data fitting, behavioral selectivity, or new OOD claim.')
 torch.save(dict(terms=terms,endpoints=endpoints),P/'SCALAR_SERIAL_JOINT_PORTS_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2))
if __name__=='__main__':main()
