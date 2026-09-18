"""Ordered city-side K1*K2*V fold; native queries and denominators stay external."""
import itertools,json
from pathlib import Path
import torch
import torch.nn.functional as F

P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def analyze():
 capture=torch.load(P/'CITY_SOURCE7_V1_ARTIFACT.pt',weights_only=True)
 reference=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 p=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
 tables=torch.load(P/'CITY_FULL_PILE_V2_TABLES.pt',weights_only=True)
 lookup={int(t):i for i,t in enumerate(tables['token_ids'])}
 labels=capture['source_labels'];eps=torch.finfo(torch.float32).eps
 stats={};groupstats={};den=0.;native_errors=[];closures=[]
 for sequence,(f,ref) in enumerate(zip(capture['fixtures'],reference,strict=True)):
  current=f['current8'];city=ref['candidate_inputs']['city'];mask=ref['candidate_inputs']['destination'];t=current.shape[1]
  z=f['mixed8_city'].double();sources=f['sources'].double()/(z.square().mean(-1,keepdim=True)+eps).sqrt()
  inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angles=torch.outer(torch.arange(t,dtype=torch.float32),inv)
  co,si=angles.cos().bfloat16().float(),angles.sin().bfloat16().float()
  def rotate(x,c,s):
   a,b=x.chunk(2,-1);return torch.cat([a*c+b*s,-a*s+b*c],-1)
  routes=[]
  for qn,kn in [('q1','k1'),('q2','k2')]:
   q=rotate(F.rms_norm(F.linear(current,p[qn]),(128,)),co,si).double()
   key=F.linear(current,p[kn])[:,city].double()
   norm=(key.square().mean(-1,keepdim=True)+eps).sqrt()
   parts=F.linear(sources,p[kn].double())/norm
   parts=rotate(parts,co[city].double(),si[city].double())
   routes.append(torch.einsum('btd,sbd->sbt',q,parts)/128)
  values=(1-p['mixture'].double())*F.linear(sources,p['current_value'].double())
  token=int(ref['candidate_inputs']['token_ids'][0,city])
  inherited=p['mixture'].double()*tables['first_table'][lookup[token],256:384].double()[None]
  support=mask & (torch.arange(t)>=city)
  def write(a,b,value):
   channels=a[...,None]*b[...,None]*value[:,None]
   return -F.linear(channels,p['output'].double())*support[None,:,None]
  terms={};groups={name:torch.zeros_like(ref['inputs']['delta'].double()) for name in ['mlp7_present','mlp7_absent','attention7_present','attention7_absent']}
  for i,j in itertools.product(range(4),repeat=2):
   for k in [-1,0,1,2,3]:
    v=inherited if k==-1 else values[k]
    term=write(routes[0][i],routes[1][j],v)
    name=f'{labels[i]}*{labels[j]}*'+('inherited' if k==-1 else labels[k])
    terms[name]=term
    for source,label in [(2,'mlp7'),(1,'attention7')]:
     groups[label+('_present' if source in (i,j,k) else '_absent')]+=term
  total=sum(terms.values());direct=write(routes[0].sum(0),routes[1].sum(0),inherited+values.sum(0))
  native=ref['inputs']['delta'].double()
  closures.append(float((total-direct).norm()/direct.norm()))
  native_errors.append(float((total-native).norm()/native.norm()))
  if sequence%2==0:
   previous=(terms,groups,native);continue
  terms={name:previous[0][name]-value for name,value in terms.items()}
  groups={name:previous[1][name]-value for name,value in groups.items()}
  native=previous[2]-native
  den+=float(native.square().sum())
  for collection,accum in [(terms,stats),(groups,groupstats)]:
   for name,term in collection.items():
    item=accum.setdefault(name,{'squared_norm':0.,'dot':0.})
    item['squared_norm']+=float(term.square().sum());item['dot']+=float((term*native).sum())
 for accum in [stats,groupstats]:
  for item in accum.values():
   item['norm_ratio']=(item['squared_norm']/den)**.5;item['aligned_fraction']=item['dot']/den
 out={'pred_d':max(native_errors)<=1e-4,'algebraic_closure_max_relative':max(closures),'native_write_max_relative':max(native_errors),'term_count':len(stats),'groups':groupstats,'terms':stats,'scope':'Opened paired UK-minus-US city-write fold only. Query factors and all RMS denominators native. Presence groups overlap between MLP7 and attention7; no causal or closed-port claim.'}
 (P/'CITY_SOURCE7_CONTRAST_V1_CPU_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='terms'},indent=2))

if __name__=='__main__':analyze()
