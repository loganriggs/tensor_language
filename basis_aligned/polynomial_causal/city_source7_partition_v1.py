"""Freeze native-context MLP7-presence terms and complementary city-removal write."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
 capture=torch.load(P/'CITY_SOURCE7_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 reference=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 p=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
 tables=torch.load(P/'CITY_FULL_PILE_V2_TABLES.pt',weights_only=True);lookup={int(t):i for i,t in enumerate(tables['token_ids'])}
 writes=[];errors=[];eps=torch.finfo(torch.float32).eps
 for f,ref in zip(capture,reference,strict=True):
  current=f['current8'];city=ref['candidate_inputs']['city'];mask=ref['candidate_inputs']['destination'];t=current.shape[1]
  z=f['mixed8_city'].double();sources=f['sources'].double()/(z.square().mean(-1,keepdim=True)+eps).sqrt()
  inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angles=torch.outer(torch.arange(t,dtype=torch.float32),inv)
  co,si=angles.cos().bfloat16().float(),angles.sin().bfloat16().float()
  def rotate(x,c,s):
   a,b=x.chunk(2,-1);return torch.cat([a*c+b*s,-a*s+b*c],-1)
  scores=[]
  for qn,kn in [('q1','k1'),('q2','k2')]:
   q=rotate(F.rms_norm(F.linear(current,p[qn]),(128,)),co,si).double()
   key=F.linear(current,p[kn])[:,city].double();den=(key.square().mean(-1,keepdim=True)+eps).sqrt()
   keys=rotate(F.linear(sources,p[kn].double())/den,co[city].double(),si[city].double())
   scores.append(torch.einsum('btd,sbd->sbt',q,keys)/128)
  values=(1-p['mixture'].double())*F.linear(sources,p['current_value'].double())
  token=int(ref['candidate_inputs']['token_ids'][0,city]);inherited=p['mixture'].double()*tables['first_table'][lookup[token],256:384].double()[None]
  def write(indices):
   route=scores[0][indices].sum(0)*scores[1][indices].sum(0)
   value=inherited+values[indices].sum(0)
   return -F.linear(route[...,None]*value[:,None],p['output'].double())*mask[None,:,None]
  expanded=write([0,1,2,3]);absent=write([0,1,3]);present=expanded-absent
  native=ref['inputs']['delta'].double();errors.append(float((expanded-native).norm()/native.norm()))
  # Preserve exactly the original native total; float-rounding residue belongs to complement.
  writes.append(torch.stack([native,present,native-present]))
 assert max(errors)<=1e-4 and all(bool(torch.isfinite(w).all()) for w in writes)
 torch.save({'writes':writes,'arms':['full','mlp7_present','complement']},P/'CITY_SOURCE7_EDIT_V1_WRITES.pt')
 out={'max_expanded_native_relative_error':max(errors),'sequences':len(writes),'scope':'Frozen source-term split; remainder includes native rounding residue. All denominators and queries fixed to native.'}
 (P/'CITY_SOURCE7_EDIT_V1_CPU_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
