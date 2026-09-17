"""Exact first-layer response to donor input sources; CPU opened-state analysis."""
from pathlib import Path
import hashlib,json
import torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 path=P/'MLP7_INPUT_SOURCE_V1_ARTIFACT.pt';s=torch.load(path,weights_only=True,map_location='cpu')['sources'].squeeze(2).double()
 program=torch.load(P/'extracted_circuits/odd_attention8h2_mlp7_donor_v1/program.pt',weights_only=True,map_location='cpu')
 p={k:v.double() for k,v in program['donor'].items()};L,R,D=p['left'],p['right'],p['down'];eps=torch.finfo(torch.float32).eps
 g=s.sum(1);paired=s[torch.arange(len(s))^1];rho0=g.square().mean(-1,keepdim=True)+eps
 lg=g@L.T;rg=g@R.T;m0=((lg*rg)@D.T)/rho0
 result={}
 for mask in range(1,8):
  dg=sum(paired[:,j]-s[:,j] for j in range(3) if mask&(1<<j));g1=g+dg;rho1=g1.square().mean(-1,keepdim=True)+eps
  ld=dg@L.T;rd=dg@R.T
  terms={'normalization':(rho0/rho1-1)*m0,'cross':((ld*rg+lg*rd)@D.T)/rho1,'quadratic':((ld*rd)@D.T)/rho1}
  target=(((g1@L.T)*(g1@R.T))@D.T)/rho1-m0
  error=float((sum(terms.values())-target).norm()/target.norm())
  terms['residual_skip']=dg
  rawtarget=dg+target
  stats={name:{'norm_ratio':float(t.norm()/rawtarget.norm()),'aligned_fraction':float((t*rawtarget).sum()/rawtarget.square().sum())} for name,t in terms.items()}
  result[str(mask)]={'relative_closure':error,'raw_residual_response_terms':stats}
 assert max(x['relative_closure'] for x in result.values())<=1e-12
 out={'pred_a':True,'source_masks':{'1':'residual6','2':'attention7','4':'initial7'},'responses':result,
  'identity':'delta M = (rho0^2/rho1^2-1)*(M0-b) + D[(Ldg)*(Rg)+(Lg)*(Rdg)]/rho1^2 + D[(Ldg)*(Rdg)]/rho1^2',
  'scope':'Exact local MLP7 response algebra on opened source states, with FP32 epsilon in FP64 evaluation. Bias cancels, residual skip retained. No downstream selective edit or OOD claim; terms nominate tests only.',
  'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 (P/'MLP7_EXACT_RESPONSE_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
