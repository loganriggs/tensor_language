"""Donor-free city-source and all-source first-value interaction fields."""
from pathlib import Path
import json,torch,importlib.util
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);a=torch.load(P/'FIRST_TOKEN_PATH_FRESH_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];pr=torch.load(P/'extracted_circuits/first_token_value_path_v1/program.pt',weights_only=True);b=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in b if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);s=importlib.util.spec_from_file_location('tok',P/'extracted_circuits/first_token_value_path_v1/execute.py');exe=importlib.util.module_from_spec(s);s.loader.exec_module(exe);fields=torch.zeros(96,2,a['fields'].shape[-1],dtype=torch.float64);identity=torch.zeros_like(a['fields'][:,1])
 for i,r in enumerate(rows):
  j=r['donor_id'];n=len(r['ids']);ids=torch.tensor(r['ids']);did=torch.tensor(rows[j]['ids']);c=int(torch.where(ids!=did)[0].item());f=exe.token_readings(ids,sd['transformer.wte.weight'],pr);df=exe.token_readings(did,sd['transformer.wte.weight'],pr)[c]-f[c];g8=a['gamma8'][i,:n,:n];q=a['q7'][i,:n];scale=2*pr['head9_gain']/(a['rho8_squared'][i,:n]*a['rho9'][i,:n]);mask=torch.arange(n)>c
  def transport(h,source_mask):return a['gamma9'][i,:n,:n]@(scale*(q*pr['eigenvalues']*h).sum(-1)*source_mask)
  fields[i,0,:n]=transport(g8[:,c,None]*f[c],mask);fields[i,1,:n]=transport(g8@f,torch.ones(n));identity[i,:n]=transport(g8[:,c,None]*(f[c]+df),mask)-fields[i,0,:n]
 rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30));result=dict(donor_identity_error=rel(identity,a['fields'][:,1]),groups=[dict(group=k,city_field_over_donor=float(fields[k*24:(k+1)*24,0].norm()/a['fields'][k*24:(k+1)*24,1].norm()),all_field_over_donor=float(fields[k*24:(k+1)*24,1].norm()/a['fields'][k*24:(k+1)*24,1].norm())) for k in range(4)],scope='Absolute crossfirst term: city-only andallsource definitions. Nativecontext supplied; no physicalremoval yet. Cityremoval retains externalcueposition; allsource doesnot.')
 assert result['donor_identity_error']<1e-5;torch.save(dict(fields=fields),P/'FIRST_TOKEN_ABSOLUTE_REMOVAL_V1_FIELDS.pt');(P/'FIRST_TOKEN_ABSOLUTE_REMOVAL_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
