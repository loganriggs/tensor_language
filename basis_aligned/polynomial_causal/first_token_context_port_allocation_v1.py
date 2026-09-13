"""Four native-context input allocation for asymmetric paired token-path effects."""
from pathlib import Path
import json,itertools,math,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);a=torch.load(P/'FIRST_TOKEN_PATH_FRESH_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];pr=torch.load(P/'extracted_circuits/first_token_value_path_v1/program.pt',weights_only=True);f=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in f if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True)
 import importlib.util
 s=importlib.util.spec_from_file_location('tok',P/'extracted_circuits/first_token_value_path_v1/execute.py');exe=importlib.util.module_from_spec(s);s.loader.exec_module(exe);records=[];totalerr=0.
 for i in range(0,96,2):
  j=i+1;n=len(rows[i]['ids']);ids=torch.tensor(rows[i]['ids']);jd=torch.tensor(rows[j]['ids']);c=int(torch.where(ids!=jd)[0].item());df=exe.token_readings(ids,sd['transformer.wte.weight'],pr)[c]-exe.token_readings(jd,sd['transformer.wte.weight'],pr)[c];mask=torch.arange(n)>c
  ports=[[a['gamma8'][k,:n,c] for k in (j,i)],[a['gamma9'][k,:n,:n] for k in (j,i)],[a['q7'][k,:n] for k in (j,i)],[(a['rho8_squared'][k,:n]*a['rho9'][k,:n]).reciprocal() for k in (j,i)]];corners={}
  for bits in itertools.product((0,1),repeat=4):
   g8,g9,q,inv=[ports[k][bits[k]] for k in range(4)];corners[bits]=g9@(2*pr['head9_gain']*g8*(q*pr['eigenvalues']*df).sum(-1)*inv*mask)
  alloc=[]
  for port in range(4):
   z=torch.zeros(n,dtype=torch.float64)
   for bits in itertools.product((0,1),repeat=4):
    if bits[port]:continue
    size=sum(bits);weight=math.factorial(size)*math.factorial(3-size)/math.factorial(4);withp=list(bits);withp[port]=1;z+=weight*(corners[tuple(withp)]-corners[bits])
   alloc.append(z)
  difference=corners[(1,1,1,1)]-corners[(0,0,0,0)];summed=torch.stack(alloc).sum(0);err=float((summed-difference).norm()/difference.norm().clamp_min(1e-30));totalerr=max(totalerr,err);den=float(difference.square().sum());aligned=[float((z*difference).sum())/max(den,1e-30) for z in alloc];records.append(dict(pair=i//2,family=rows[i]['family_name'],cities=[rows[i]['city'],rows[j]['city']],endpoint=rows[i]['uk_token'],aligned_allposition=aligned,finalposition_contributions=[float(z[-1]) for z in alloc],final_British=float(corners[(1,1,1,1)][-1]),final_American=float(corners[(0,0,0,0)][-1])))
 result=dict(max_sum_error=totalerr,port_names=['gamma8_city_column','gamma9','Q7','inverseRMS8_RMS9'],records=records,scope='Four-input exact Shapley on computational donor/recipientcontext corners. Tokencontrast fixedBritish-American. Nativehybridinputs neednotbe textreachable. No physical context-port intervention or correction of failed sign criterion.')
 assert totalerr<1e-12;(P/'FIRST_TOKEN_CONTEXT_PORT_ALLOCATION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(max_sum_error=totalerr,distant_sheffield_baltimore=records[-6:]),indent=2))
if __name__=='__main__':main()
