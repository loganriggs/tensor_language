"""Native six-path cue and routed-donor allocation; no fitted selection."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 a=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);v=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True)
 old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh]
 gain=float(torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True)['lambdas'][0]);records=[];checks=[]
 for group in range(4):
  citypaths=[];routedcity=[];routedpost=[];totalcity=[];totalpost=[]
  for i in range(group*24,(group+1)*24):
   n=len(rows[i]['ids']);c=int(v['cue_positions'][i]);j=rows[i]['donor_id'];sgn=-1 if i%2==0 else 1
   difference=a['phi_paths'][j,:n]-a['phi_paths'][i,:n];dv=difference*gain/v['rho9'][i,:n,None];field=v['gamma'][i,:n,:n]@dv
   checks.append(float((field.sum(-1)-v['donor_scalar'][i,:n]).norm()/v['donor_scalar'][i,:n].norm().clamp_min(1e-30)))
   source=v['gamma'][i,n-1,:n,None]*dv*sgn
   citypaths.append(difference[c]*sgn);routedcity.append(source[c]);routedpost.append(source[c+1:].sum(0));totalcity.append(v['final_source_mode_contributions'][i,c].sum()*sgn);totalpost.append(v['final_source_mode_contributions'][i,c+1:n].sum()*sgn)
  citypaths=torch.stack(citypaths);rc=torch.stack(routedcity);rp=torch.stack(routedpost);tc=torch.stack(totalcity);tp=torch.stack(totalpost);cv=citypaths.sum(-1)
  records.append(dict(group=group,city_phi_aligned=(citypaths.T@cv/cv.square().sum()).tolist(),routed_city_aligned=(rc.T@tc/tc.square().sum()).tolist(),routed_post_aligned=(rp.T@tp/tp.square().sum()).tolist()))
 result=dict(path_names=['B_squared','Q7_squared','H8_squared','2_B_Q7','2_B_H8','2_Q7_H8'],records=records,max_donor_field_reconstruction_error=max(checks),scope='Native sixpath computational allocation, retains alloriginalnormalizers andcontext; signedaligned fractions not independentcausal shares.')
 assert max(checks)<=1e-4
 (P/'MLP7_PHI_PARENT_PATHS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
