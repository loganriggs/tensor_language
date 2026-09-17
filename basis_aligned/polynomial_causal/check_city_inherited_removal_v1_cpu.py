"""Opened local comparison; not a model-effect or selective-removal result."""
from pathlib import Path
import sys,json,torch,hashlib
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);folder=P/'extracted_circuits/typed_face_single_head_norm_v1';sys.path.insert(0,str(folder));import city_inherited_removal_v1 as removal
 from regional_endpoint_batching_v1 import group_rows
 groups,_=group_rows(json.loads((P/'SINGLE_HEAD_FRESH_V1_ROWS.json').read_text())['rows']);p=torch.load(folder/'program.pt',weights_only=True);fixtures=torch.load(P/'SINGLE_HEAD_FRESH_V1_ARTIFACT.pt',weights_only=True)['fixtures'];stats={};zero=[];outside=[]
 m=p['mlp8'];L=m['left'].double();R=m['right'].double();D=m['down'].double();eps=torch.finfo(torch.float32).eps
 for row,f in zip(groups,fixtures):
  x={k:f['candidate_inputs'][k] for k in ['residual7','token_ids','city','destination']};d,_=removal.source(p,**x);d=d.double();z=f['inputs']['post_attention8'].double();s0=z.square().mean(-1,keepdim=True)+eps;s1=(z+d).square().mean(-1,keepdim=True)+eps
  exact=d+(((z+d)@L.T)*((z+d)@R.T)/s1-(z@L.T)*(z@R.T)/s0)@D.T;got=removal.execute(p,**x)
  acc=stats.setdefault(row['variant'],[0.,0.,0.,0.]);acc[0]+=float((got-exact).square().sum());acc[1]+=float(exact.square().sum());acc[2]+=float(got.square().sum());acc[3]+=float((got*exact).sum())
  zero.append(float(removal.execute(p,**x,strength=0).abs().max()));outside.append(float(got[:,~x['destination']].abs().max()))
 result={'instrument_pass':len(fixtures)==40 and max(zero)==0 and max(outside)==0,'zero_strength_max':max(zero),'outside_max':max(outside),'native_state_arrays':1,'families':{fam:{'relative_vector_error':(v[0]/v[1])**.5,'norm_ratio':(v[2]/v[1])**.5,'aligned_fraction':v[3]/v[1]} for fam,v in stats.items()},'scope':'New donor-free inherited-city half-removal. CPU local-response diagnostics on40opened states only; all downstream prediction/selectivity and composition untested. One native recipient residual7 input; no donor-state substitution for earlier swap counterfactual.','source_sha256':hashlib.sha256((P/'city_inherited_removal_v1.py').read_bytes()).hexdigest()}
 assert result['instrument_pass'];(P/'CITY_INHERITED_REMOVAL_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
