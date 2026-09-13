from pathlib import Path
import json,torch
from joint_attention_mixed_ports_v1 import decompose
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);a=torch.load(P/'CROSSFIRST_ATTENTION17_PORTS_V1_ARTIFACT.pt',weights_only=True);bindings=json.loads((P/'CROSSFIRST_ATTENTION17_PORTS_V1_BINDING.json').read_text())['files'];sd=torch.load(next(x for x in bindings if x.endswith('pytorch_model.bin')),weights_only=True,mmap=True);W=sd['transformer.h.17.attn.c_proj.weight'][:,256:384].double();vectors=[]
 for ports in a['ports']:
  d=decompose(ports[0],ports[1],ports[3],ports[4]);terms={'score_score':sum(v for k,v in d['cross_terms'].items() if k[-1]=='0'),'score_value':sum(v for k,v in d['cross_terms'].items() if k[-1]!='0'),**{'defect_'+k:v for k,v in d['defect_terms'].items()}}
  vectors.append({k:(v@W.T).flatten() for k,v in terms.items()})
 groups=[]
 for lo,hi,label in [(0,96,'regional'),(96,160,'FineWeb')]:
  terms={k:torch.stack([v[k] for v in vectors[lo:hi]]) for k in vectors[0]};total=sum(terms.values());energy=total.square().sum();records={k:dict(aligned_fraction=float((v*total).sum()/energy),norm_over_total=float(v.norm()/total.norm()),omission_error=float(v.norm()/total.norm())) for k,v in terms.items()};groups.append(dict(panel=label,terms=records,aligned_sum=sum(x['aligned_fraction'] for x in records.values())))
 outcomes=[]
 for k in range(4):
  z=a['readouts'][24*k:24*(k+1),:,0];e=z-z[:,0:1];ref=e[:,1];c=e[:,3];d=e[:,4]
  outcomes.append(dict(group=k,cross_defect_cosine=float((c*d).sum()/(c.norm()*d.norm())),cross_signs_matching_reference=int(((c*ref)>0).sum()),defect_signs_matching_reference=int(((d*ref)>0).sum()),full_signs_matching_reference=int(((e[:,2]*ref)>0).sum())))
 r=dict(write_geometry=groups,outcomes=outcomes,scope='Cached nativeport algebra and measured outcome audit. Geometry uses residualwrite norm, not finaleffect attribution. Maskbits1=score1,2=score2,4=value. score_score holdsnativevalue; score_value contains a changed value and opposite-edit score change. All terms retain bothscores. No fittedstructure or newheldout evidence.')
 (P/'CROSSFIRST_ATTENTION17_PORTS_V1_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
