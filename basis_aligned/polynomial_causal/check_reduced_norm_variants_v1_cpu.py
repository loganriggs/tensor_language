"""Opened-state local diagnostics; no inference about downstream causal error."""
from pathlib import Path
import sys,json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);folder=P/'extracted_circuits/typed_face_reduced_residual7_v1';sys.path.insert(0,str(folder));import execute as original
 import typed_face_reduced_norm_variants_v1 as variant
 p={k:torch.load(folder/f,weights_only=True) for k,f in [('local','local_program.pt'),('context','context_program.pt'),('reentry','reentry_program.pt')]};fixtures=torch.load(P/'TYPED_FACE_REDUCED_RESIDUAL7_V1_ARTIFACT.pt',weights_only=True)['fixtures'];sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'));from regional_endpoint_batching_v1 import group_rows
 groups,_=group_rows(json.loads((P/'HEAD2_MLP8_CROSS_FRESH_V1_ROWS.json').read_text())['rows']);errors=[];stats={}
 for row,f in zip(groups,fixtures):
  x=f['inputs'];reference=original.execute(p,**x);exact=variant.execute(p,**x);errors.append(float((exact-reference).norm()/reference.norm()))
  for mode in ['omit_correction','freeze_denominator']:
   delta=variant.execute(p,**x,mode=mode);acc=stats.setdefault(row['variant'],{}).setdefault(mode,[0.,0.,0.,0.]);acc[0]+=float((delta-reference).square().sum());acc[1]+=float(reference.square().sum());acc[2]+=float(delta.square().sum());acc[3]+=float((delta*reference).sum())
 zero={m:float(variant.execute(p,**fixtures[0]['inputs'],strength=0,mode=m).abs().max()) for m in ['exact','omit_correction','freeze_denominator']}
 result={'instrument_pass':max(errors)<=1e-10 and max(zero.values())==0,'exact_max_error':max(errors),'zero_strength':zero,'families':{fam:{mode:{'relative_vector_error':(a[0]/a[1])**.5,'norm_ratio':(a[2]/a[1])**.5,'aligned_fraction':a[3]/a[1]} for mode,a in modes.items()} for fam,modes in stats.items()},'scope':'CPUlocalvector diagnostics on40opened states only. Full native suffix causal screen pending; no source or denominator omission adopted.'}
 assert result['instrument_pass'];(P/'REDUCED_MLP8_NORMALIZATION_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
