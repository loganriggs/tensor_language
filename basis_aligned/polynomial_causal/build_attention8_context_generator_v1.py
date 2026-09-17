"""Freeze weight/token generator and check it against existing native captures."""
from pathlib import Path
import torch,torch.nn.functional as F,json,hashlib,sys
import attention8_context_generator_v1 as generator
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);b=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];ckpt=next(x for x in b if x.endswith('pytorch_model.bin'));sd=torch.load(ckpt,weights_only=True,mmap=True,map_location='cpu')
 rows=json.loads((P/'TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json').read_text())['rows'];sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'));from regional_endpoint_batching_v1 import group_rows
 groups,_=group_rows(rows);ids=torch.tensor(sorted({i for r in rows for i in r['ids']}));x=F.rms_norm(sd['transformer.wte.weight'][ids].float(),(1152,));lam=sd['transformer.h.0.lambdas'];x=F.rms_norm(lam[0]*x+lam[1]*x,(1152,));first=F.linear(x,sd['transformer.h.0.attn.c_v.weight'])
 params={k:sd['transformer.h.8.attn.'+v+'.weight'].clone() for k,v in [('q1','c_q'),('k1','c_k'),('q2','c_q2'),('k2','c_k2'),('value','c_v'),('output','c_proj')]};params.update(mixture=sd['transformer.h.8.attn.lamb'].clone(),token_ids=ids,first_table=first)
 sources=torch.load(P/'MLP8_CONTEXT_SOURCES_V1_ARTIFACT.pt',weights_only=True)['sources'];fixtures=torch.load(P/'TYPED_FACE_MLP8_COUPLED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];scales=torch.load(P/'MLP8_CONTEXT_SCALE_V1_ARTIFACT.pt',weights_only=True)['rho8']
 sys.path.insert(0,str(P/'extracted_circuits/typed_face_mlp8_coupled_v1'));import execute as coupled
 cp=torch.load(P/'extracted_circuits/typed_face_mlp8_coupled_v1/program.pt',weights_only=True)
 errors=[];context_errors=[];write_errors=[]
 for row,s,f,rho in zip(groups,sources,fixtures,scales):
  inputs=dict(f['inputs']);a=generator.execute(params,inputs['current8'],torch.tensor([row['ids']]));errors.append(float((a-s[2]).norm()/s[2].norm()))
  g=(rho*inputs['current8'].double()+a.double()).float();context_errors.append(float((g-inputs['post_attention8']).norm()/inputs['post_attention8'].norm()));inputs['post_attention8']=g
  actual=coupled.execute(cp,**inputs);target=f['expected_native_delta'];write_errors.append(float((actual-target).norm()/target.norm()))
 result={'pred_a':len(errors)==40 and max(errors)<=1e-4,'pred_b':max(context_errors)<=1e-4,'pred_c':max(write_errors)<=1e-4,'fixtures':len(errors),'max_attention8_error':max(errors),'max_context_error':max(context_errors),'max_coupled_native_delta_error':max(write_errors),'supported_tokens':len(ids),'generator_float_scalars':sum(v.numel() for v in params.values() if v.is_floating_point()),'state_scalars_T32':38048,'source_checkpoint_sha256':b[ckpt],'scope':'CPU learned-weight replay on40opened native fixtures. Token table compiled algebraically, not fitted. Native suffix readout replay and isolated widened package pending; raw scale remains a native input. No simplicity gain from weight count.'}
 torch.save(params,P/'ATTENTION8_CONTEXT_GENERATOR_V1_PROGRAM.pt');(P/'ATTENTION8_CONTEXT_GENERATOR_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
