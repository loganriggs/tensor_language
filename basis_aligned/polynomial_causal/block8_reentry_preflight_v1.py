"""Move the declared state boundary to residual7, with exact token initial states."""
from pathlib import Path
import sys,json,torch,torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);sys.path.insert(0,str(P/'extracted_circuits/typed_face_context_shared_v1'));import execute
 folder=P/'extracted_circuits/typed_face_context_shared_v1';program={'local':torch.load(folder/'local_program.pt',weights_only=True),'context':torch.load(folder/'context_program.pt',weights_only=True)}
 b=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(x for x in b if x.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');lam=sd['transformer.h.8.lambdas'].clone();assert float(lam[0])!=0
 ids=program['context']['token_ids'];initial=F.rms_norm(sd['transformer.wte.weight'][ids].float(),(1152,));lookup={int(t):i for i,t in enumerate(ids)}
 fixtures=torch.load(P/'TYPED_FACE_CONTEXT_GENERATED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];sources=torch.load(P/'MLP8_CONTEXT_SOURCES_V1_ARTIFACT.pt',weights_only=True)['sources'];errors=[];writes=[]
 for i,(f,s) in enumerate(zip(fixtures,sources)):
  x=dict(f['inputs']);tokens=x['token_ids'];table_indices=torch.tensor([[lookup[t] for t in row] for row in tokens.tolist()]);h7=s[0]/lam[0];mixed=lam[0]*h7+lam[1]*F.embedding(table_indices,initial);current=F.rms_norm(mixed,(1152,));rho=(mixed.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
  city=x['city'];donor_h7=sources[i^1][0][:,city]/lam[0];donor_x0=initial[lookup[x['donor_token']]][None];donor=F.rms_norm(lam[0]*donor_h7+lam[1]*donor_x0,(1152,))
  errors.append(float((current-x['current8']).norm()/x['current8'].norm()));x.update(current8=current,donor_city8=donor,rho8=rho);actual=execute.execute(program,**x);target=f['expected_native_delta'];writes.append(float((actual-target).norm()/target.norm()))
 result={'pred_a':max(errors)<=1e-6,'pred_b':max(writes)<=1e-4,'max_current8_error':max(errors),'max_native_delta_error':max(writes),'native_vector_state_arrays':2,'native_state_scalars_T32':38016,'added_initial_table_floats':initial.numel()+lam.numel(),'total_program_floats':23975681+initial.numel()+lam.numel(),'scope':'Opened CPU feasibility using h7 reconstructed by dividing a rounded scaled source. Exact native h7 capture/replay pending. Initial states from weights; no fitted proxy or external rho. Two upstream native vector states and later suffix remain.'}
 torch.save({'token_ids':ids,'initial_table':initial,'lambdas8':lam},P/'BLOCK8_REENTRY_V1_PROGRAM.pt');(P/'BLOCK8_REENTRY_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
