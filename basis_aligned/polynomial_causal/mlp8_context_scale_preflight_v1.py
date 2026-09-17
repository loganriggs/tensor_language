"""Can the context be replaced by an explicit raw scale plus native attention?"""
from pathlib import Path
import torch,json
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);sources=torch.load(P/'MLP8_CONTEXT_SOURCES_V1_ARTIFACT.pt',weights_only=True)['sources'];fixtures=torch.load(P/'TYPED_FACE_MLP8_COUPLED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];errors=[];rho_values=[]
 for s,f in zip(sources,fixtures):
  raw=(s[0]+s[1]).double();rho=(raw.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();g=f['inputs']['current8'].double()*rho+s[2].double();target=f['inputs']['post_attention8'].double();errors.append(float((g-target).norm()/target.norm()));rho_values.append(rho)
 result={'pred_a':len(errors)==40 and max(errors)<=1e-6,'max_context_reconstruction_error':max(errors),'state_scalars_T32_current_boundary':74880,'state_scalars_T32_if_attention8_generated':38048,'attention8_dense_maps_scalars':6*1152*1152,'scope':'Opened CPU feasibility check with supplied attention8. No actual context port closure until all attention8 outputs are generated from current8 and exact token-only inherited values. Raw RMS scale remains explicit. Added full-attention weights/table must be charged; no compression claim.'}
 assert result['pred_a'];(P/'MLP8_CONTEXT_SCALE_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');torch.save({'rho8':rho_values},P/'MLP8_CONTEXT_SCALE_V1_ARTIFACT.pt');print(result)
if __name__=='__main__':main()
