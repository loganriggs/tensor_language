"""Exact normalized-input read span; raw-state norm counterexample and repair."""
from pathlib import Path
import json,time,torch
import torch.nn.functional as F
from compiled_scalar_producers_v1 import head_scalar
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_PRODUCER_INPUT_QUOTIENT_V1_RESULT.json';assert not out.exists();gen=torch.Generator().manual_seed(170609)
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');pd={k:v.double() for k,v in p.items()};eps=torch.finfo(torch.float32).eps;records=[];bases=[]
 for h in range(2):
  stack=torch.cat([pd[k][h] for k in ('q1','k1','q2','k2')]+[pd['current_value_readers'][h][None]],0)
  singular=torch.linalg.svdvals(stack);rank=int((singular>singular[0]*1e-12).sum());assert rank==513
  basis=torch.linalg.qr(stack.T,mode='reduced')[0];bases.append(basis)
  raw=torch.randn(8,7,1152,dtype=torch.float64,generator=gen);tokens=torch.randint(50304,(8,7),generator=gen);rho2=raw.square().mean(-1,keepdim=True)+eps;current=raw/rho2.sqrt()
  projected=current@basis@basis.T
  reference=head_scalar(current,tokens,pd,h);reduced=head_scalar(projected,tokens,pd,h)
  normalized_error=float((reduced-reference).norm()/reference.norm())
  noise=torch.randn(8,1152,dtype=torch.float64,generator=gen);null=noise-noise@basis@basis.T;r0=raw[:,0];orth=r0-r0@basis@basis.T;null-=((null*orth).sum(-1)/orth.square().sum(-1))[:,None]*orth;null/=null.norm(dim=-1,keepdim=True)
  changed=raw.clone();changed[:,0]+=r0.norm(dim=-1,keepdim=True)*null
  coordinate_error=float(((changed-raw)@basis).norm()/(raw@basis).norm());changed_rho2=changed.square().mean(-1,keepdim=True)+eps
  new=head_scalar(changed/changed_rho2.sqrt(),tokens,pd,h)
  output_change=float((new-reference).norm()/reference.norm())
  def from_observables(z,rho_squared):
   width=128;length=z.shape[1];projections=[]
   inv=1/(10000**(torch.arange(0,width,2,dtype=torch.float32)/width));angle=torch.outer(torch.arange(length,dtype=torch.float32),inv);cos=angle.cos().bfloat16().double();sin=angle.sin().bfloat16().double()
   for key in ('q1','k1','q2','k2'):
    raw_projection=z@(pd[key][h]@basis).T
    normalized=raw_projection/(raw_projection.square().mean(-1,keepdim=True)+eps*rho_squared).sqrt()
    a,b=normalized.chunk(2,-1);projections.append(torch.cat([a*cos+b*sin,-a*sin+b*cos],-1))
   q,k,q2,k2=projections;gamma=(q@k.transpose(-1,-2)/width)*(q2@k2.transpose(-1,-2)/width);gamma*=torch.ones(length,length,dtype=torch.bool).tril()
   v=(z@(pd['current_value_readers'][h]@basis))/rho_squared[...,0].sqrt()+pd['first_token_values'][tokens,h]
   return (gamma@v[...,None])[...,0]
  augmented=from_observables(changed@basis,changed_rho2)
  augmented_error=float((augmented-new).norm()/new.norm())
  omitted_norm=from_observables(changed@basis,rho2)
  omitted_error=float((omitted_norm-new).norm()/new.norm())
  nativefloat=head_scalar(F.rms_norm(changed.float(),(1152,),eps=eps),tokens,p,h)
  float_error=float((augmented-nativefloat).norm()/nativefloat.norm())
  records.append(dict(head=['8.2','9.8'][h],numerical_read_rank=rank,stack_singular_condition=float(singular[0]/singular[-1]),normalized_input_projection_error=normalized_error,raw_coordinates_change=coordinate_error,raw_nullspace_output_change=output_change,augmented_norm_coordinate_error=augmented_error,wrong_norm_output_error=omitted_error,augmented_vs_native_fp32_error=float_error,original_linear_map_scalars=stack.numel(),reduced_maps_scalars=513*rank,encoder_scalars=1152*rank))
 result={'pred_a':all(r['normalized_input_projection_error']<=1e-10 and r['raw_coordinates_change']<=1e-10 for r in records),'pred_b':all(r['raw_nullspace_output_change']>=1e-3 for r in records),'pred_c':all(r['augmented_norm_coordinate_error']<=1e-10 and r['augmented_vs_native_fp32_error']<=1e-5 for r in records),'records':records,'seconds':time.perf_counter()-tic,'scope':'Random mathematical interface controls, not language validation. Linear read span is sufficient after native input RMS; raw read coordinates alone omit a necessary norm observable. Augmenting with actual rho² replays. Encoder/native prefix costs remain; no global minimum or closed producer-input generator claim.'}
 torch.save(dict(bases=bases),P/'SCALAR_PRODUCER_INPUT_QUOTIENT_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
