"""Post-result descriptive geometry audit; historical calibration only, CPU."""
import json,time
from pathlib import Path
import torch
from spherical_source_moments import moments

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].double()
 data=torch.load(p/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);z=data['z'].double();h=data['h'].double();scale=w('transformer.h.17.lambdas')[0]
 L,R,D=[w('transformer.h.16.mlp.'+k+'.weight') for k in ['Left','Right','Down']];D=scale*D
 source=((z@L.T)*(z@R.T))@D.T;bias=scale*w('transformer.h.16.mlp.Down_bias');carry=h-source-bias
 mean,cov=moments(L,R,D);sm=source.mean(0);sc=source-sm;cm=carry.mean(0);cc=carry-cm;nativecov=sc.T@sc/len(sc)
 def rms(x):return float(x.square().mean().sqrt())
 def cosine(x,y):return float((x*y).sum()/(x.norm()*y.norm()))
 def shape_distance(a,b):return float((a/a.trace()-b/b.trace()).norm()/(b/b.trace()).norm())
 combined=source+carry
 result=dict(samples=len(z),source_rms=rms(source),carry_rms=rms(carry),h_rms=rms(h),source_bias_rms=rms(bias),carry_source_rms_ratio=rms(carry)/rms(source),source_carry_uncentered_cosine=cosine(source,carry),source_carry_centered_cosine=cosine(sc,cc),combined_energy_over_independent_sum=float(combined.square().sum()/(source.square().sum()+carry.square().sum())),sphere_source_rms=float(((cov.trace()+mean.square().sum())/z.shape[1]).sqrt()),sphere_to_native_source_energy_ratio=float((cov.trace()+mean.square().sum())/source.square().sum(1).mean()),source_covariance_trace_normalized_relative_distance=shape_distance(cov,nativecov),source_mean_sphere_vs_native_relative_to_native_total=float((mean-sm).norm()/source.square().sum(1).mean().sqrt()),input_radius_mean=float(z.square().sum(1).mean()),seconds=time.monotonic()-start,scope='Descriptive historical calibration audit after artificial rank reversal. Independent carry assumption and isotropic sphere source law are tested separately. No new model fitting, no native OOD evaluation or causal attribution.')
 (p/'SOURCE_CARRY_GEOMETRY_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
