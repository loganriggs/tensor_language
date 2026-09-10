#!/usr/bin/env python3
# BQGATE: 0forwards0seq; exact weight-only U→MLP17→attention17 output pullback.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from attention_output_quadratic_pullback_v1 import head_energies,output_spectrum,cross_port_energy
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    binding=json.loads((P/'ATTENTION_OUTPUT_PULLBACK_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'ATTENTION_OUTPUT_PULLBACK_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,maps=['O','P O','O P'],heads=9,head_width=128,alarm_seconds=900)));return
    out=P/'ATTENTION_OUTPUT_PULLBACK_V1_RESULT.json';assert not out.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();u-=u.mean(0);root=torch.linalg.cholesky(u.T@u).T
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    writer=root@d;output_gram=writer.T@writer;o=sd['transformer.h.17.attn.c_proj.weight'].double().cuda()
    torch.manual_seed(77119);random=torch.randn(1152,1152,dtype=o.dtype,device='cpu');p,_=torch.linalg.qr(random);p=p.cuda()
    original,original_total=output_spectrum(l,r,writer)
    baseline={str(k):float(original[:k].sum()/original_total) for k in [1,8,32,64,128,256,512]}
    original_bridge=abs(float(original_total)/92104252412.19983-1)
    rows={};spectra={};valid=original_bridge<=1e-10
    for name,mapping in [('native_O',o),('left_scramble',p@o),('right_scramble',o@p)]:
        start=time.perf_counter();lp=l@mapping;rp=r@mapping
        values,total=output_spectrum(lp,rp,writer);table=head_energies(lp,rp,output_gram,9)
        mixed=2*cross_port_energy(l,r,lp,rp,output_gram)
        trace_error=abs(float(table.sum()/total)-1);negative=max(0.,-float(values.min()/total),-float(table.min()/total))
        valid=valid and max(trace_error,negative)<=1e-10
        spectra[name]=values
        row=dict(total_coefficient_energy=float(total),relative_to_residual_port=float(total/original_total),
                 output_capture={str(k):float(values[:k].sum()/total) for k in [1,8,32,64,128,256,512]},
                 head_pair_energy_fractions=(table/total).cpu().tolist(),within_head_fraction=float(table.diag().sum()/total),
                 mixed_symmetric_port_energy=float(mixed),mixed_relative_to_residual_port=float(mixed/original_total),
                 augmented_port_fractions=dict(residual=float(original_total/(original_total+total+mixed)),mixed=float(mixed/(original_total+total+mixed)),attention=float(total/(original_total+total+mixed))),
                 trace_bridge=trace_error,negative_fraction=negative,seconds=time.perf_counter()-start)
        rows[name]=row;print(json.dumps(dict(map=name,result=row)),flush=True)
    spectrum_bridge=float((spectra['native_O']-spectra['right_scramble']).norm()/spectra['native_O'].norm())
    # Native quadratic numerator with its actual input-RMS scalar on arbitrary
    # residual/head-output coordinates. This is algebra, not natural text.
    torch.manual_seed(127);b=torch.randn(16,1152,dtype=o.dtype,device='cuda');z=torch.randn_like(b);x=b+z@o.T
    denominator=x.square().mean(1)+torch.finfo(torch.float32).eps
    direct=(((x@l.T)*(x@r.T))@d.T)@u.T/denominator[:,None]
    bl=b@l.T;br=b@r.T;zl=z@(l@o).T;zr=z@(r@o).T
    folded=((bl*br+bl*zr+br*zl+zl*zr)@d.T)@u.T/denominator[:,None]
    replay=float((direct-folded).norm()/direct.norm());valid=valid and max(spectrum_bridge,replay)<=1e-10
    native=rows['native_O'];left=rows['left_scramble'];right=rows['right_scramble']
    spectrum_gain=native['output_capture']['128']-baseline['128'];alignment_gain=native['output_capture']['128']-left['output_capture']['128'];head_gain=native['within_head_fraction']-right['within_head_fraction']
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_simpler_aligned_output':valid and spectrum_gain>=.05 and alignment_gain>=.02,'pred_c_head_concentration':valid and native['within_head_fraction']>=.5 and head_gain>=.10},original_centered_total=float(original_total),original_output_capture=baseline,maps=rows,top128_gain_vs_residual=spectrum_gain,top128_gain_vs_left_scramble=alignment_gain,within_head_gain_vs_right_scramble=head_gain,bridges=dict(original_total=original_bridge,right_rotation_spectrum=spectrum_bridge,native_normalized_polynomial=replay),price=dict(body_forwards=0,native_O_numbers=o.numel(),native_LR_numbers=l.numel()+r.numel(),native_D_numbers=d.numel(),native_U_numbers=u.numel(),persistent_large_arrays=0),wall_seconds=time.perf_counter()-tic,scope='ExactcenteredcoefficientpullbackthroughO17only. QK/value/history,residualandnormalizationremainexplicitunexplainedinterfaces. Portenergyusesformalcoordinates,notnaturalactivationorcausalimportance. No circuitidentification.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
