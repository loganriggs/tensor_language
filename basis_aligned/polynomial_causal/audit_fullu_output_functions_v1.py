"""Output-function relaxation of the full-U coefficient tensor, CPU only.

A: implicit/dense toy and native trace/eigen replay <=1e-10.
B:32 arbitrary quadratic output functions capture>=.50 coefficient energy.
C: each leading4 function admits a single-real-product approximation capturing>=.90.
This is not the old activation-PCA four-output/rank2 surrogate, nor circuit evidence.
"""
import json,time
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def spectrum(u,l,r,d):
    root=torch.linalg.cholesky(u.T@u)
    writer=root.T@d
    gram=product_cross(l,r,l,r)
    covariance=writer@gram@writer.T
    covariance=(covariance+covariance.T)/2
    values,vectors=torch.linalg.eigh(covariance)
    return values.flip(0),vectors.flip(1),writer,covariance

def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(9116601);dt=torch.float64
    u,l,r,d=torch.randn(9,4,dtype=dt),torch.randn(6,5,dtype=dt),torch.randn(6,5,dtype=dt),torch.randn(4,6,dtype=dt)
    values,_,_,_=spectrum(u,l,r,d)
    h=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    dense=torch.einsum('vk,kab->vab',u@d,h).flatten(1)
    direct=torch.linalg.svdvals(dense).square()[:4]
    toy_error=float((values-direct).norm()/direct.norm());assert toy_error<=1e-10
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double()
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']]
    print('Computing full-U output-function spectrum',flush=True)
    eigen,vectors,writer,covariance=spectrum(u,l,r,d);del u
    total=float(covariance.trace())
    reference=json.loads((P/'STRUCTURED_FIT_STALL_V1_AUDIT.json').read_text())['independent_native_total']
    trace_error=abs(total/reference-1)
    eigen_error=float((covariance-(vectors*eigen)@vectors.T).norm()/covariance.norm())
    captured={str(k):float(eigen[:k].sum()/total) for k in [1,2,4,8,16,32,64,128,256,512,1024,1152]}
    functions=[]
    for i in range(4):
        coefficients=vectors[:,i]@writer
        form=(l.T*coefficients)@r;form=(form+form.T)/2
        ev=torch.linalg.eigvalsh(form);norm=float(form.square().sum())
        positive=max(float(ev[-1]),0.);negative=min(float(ev[0]),0.)
        functions.append(dict(mode=i,coefficient_energy_fraction=float(eigen[i]/total),
            form_energy_replay_error=abs(norm/float(eigen[i])-1),
            best_single_real_product_capture=(positive**2+negative**2)/norm,
            best_rank8_symmetric_capture=float(ev.square().topk(8).values.sum()/norm),
            positive_inertia=int((ev>1e-9*ev.abs().max()).sum()),negative_inertia=int((ev<-1e-9*ev.abs().max()).sum())))
    valid=max(toy_error,trace_error,eigen_error,max(f['form_energy_replay_error'] for f in functions))<=1e-10
    result=dict(schema='fullu.output.functions.v1',captured_energy=captured,leading_functions=functions,
        predictions=dict(pred_a_instrument=valid,pred_b_output_sharing=captured['32']>=.50,
            pred_c_single_product_functions=all(f['best_single_real_product_capture']>=.90 for f in functions)),
        toy_relative_error=toy_error,native_trace_relative_error=trace_error,eigen_replay_relative_error=eigen_error,
        minimum_output_eigenvalue=float(eigen[-1]),native_total=total,
        wall_seconds=time.perf_counter()-start,body_forwards=0,
        scope='Optimal output-mode relaxation in all-token coefficient Frobenius metric. Arbitrary quadratic functions are charged as dense unless further factored. Not an activation-PCA rerun, fresh/OOD prediction, or identified circuit.')
    with (P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
