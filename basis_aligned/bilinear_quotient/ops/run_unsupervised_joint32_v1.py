#!/usr/bin/env python3
# BQGATE: full-vocabulary joint32 products,2starts240steps,0nativeforwards.
"""pred_a algebra/finite/condition checks; pred_b each start gains>=.05 native
squared tensor energy versus LS-refitted native32; pred_c >=16 mutual component
matches with signed full-term cosine>=.95. Native remainder stays explicit.
"""
import os,json,sys,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(POLY)]
import torch
import torch.nn.functional as F
from joint_quadratic_optimizer_v1 import fit,solve
from joint_quadratic_fit_v1 import product_cross
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'UNSUPERVISED_JOINT32_V1_RESULT.json';BIND=POLY/'UNSUPERVISED_JOINT32_V1_BINDING.json'
CHECKPOINT=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def serial(x):return x.detach().cpu().tolist()
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
def matrix(c,l,r):
    raw=l.T@(c[:,None]*r);return (raw+raw.T)/2

def usage(coeff,a,b):
    energy=coeff.square();total=energy.sum(0);ordered=energy.sort(dim=0,descending=True).values.cumsum(0)
    effective=total.square()/energy.square().sum(0).clamp_min(1e-30)
    readers=torch.cat([a,b]);cos=readers@readers.T;cos.fill_diagonal_(0);values,indices=cos.abs().max(1)
    return dict(effective_token_count=serial(effective),tokens_for_90pct_squared_loading=serial((ordered<.9*total).sum(0)+1),tokens_for_99pct_squared_loading=serial((ordered<.99*total).sum(0)+1),positive_loading_count=serial((coeff>0).sum(0)),negative_loading_count=serial((coeff<0).sum(0)),top_positive_ids=serial(coeff.topk(12,dim=0).indices.T),top_negative_ids=serial((-coeff).topk(12,dim=0).indices.T),nearest_linear_reader=serial(indices),nearest_linear_reader_abs_cosine=serial(values),scope='Scale-invariant within-factor loading concentration; dense signs and thresholded mass are not exact sparse edges. Linear readers indexed a0..a31,b0..b31.')

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    fixture=json.loads((POLY/'JOINT_QUADRATIC_OPTIMIZER_V1_CONTROL.json').read_text());assert fixture['planted_dag_relative_error']<=.01 and fixture['gradient_absolute_error']<=1e-7
    algebra=json.loads((POLY/'JOINT_QUADRATIC_FIT_V1_ALGEBRA_RESULT.json').read_text());assert max(algebra[k] for k in ['implicit_loss_relative_error','all_output_least_squares_relative_error'])<=1e-10
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,starts=2,steps_per_start=240,products=32,full_vocabulary=50304,fixture=fixture)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True);U=state['lm_head.weight'].cuda();D=state['transformer.h.17.mlp.Down.weight'].double().cuda();L=state['transformer.h.17.mlp.Left.weight'].double().cuda();R=state['transformer.h.17.mlp.Right.weight'].double().cuda()
    assert U.shape==(50304,1152) and D.shape==(1152,4608)
    with torch.no_grad():
        M=torch.zeros(1152,1152,device='cuda',dtype=torch.float64)
        for block in U.split(512):M+=block.double().T@block.double()
        GO=D.T@M@D;GI=product_cross(L,R,L,R);total=(GO*GI).sum();power=GO.diag()*GI.diag();indices=power.topk(32).indices
        a0=F.normalize(L[indices],dim=1);b0=F.normalize(R[indices],dim=1);w,c,g=solve(L,R,D,a0,b0)
        base_loss=float((total+((w.T@M@w)*g).sum()-2*((D.T@M@w)*c).sum())/total)
        baseline=dict(a=a0,b=b0,w=w);del GO,GI
    runs={};programs={'native32_refit':baseline};checks={}
    torch.manual_seed(9114101)
    starts={'native':(a0,b0),'random':(torch.randn(32,1152,device='cuda',dtype=torch.float64),torch.randn(32,1152,device='cuda',dtype=torch.float64))}
    for name,(a,b) in starts.items():
        best,diagnostics=fit(L,R,D,M,a,b,total,steps=240,lr=.03);programs[name]=best;runs[name]=diagnostics
        print(json.dumps(dict(start=name,best_squared_error=diagnostics['best_squared_relative_error'],condition=diagnostics['condition'])),flush=True)
    saved={};profiles={};energy_norms={}
    with torch.no_grad():
        for name,program in programs.items():
            a,b,w=(program[k] for k in ['a','b','w']);cross=product_cross(L,R,a,b);gram=product_cross(a,b,a,b)
            coeff=torch.cat([block.double()@w for block in U.split(512)])
            profiles[name]=usage(coeff[:50257],a,b)
            ids=torch.tensor([0,1,50256],device='cuda');cn=U[ids].double()@D;ca=U[ids].double()@w
            s=torch.stack([matrix(v,L,R) for v in cn]);t=torch.stack([matrix(v,a,b) for v in ca]);dense=(s-t).square().sum()
            target3=(cn@product_cross(L,R,L,R)*cn).sum();approx3=(ca@gram*ca).sum();overlap3=(cn@cross*ca).sum();implicit=target3+approx3-2*overlap3
            checks[name+'_loss']=float((implicit-dense).abs()/dense.clamp_min(1e-30))
            expected=torch.linalg.solve(gram,(cn@cross).T).T;checks[name+'_coefficients']=rel(ca,expected)
            energy_norms[name]=float(((w.T@M@w)*gram).sum());saved[name]=dict(a=a.cpu(),b=b.cpu(),w=w.cpu(),token_coefficients=coeff.cpu())
        aa,bb=programs['native'],programs['random'];cross=product_cross(aa['a'],aa['b'],bb['a'],bb['b']);ga=product_cross(aa['a'],aa['b'],aa['a'],aa['b']);gb=product_cross(bb['a'],bb['b'],bb['a'],bb['b']);ow=aa['w'].T@M@bb['w']
        na=((aa['w'].T@M@aa['w']).diag()*ga.diag()).sqrt();nb=((bb['w'].T@M@bb['w']).diag()*gb.diag()).sqrt();cos=ow*cross/na[:,None].clamp_min(1e-30)/nb[None,:].clamp_min(1e-30)
        va,ja=cos.max(1);_,jb=cos.max(0);ix=torch.arange(32,device='cuda');matches=(jb[ja]==ix)&(va>=.95)&(na>=1e-6*na.max())&(nb[ja]>=1e-6*nb.max())
        overlap=float(torch.trace(torch.linalg.solve(ga,cross)@torch.linalg.solve(gb,cross.T))/32)
        difference=(energy_norms['native']+energy_norms['random']-2*(ow*cross).sum()).clamp_min(0)
        stability=dict(mutual_term_matches=int(matches.sum()),native_partner=serial(ja),native_term_cosine=serial(va),matched=serial(matches),input_function_subspace_overlap=overlap,program_disagreement_relative_to_full_tensor=float((difference/total).sqrt()))
    valid=max(checks.values())<=1e-8 and all(d['condition']<=1e10 and d['steps']==240 for d in runs.values()) and all(torch.isfinite(v).all() for p in saved.values() for v in p.values())
    gain={n:base_loss-d['best_squared_relative_error'] for n,d in runs.items()}
    ap=POLY/'UNSUPERVISED_JOINT32_V1_PROGRAM.pt';assert not ap.exists();torch.save(saved,ap);torch.cuda.synchronize()
    result=dict(schema='unsupervised.joint32.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_learned_factor_gain':bool(valid and min(gain.values())>=.05),'pred_c_stable_shared_components':bool(valid and int(matches.sum())>=16)},baseline_squared_relative_error=base_loss,baseline_native_indices=serial(indices),runs=runs,gain_in_captured_energy=gain,stability=stability,usage=profiles,checks=checks,artifact_sha256=digest(ap),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=0,optimizer_steps=480,products=32,linear_readers=64,residual_writer_coefficients=1152*32,program_bytes=ap.stat().st_size,native_weight_saving=0,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()),scope='Unsupervised full-vocabulary coefficient fit;32-product shared component with native remainder. No sparse-loading penalty, no native causal promotion, no independent upstream producer.')
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ['predictions','baseline_squared_relative_error','gain_in_captured_energy','stability','checks','wall_seconds','price']}))
if __name__=='__main__':main()
