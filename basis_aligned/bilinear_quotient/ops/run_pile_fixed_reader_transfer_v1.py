#!/usr/bin/env python3
# BQGATE: 0forwards0seq; cached58368states, five fixed-reader transfer/refit tests.
"""pred_a numerical/shape/refit instrument; pred_b matched-position shared-reader
transfer <=1.5x old validation; pred_c writer refit reduces matched error>=10%.
Null: input functions do not transfer, or writers need no recalibration.
Preregistration contains frozen scopes, bars and literal price. No test access.
"""
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(P)]
import torch
from structured_quadratic_models_v1 import QuadraticModel
from stable_empirical_quadratic_v1 import qr_writers
from prepare_million_token_panel_v1 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
ARTIFACTS={'data_product':'STRUCTURED_FIT_V2_data_product_s0_CHUNK_00_BEST.pt','data_shared_reader':'STRUCTURED_FIT_V2_data_shared_reader_s0_CHUNK_00_BEST.pt','data_block':'STRUCTURED_FIT_V2_data_block_s0_CHUNK_00_BEST.pt','weight_product':'STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00_BEST.pt','penalized_weight_product':'PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_00_BEST.pt'}
BIND=P/'PILE_FIXED_READER_TRANSFER_V1_BINDING.json';OUT=P/'PILE_FIXED_READER_TRANSFER_V1_RESULT.json'

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    control=json.loads((P/'STABLE_EMPIRICAL_QUADRATIC_V1_CONTROL.json').read_text());assert control['passed']
    receipt=json.loads((P/'MILLION_TOKEN_PANEL_V1_RESULT.json').read_text());assert all(receipt['predictions'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,training_states=51200,validation_states=7168,functions=list(ARTIFACTS),test_access=False)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u;root=torch.linalg.cholesky(metric)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    data=torch.load(P/'MILLION_TOKEN_PANEL_V1_INPUTS.pt',map_location='cpu',weights_only=True,mmap=True)
    panels={}
    for split,n in [('train',51200),('validation',7168)]:
        ids=data[split+'_rows'];x=(data['x'][ids].float()*data['x_scale'][ids]).reshape(-1,1152).double().cuda();assert len(x)==n
        chunks=[]
        for start in range(0,n,1024):
            xx=x[start:start+1024];chunks.append((((xx@l.T)*(xx@r.T))@d.T)@root)
        z=torch.cat(chunks);assert torch.isfinite(z).all()
        panels[split]=(x,z,data['positions'][ids].reshape(-1).cuda()>=64)
    x,z,_=panels['train'];xt,zt,matched=panels['validation']
    def errors(pred,target,mask=None):
        out={'all_positions':float((pred-target).square().sum()/target.square().sum())}
        if mask is not None:out['positions_ge64']=float((pred[mask]-target[mask]).square().sum()/target[mask].square().sum())
        return out
    results={};saved={};valid=True
    for name,filename in ARTIFACTS.items():
        frozen=torch.load(P/filename,map_location='cuda',weights_only=False)
        model=QuadraticModel(frozen['config']['kind'],1152,device='cuda');model.load_state_dict(frozen['best']['model']);a,b,c=model.components()
        f=((x@a.T)*(x@b.T))@c;ft=((xt@a.T)*(xt@b.T))@c
        oldbeta=frozen['best']['writer'].T@root
        begin=time.perf_counter();beta,rr=qr_writers(f,z);qr_seconds=time.perf_counter()-begin
        uu,ss,vh=torch.linalg.svd(f,full_matrices=False);svdbeta=(vh.T/ss)@(uu.T@z)
        normal=torch.linalg.solve(f.T@f,f.T@z)
        prediction=f@beta;pn=prediction.norm().clamp_min(1e-30)
        svderr=float((f@svdbeta-prediction).norm()/pn);normalerr=float((f@normal-prediction).norm()/pn)
        oldtrain=errors(f@oldbeta,z)['all_positions'];newtrain=errors(prediction,z)['all_positions']
        good=svderr<=1e-8 and normalerr<=1e-6 and newtrain<=oldtrain+1e-8 and bool(torch.isfinite(beta).all()) and float(ss[-1])>0
        valid=valid and good
        results[name]=dict(frozen=filename,parameter_numbers=model.storage_numbers()+frozen['best']['writer'].numel(),frozen_training_error=oldtrain,refit_training_error=newtrain,frozen_validation=errors(ft@oldbeta,zt,matched),refit_validation=errors(ft@beta,zt,matched),qr_svd_function_relative_error=svderr,qr_normal_function_relative_error=normalerr,feature_condition=float(ss[0]/ss[-1]),qr_seconds=qr_seconds,instrument_passed=bool(good))
        saved[name]=dict(source=filename,writer=torch.linalg.solve_triangular(root.T,beta.T,upper=True).cpu())
        print(json.dumps(dict(name=name,**results[name])),flush=True)
    # Training-only calibration controls; keep native output metric identical.
    xa=torch.cat([x,torch.ones(len(x),1,device=x.device,dtype=x.dtype)],1);xta=torch.cat([xt,torch.ones(len(xt),1,device=xt.device,dtype=xt.dtype)],1)
    ag=xa.T@xa;affine_condition=float(torch.linalg.cond(ag));assert affine_condition<=1e12
    affine=torch.linalg.solve(ag,xa.T@z);constant=z.mean(0)
    baselines=dict(constant=errors(constant.expand_as(zt),zt,matched),affine=errors(xta@affine,zt,matched),affine_gram_condition=affine_condition,affine_parameters=1152*1153)
    shared=results['data_shared_reader'];frozenerr=shared['frozen_validation']['positions_ge64'];refiterr=shared['refit_validation']['positions_ge64']
    ap=P/'PILE_FIXED_READER_TRANSFER_V1_WRITERS.pt';assert not ap.exists();torch.save(saved,ap)
    result=dict(schema='pile.fixed.reader.transfer.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_shared_input_transfer':bool(valid and frozenerr<=1.5*.019442108714888683),'pred_c_shared_writer_recalibration':bool(valid and refiterr<=.9*frozenerr)},results=results,baselines=baselines,validation_position_counts=dict(all=7168,positions_ge64=int(matched.sum())),artifact_sha256=digest(ap),binding_sha256=digest(BIND),runner_sha256=digest(RUNNER),price=dict(body_forwards=0,training_states=51200,validation_states=7168,writer_artifact_bytes=ap.stat().st_size),wall_seconds=time.perf_counter()-tic,scope='Frozen input functions and training-only writer refits on named Pile panel. No nonlinear training, test split, semantic labels or circuit identification. Matched-position comparisons control the old panel exclusion of first64positions; corpus/prefix/near-duplicate limitations remain.')
    atomic_create_json(OUT,result);print(json.dumps(result))

if __name__=='__main__':main()
