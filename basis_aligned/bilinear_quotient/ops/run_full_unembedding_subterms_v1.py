#!/usr/bin/env python3
# BQGATE: all50304 unembedding rows,4608 product subterms,0nativeforwards.
"""pred_a exact Gram/algebra controls; pred_b >=8 shared-input/different-output
products; pred_c >=8 shared-output/different-input products. Errors<=.10,
other side error>.50, full and tokenizer-valid vocabulary, signed comparisons.
No fitting/rank sweep; original weights retained. Null no primitive sharing.
"""
import os,json,sys,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(POLY)]
import torch
from joint_weight_composition_v1 import input_product_gram,full_output_code_grams,nearest_signed_profile
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'FULL_UNEMBEDDING_SUBTERMS_V1_RESULT.json';BIND=POLY/'FULL_UNEMBEDDING_SUBTERMS_V1_BINDING.json'
CHECKPOINT=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def serial(x):return x.detach().cpu().tolist()
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
def pair_error(gram,partners):
    ix=torch.arange(len(gram),device=gram.device);return (1-gram[ix,partners].square()/(gram.diag()*gram[partners,partners]).clamp_min(1e-30)).clamp_min(0).sqrt()
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    fixture=json.loads((POLY/'JOINT_WEIGHT_COMPOSITION_V1_FIXTURE_RESULT.json').read_text());assert max(fixture[k] for k in ['full_unembedding_fold_max','product_antipodal_max','full_output_gram_max','centered_output_gram_max'])<=1e-10
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,vocabulary=50304,products=4608,max_analysis_tensor_bytes=4608**2*8)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True)
    U=state['lm_head.weight'].cuda();D=state['transformer.h.17.mlp.Down.weight'].double().cuda();L=state['transformer.h.17.mlp.Left.weight'].double().cuda();R=state['transformer.h.17.mlp.Right.weight'].double().cuda()
    assert U.shape==(50304,1152) and D.shape==(1152,4608)
    GI=input_product_gram(L,R);ip=nearest_signed_profile(GI);GO,GC,mean=full_output_code_grams(U,D);op=nearest_signed_profile(GC)
    # Full-width independent explicit controls, using fixed columns including endpoints.
    ix=torch.tensor([0,1,4607],device='cuda');a=torch.cat([block.double()@D[:,ix] for block in U.split(512)]);H=(L[ix,:,None]*R[ix,None,:]+R[ix,:,None]*L[ix,None,:])/2
    checks=dict(output_gram=rel(GO[ix][:,ix],a.T@a),centered_output_gram=rel(GC[ix][:,ix],(a-a.mean(0)).T@(a-a.mean(0))),input_gram=rel(GI[ix][:,ix],H.flatten(1)@H.flatten(1).T))
    input_partner_output_error=pair_error(GC,ip['partner']);output_partner_input_error=pair_error(GI,op['partner']);output_error=op['error'];raw_code_error=pair_error(GO,op['partner']);del GO,GC
    GV,CV,meanvalid=full_output_code_grams(U[:50257],D)
    input_partner_valid_error=pair_error(CV,ip['partner']);output_partner_valid_error=pair_error(CV,op['partner'])
    input_candidates=ip['valid']&(ip['error']<=.1)&(input_partner_output_error>.5)&(input_partner_valid_error>.5)
    output_candidates=op['valid']&(output_error<=.1)&(output_partner_valid_error<=.1)&(output_partner_input_error>.5)
    valid=max(checks.values())<=1e-9 and all(bool(torch.isfinite(t).all()) for t in [GI,GV,CV])
    rawD=D.T@D
    summaries=dict(input_match={k:serial(v) for k,v in ip.items()},output_match={k:serial(v) for k,v in op.items()},input_partner_output_error=serial(input_partner_output_error),input_partner_valid_output_error=serial(input_partner_valid_error),output_partner_input_error=serial(output_partner_input_error),output_partner_valid_error=serial(output_partner_valid_error),output_partner_uncentered_error=serial(raw_code_error),output_partner_raw_D_error=serial(pair_error(rawD,op['partner'])),input_candidates=serial(input_candidates),output_candidates=serial(output_candidates))
    torch.cuda.synchronize();result=dict(schema='full_unembedding.subterms.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_shared_input_operations':bool(valid and input_candidates.sum()>=8),'pred_c_shared_output_operations':bool(valid and output_candidates.sum()>=8)},input_candidate_count=int(input_candidates.sum()),output_candidate_count=int(output_candidates.sum()),checks=checks,summaries=summaries,runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=0,full_vocabulary=50304,valid_vocabulary=50257,native_products=4608,native_weight_saving=0,max_analysis_tensor_bytes=4608**2*8,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()),scope='Full-vocabulary shared product/consumer geometry. Original factorization retained, no identified circuit or behavioral sufficiency.')
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ['predictions','input_candidate_count','output_candidate_count','checks','wall_seconds','price']}))
if __name__=='__main__':main()
