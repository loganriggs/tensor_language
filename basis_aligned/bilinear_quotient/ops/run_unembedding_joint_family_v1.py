#!/usr/bin/env python3
# BQGATE: userfile fixed is/are/was/were3-product program,0nativeforwards.
"""pred_a algebra/fold/full-vocabulary projection identities; pred_b shared
number contrast and all4 joint quadratic errors<=.10; pred_c both tense means
single-product error<=.10 and shared linear-factor cosine>=.99. No rank sweep.
"""
import json,os,sys,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(POLY)]
import torch
import tiktoken
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
from joint_weight_composition_v1 import input_product_gram,full_output_code_grams
OUT=POLY/'UNEMBEDDING_JOINT_FAMILY_V1_RESULT.json';BIND=POLY/'UNEMBEDDING_JOINT_FAMILY_V1_BINDING.json'
CHECKPOINT=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def sym(x):return (x+x.T)/2
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
def product_fit(S):
    vals,vecs=torch.linalg.eigh(S);plus=vals[-1].clamp_min(0).sqrt()*vecs[:,-1];minus=(-vals[0]).clamp_min(0).sqrt()*vecs[:,0]
    a=plus+minus;b=plus-minus;fit=sym(a[:,None]*b[None,:]);return a,b,fit,dict(relative_error=rel(fit,S),positive=float(vals[-1]),negative=float(vals[0]),norm=float(S.norm()))
def fixture():
    eye=torch.eye(5,dtype=torch.float64);a,b,p,t,n=eye;u=a+b
    P=sym(u[:,None]*p[None,:]);T=sym(u[:,None]*t[None,:]);N=sym(a[:,None]*n[None,:]);S=[P+N,P-N,T+N,T-N]
    fits=[product_fit(x) for x in [P,T,N]]
    errs=[v[3]['relative_error'] for v in fits];assert max(errs)<=1e-10
    assert torch.equal((S[0]-S[1])/2,(S[2]-S[3])/2)
    return dict(product_errors=errs,shared_contrast_exact=True)
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items());torch.set_num_threads(2);tiny=fixture()
    enc=tiktoken.get_encoding('gpt2');tokens=[' is',' are',' was',' were'];ids=[enc.encode(t) for t in tokens];assert all(len(x)==1 for x in ids);ids=[x[0] for x in ids]
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,tokens=tokens,ids=ids,fixture=tiny)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True);U=state['lm_head.weight'].cuda();D=state['transformer.h.17.mlp.Down.weight'].double().cuda();L=state['transformer.h.17.mlp.Left.weight'].double().cuda();R=state['transformer.h.17.mlp.Right.weight'].double().cuda()
    C=U[ids].double()@D;S=torch.stack([sym(L.T@(c[:,None]*R)) for c in C]);P=(S[0]+S[1])/2;T=(S[2]+S[3])/2;Np=(S[0]-S[1])/2;Nt=(S[2]-S[3])/2;N=(Np+Nt)/2
    fitted=[product_fit(x) for x in [P,T,N]];aa=torch.stack([x[0] for x in fitted]);bb=torch.stack([x[1] for x in fitted]);H=torch.stack([x[2] for x in fitted]);program=torch.stack([H[0]+H[2],H[0]-H[2],H[1]+H[2],H[1]-H[2]])
    errors=[rel(v,s) for v,s in zip(program,S)];contrast=float((Np-Nt).norm()/((Np.norm().square()+Nt.norm().square())/2).sqrt().clamp_min(1e-30))
    f1=torch.stack([aa[0],bb[0]]);f2=torch.stack([aa[1],bb[1]]);cos=(f1/f1.norm(dim=-1,keepdim=True).clamp_min(1e-30))@(f2/f2.norm(dim=-1,keepdim=True).clamp_min(1e-30)).T
    cross=((L@aa.T)*(R@bb.T)+(L@bb.T)*(R@aa.T))/2;G=H.flatten(1)@H.flatten(1).T;pinv=torch.linalg.pinv(G,rtol=1e-10);W=D@cross@pinv
    # All vocabulary rows contribute to numerator-space accounting, streamed.
    GO,GC,mean=full_output_code_grams(U,D);GI=input_product_gram(L,R);total=(GO*GI).sum();del GC
    Bgram=torch.zeros(3,3,dtype=torch.float64,device='cuda');vocab_coeff=[]
    for block in U.split(512):
        coeff=block.double()@W;Bgram+=coeff.T@coeff;vocab_coeff.append(coeff.cpu())
    retained=(Bgram*G).sum();coef4=U[ids].double()@W;projected=torch.einsum('vj,jab->vab',coef4,H)
    expected=C@cross@pinv;checks=dict(projection_coefficients=rel(coef4,expected),native_cross=rel(C@cross,S.flatten(1)@H.flatten(1).T))
    x=torch.arange(1152*4,device='cuda',dtype=torch.float64).reshape(4,1152).sin()/10
    direct=((x@L.T)*(x@R.T))@C.T;quad=torch.einsum('bi,vij,bj->bv',x,S,x);checks['fold_values']=rel(quad,direct)
    product=(x@aa.T)*(x@bb.T);checks['product_values']=rel(product,torch.einsum('bi,jik,bk->bj',x,H,x))
    valid=max(checks.values())<=1e-8 and checks['fold_values']<=1e-9 and checks['product_values']<=1e-9 and max(tiny['product_errors'])<=1e-10 and all(bool(torch.isfinite(t).all()) for t in [S,H,W]) and min(float(v.norm()) for v in [Np,Nt,P,T,N])>1e-8
    joint=valid and contrast<=.1 and max(errors)<=.1;shared=valid and all(x[3]['relative_error']<=.1 for x in fitted[:2]) and float(cos.abs().max())>=.99
    ap=POLY/'UNEMBEDDING_JOINT_FAMILY_V1_PROGRAM.pt';assert not ap.exists();torch.save(dict(tokens=tokens,ids=ids,a=aa.cpu(),b=bb.cpu(),residual_writers=W.cpu(),full_vocabulary_coefficients=torch.cat(vocab_coeff),selected_native_quadratics=S.cpu(),selected_program_quadratics=program.cpu(),selected_projected_quadratics=projected.cpu()),ap)
    torch.cuda.synchronize();result=dict(schema='unembedding.joint_family.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_three_product_joint_program':bool(joint),'pred_c_shared_linear_factor':bool(shared)},tokens=tokens,ids=ids,number_contrast_relative_mismatch=contrast,joint_quadratic_errors=errors,product_fits={k:v[3] for k,v in zip(['present_mean','past_mean','shared_number'],fitted)},mean_factor_cosines=cos.cpu().tolist(),full_vocabulary_projection=dict(coefficient_energy_fraction=float(retained/total),selected_projection_errors=[rel(v,s) for v,s in zip(projected,S)],gram_eigenvalues=torch.linalg.eigvalsh(G).cpu().tolist()),checks=checks,fixture=tiny,artifact_sha256=digest(ap),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=0,full_vocabulary=50304,products=3,linear_forms=6,program_tensor_bytes=ap.stat().st_size,native_weight_saving=0,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()),scope='Fixed related-output family and joint3-product DAG screen from user math file. Full-vocabulary writes retained in component projection. Numerator geometry, not full logits, native removal or extracted upstream input producer.')
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ['predictions','number_contrast_relative_mismatch','joint_quadratic_errors','product_fits','mean_factor_cosines','checks','wall_seconds']}))
if __name__=='__main__':main()
