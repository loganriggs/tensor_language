#!/usr/bin/env python3
"""A numerical fidelity; B originalFP64stationarity; C>=30%capture/median<=16."""
# BQGATE: 0forwards0seq; exact all-token coefficient embedding, sparse dictionary.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from quadratic_token_dictionary_v1 import embed,soft,cycle,diagnostics
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
CACHE=Path('/dev/shm/bilin18_token_function_dictionary_v1.pt')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'TOKEN_FUNCTION_DICTIONARY_V1_BINDING.json').read_text())
    assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'QUADRATIC_TOKEN_DICTIONARY_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            tokens=50304,exact_feature_width=1152,dictionary_size=512,seconds=540,
            code_solver_steps=200,dictionary_solver_steps=200,final_fp64_stationarity=1e-4)))
        return
    out=P/'TOKEN_FUNCTION_DICTIONARY_V1_RESULT.json';assert not out.exists() and not CACHE.exists()
    signal.alarm(1200);start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda()
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    assert u.shape==(50304,1152) and l.shape==r.shape==(4608,1152)
    x64,root,native_basis,scale,mean=embed(u,l,r,down)
    energy_error=abs(float(scale.square()*len(u))/92104252412.19983-1)
    ids=torch.tensor([0,1,10,100,1000,10000,30000,50256],device='cuda')
    forms=[]
    for coefficient in (u[ids]-mean)@down:
        q=(l.T*coefficient)@r;forms.append(((q+q.T)/2).flatten()/scale)
    forms=torch.stack(forms);direct=forms@forms.T;implicit=x64[ids]@x64[ids].T
    embedding_error=float((direct-implicit).norm()/direct.norm())
    eigen,vectors=torch.linalg.eigh(x64.T@x64)
    dictionary=vectors[:,-512:].T.flip(0).float().contiguous();x=x64.float()
    initial_projection=x@dictionary.T
    penalty=float(initial_projection.abs().topk(9,dim=1).values[:,-1].median())
    assert penalty>0
    codes=soft(initial_projection,penalty)
    initial=diagnostics(x64,codes.double(),dictionary.double(),penalty)
    best_value=initial['objective'];best_codes=codes.clone();best_dictionary=dictionary.clone()
    history=[];streak=0;converged=False;previous=initial['objective'];largest_increase=0.;iteration=0
    training_start=time.perf_counter()
    while time.perf_counter()-training_start<540:
        iteration+=1;codes,dictionary,conditional_report=cycle(x,codes,dictionary,penalty,200,1e-5)
        stats=diagnostics(x,codes,dictionary,penalty)
        largest_increase=max(largest_increase,stats['objective']-previous);previous=stats['objective']
        if stats['objective']<best_value:
            best_value=stats['objective'];best_codes=codes.clone();best_dictionary=dictionary.clone()
        verified=None
        if stats['relative_stationarity']<=2e-4:
            verified=diagnostics(x64,codes.double(),dictionary.double(),penalty)
            streak=streak+1 if verified['relative_stationarity']<=1e-4 else 0
        else:streak=0
        row=dict(iteration=iteration,elapsed=time.perf_counter()-training_start,**stats,
                 conditional=conditional_report,fp64_check=verified,convergence_streak=streak)
        history.append(row)
        if iteration==1 or iteration%5==0 or streak:
            print(json.dumps(row),flush=True)
        if streak>=2:
            converged=True;break
    final32=diagnostics(x,codes,dictionary,penalty)
    final=diagnostics(x64,codes.double(),dictionary.double(),penalty)
    objective_error=abs(final32['objective']-final['objective'])
    valid=max(energy_error,embedding_error)<=1e-8 and objective_error<=1e-5 and final['atom_norm_max']<=1+1e-6 and bool(torch.isfinite(codes).all()) and bool(torch.isfinite(dictionary).all())
    torch.save(dict(codes=codes.cpu(),dictionary=dictionary.cpu(),best_codes=best_codes.cpu(),
        best_dictionary=best_dictionary.cpu(),penalty=penalty,scale=float(scale),mean=mean.cpu(),
        initial=initial,history=history,final=final,binding=binding),CACHE)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_converged':valid and converged,
        'pred_c_quality_and_sparsity':valid and converged and final['captured_energy']>=.30 and final['median_active']<=16},
        initial=initial,final=final,final_fp32=final32,penalty=penalty,
        status='locally_converged' if converged else 'optimization_unfinished',
        terminal_reason='two_fp64_stationarity_checks' if converged else 'time_limit',
        exact_centered_energy_error=energy_error,exact_embedding_gram_error=embedding_error,
        fp32_fp64_objective_absolute_difference=objective_error,maximum_outer_objective_increase=largest_increase,
        history=history,cache=dict(path=str(CACHE),sha256=digest(CACHE),bytes=CACHE.stat().st_size,ephemeral=True),
        price=dict(body_forwards=0,corpus_access=False,exact_feature_width=1152,dictionary_size=512,
                   dictionary_coordinate_numbers=512*1152,nonzero_codes=final['nonzero_codes'],
                   input_function_basis_still_required=True,fit_seconds=time.perf_counter()-training_start),
        wall_seconds=time.perf_counter()-start,
        scope='Single-start nonorthogonal sparse dictionary over exact centered token quadratic functions; common channel retained separately. No global optimum or circuit identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
