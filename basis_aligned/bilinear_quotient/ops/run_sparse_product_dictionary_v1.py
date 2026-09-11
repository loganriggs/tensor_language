#!/usr/bin/env python3
"""A exactness/monotonicity; B coordinate convergence; C capture>=.10/median<=16; D support survival>=.25."""
# BQGATE: 0forwards0seq; weight-only all-token sparse real-product dictionary.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from quadratic_token_dictionary_v1 import embed,conditional
from sparse_product_dictionary_v1 import form,unit_product,grams,report,atom_sweep
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
CACHE=Path('/dev/shm/bilin18_sparse_product_dictionary_v1.pt')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    binding=json.loads((P/'SPARSE_PRODUCT_DICTIONARY_V1_BINDING.json').read_text())
    assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'SPARSE_PRODUCT_DICTIONARY_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,
            corpus_access=False,tokens=50304,products=512,fit_chunk_seconds=540,
            exact_fp64_atom_eigensolves=True,convergence='two frozen-point gap and code-stationarity checks')))
        return
    out=P/'SPARSE_PRODUCT_DICTIONARY_V1_RESULT.json';assert not out.exists() and not CACHE.exists()
    signal.alarm(2400);start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    source_receipt=json.loads((P/'TOKEN_DICTIONARY_DEBIAS_V1_AUDIT.json').read_text())
    source_cache=Path(source_receipt['cache']['path']);assert digest(source_cache)==source_receipt['cache']['sha256']
    source=torch.load(source_cache,weights_only=True,map_location='cpu')
    original_codes=source['codes'].cuda();dictionary=source['dictionary'].cuda()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda()
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    assert u.shape==(50304,1152) and l.shape==r.shape==(4608,1152) and dictionary.shape==(512,1152)
    x,root,basis,scale,mean=embed(u,l,r,down);u=(u-mean)/scale;energy=float(x.square().sum())
    energy_error=abs(float(scale.square()*len(u))/92104252412.19983-1)
    pl=[];pr=[];magnitudes=[];init_start=time.perf_counter()
    for j in range(len(dictionary)):
        left,right,magnitude=unit_product(form(l,r,dictionary[j]@basis))
        pl.append(left);pr.append(right);magnitudes.append(magnitude)
        if (j+1)%64==0:print(json.dumps(dict(stage='initializing',atoms=j+1,elapsed=time.perf_counter()-init_start)),flush=True)
    pl=torch.stack(pl);pr=torch.stack(pr);magnitudes=torch.stack(magnitudes)
    gram,cross=grams(u,l,r,down,pl,pr)
    penalty=float(cross.abs().topk(9,dim=1).values[:,-1].median());assert penalty>0
    codes=original_codes*magnitudes
    projected=report(u,l,r,down,pl,pr,codes,penalty,energy)
    codes,initial_code=conditional(codes,gram,cross,penalty,'codes',2000,1e-6)
    initial=report(u,l,r,down,pl,pr,codes,penalty,energy)
    # Direct full matrix evaluation of eight fixed token rows, independent of implicit Gram objective.
    indices=[0,1,10,100,1000,10000,30000,50256];direct=0.
    for i in indices:direct+=float((form(l,r,u[i]@down)-form(pl,pr,codes[i])).square().sum())
    targets=torch.stack([form(l,r,u[i]@down) for i in indices]);selected=codes[indices]
    implicit=float(targets.square().sum()+((selected.T@selected)*gram).sum()-2*(selected*cross[indices]).sum())
    bridge_error=abs(direct-implicit)/max(direct,1e-30)
    history=[];converged=False;streak=0;previous=initial['objective'];largest_increase=0.
    train_start=time.perf_counter();initialization_seconds=train_start-start
    while time.perf_counter()-train_start<540:
        iteration=len(history)+1;sweep_start=time.perf_counter()
        sweep=atom_sweep(u,l,r,down,pl,pr,codes)
        gram,cross=grams(u,l,r,down,pl,pr)
        codes,code_report=conditional(codes,gram,cross,penalty,'codes',2000,1e-6)
        stats=report(u,l,r,down,pl,pr,codes,penalty,energy)
        decrease=previous-stats['objective'];largest_increase=max(largest_increase,-decrease);previous=stats['objective']
        frozen=None
        if decrease<=1e-5 and stats['code_stationarity']<=1e-4:
            frozen=atom_sweep(u,l,r,down,pl,pr,codes,False)
            streak=streak+1 if frozen['sum_conditional_gain']<=1e-8 else 0
        else:streak=0
        row=dict(iteration=iteration,elapsed=time.perf_counter()-train_start,seconds=time.perf_counter()-sweep_start,
                 **stats,atom_sweep=sweep,codes=code_report,frozen_gap=frozen,convergence_streak=streak)
        history.append(row);print(json.dumps(row),flush=True)
        torch.save(dict(left=pl.cpu(),right=pr.cpu(),codes=codes.cpu(),penalty=penalty,
            scale=float(scale),mean=mean.cpu(),history=history,binding=binding,
            source_cache_sha256=source_receipt['cache']['sha256']),CACHE.with_suffix('.tmp'))
        CACHE.with_suffix('.tmp').replace(CACHE)
        if streak>=2:converged=True;break
    final=report(u,l,r,down,pl,pr,codes,penalty,energy)
    old=original_codes!=0;new=codes!=0;intersection=(old&new).sum(1);union=(old|new).sum(1)
    median_jaccard=float((intersection/union.clamp_min(1)).median())
    cosine=(original_codes*codes).sum(0)/(original_codes.norm(dim=0)*codes.norm(dim=0)).clamp_min(1e-30)
    valid=max(energy_error,bridge_error,final['maximum_atom_norm_error'],largest_increase)<=1e-8
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_coordinate_convergence':valid and converged,
        'pred_c_capture_and_sparsity':valid and converged and final['captured_energy']>=.10 and final['median_active']<=16,
        'pred_d_token_support_continuity':valid and converged and median_jaccard>=.25},
        status='coordinatewise_converged' if converged else 'optimization_unfinished',
        projected_initial=projected,initial=initial,final=final,penalty=penalty,
        initial_code_report=initial_code,exact_energy_error=energy_error,direct_matrix_bridge_error=bridge_error,
        maximum_outer_objective_increase=largest_increase,median_token_support_jaccard=median_jaccard,
        median_signed_writer_cosine=float(cosine.median()),history=history,
        cache=dict(path=str(CACHE),sha256=digest(CACHE),bytes=CACHE.stat().st_size,ephemeral=True),
        price=dict(products=512,input_reader_numbers=2*512*1152,nonzero_output_coefficients=final['nonzero_codes'],
                   sparse_indices_also_required=True,common_channel_retained=True,body_forwards=0,corpus_access=False),
        initialization_seconds=initialization_seconds,fit_seconds=time.perf_counter()-train_start,wall_seconds=time.perf_counter()-start,
        scope='Joint sparse token usage and single real-product input functions. Coordinate stationarity, not global optimum; support continuity follows indices, not canonical identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
