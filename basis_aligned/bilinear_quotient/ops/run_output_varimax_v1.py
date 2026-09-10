#!/usr/bin/env python3
# BQGATE: 0forwards0seq; centered weight outputvarimax128 functions240s.
import os,sys,json,time,signal,hashlib,shutil
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from orthogonal_output_varimax_v1 import output_factors,fit,sparsity
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    return v
def main():
    binding=json.loads((P/'OUTPUT_VARIMAX_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'OUTPUT_VARIMAX_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,seconds=240,output_functions=128,criterion='raw_varimax_global_scale_only')));return
    out=P/'OUTPUT_VARIMAX_V1_RESULT.json';assert not out.exists();assert shutil.disk_usage(P).free>20_000_000
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda()
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    f=output_factors(u,l,r,d,128);loadings=f['loadings'];core=f['core'];ev=f['eigenvalues'];total=ev.sum()
    expected=92104252412.19983;trace_bridge=abs(float(total)/expected-1)
    orth=float((core@f['native_gram']@core.T-torch.eye(128,dtype=core.dtype,device=core.device)).abs().max())
    loading_bridge=float((loadings.T@loadings-torch.diag(ev[:128])).norm()/ev[:128].norm())
    assert max(trace_bridge,orth,loading_bridge)<=1e-10
    before=sparsity(loadings);state=fit(loadings,seconds=240);rotation=state['rotation'];rotated=loadings@rotation;after=sparsity(rotated)
    # Replay preserved rank128 tensor on fixed random quadratic input probes.
    torch.manual_seed(285);x=torch.randn(16,1152,device=l.device,dtype=l.dtype);h=(l@x.T)*(r@x.T)
    original=loadings@(core@h);replayed=rotated@((rotation.T@core)@h)
    replay=float((original-replayed).norm()/original.norm())
    energy_bridge=abs(float(rotated.square().sum()/loadings.square().sum())-1)
    final=state['history'][-1];valid=max(trace_bridge,orth,loading_bridge,replay,energy_bridge,final['orthogonality_error'],state['maximum_score_decrease'])<=1e-10
    ap=P/'OUTPUT_VARIMAX_V1_CHECKPOINT.pt';assert not ap.exists()
    saved=cpu(state);saved.update(writer=f['writer'].cpu(),core=core.cpu(),unembedding_mean=f['mean'].cpu(),eigenvalues=ev.cpu(),config=dict(output_functions=128,metric='centered_U_weight',criterion='raw_varimax_global_scale_only'),next_chunk=1)
    torch.save(saved,ap)
    reduction=after['median_token_factor_participation']<=.75*before['median_token_factor_participation'];top4=after['token_top4_energy_fraction']>=.5
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_converged':valid and state['converged'],'pred_c_sparse_overlapping_usage':valid and reduction and top4},before=before,after=after,median_participation_reduction_clause=reduction,top4_energy_clause=top4,initial=state['history'][0],final=final,centered_rank128_capture=float(ev[:128].sum()/total),terminal_reason=state['terminal_reason'],bridges=dict(native_total=trace_bridge,core_orthogonality=orth,loadings_gram=loading_bridge,rotation_replay=replay,rotation_energy=energy_bridge),price=dict(body_forwards=0,native_input_reader_numbers=l.numel()+r.numel(),dense_core_mixture_numbers=core.numel(),residual_writer_numbers=f['writer'].numel(),native_unembedding_numbers=u.numel(),native_remainder_down_numbers=d.numel(),rotation_numbers=rotation.numel(),separate_exact_common_symmetric_numbers=664128,iterations=state['iteration'],fit_seconds=state['chunk_seconds'],backtracks=state['chunk_backtracks'],direction_restarts=state['chunk_direction_restarts'],checkpoint_bytes=ap.stat().st_size),checkpoint_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,scope='Fixedoptimalcenteredoutputrank128projection; orthogonalrotationdoesnotimprovefiterror. Sparseoverlappingtokenloadingshypothesis, noclusteringorinputlowrankassumption. No recoveredsemanticcircuit/independence/globaloptimumclaim.')
    with out.open('x') as fp:json.dump(result,fp,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
