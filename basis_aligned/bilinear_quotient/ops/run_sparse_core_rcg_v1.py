#!/usr/bin/env python3
# BQGATE: 0forwards0seq; centered-weight sparseframe RCG240s.
import os,sys,json,time,signal,hashlib,shutil
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from fit_sparse_core_rcg_v1 import fit
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    binding=json.loads((P/'SPARSE_CORE_RCG_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    for f in ['SPARSE_CORE_RCG_V1_CONTROL.json','SPARSE_CORE_STIEFEL_V1_CONTROL.json','SPARSE_ORTHOGONAL_CORE_V1_CONTROL.json']:assert json.loads((P/f).read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,seconds=240,readers=128,edges=256)));return
    out=P/'SPARSE_CORE_RCG_V1_RESULT.json';assert not out.exists();assert shutil.disk_usage(P).free>20_000_000
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u;root=torch.linalg.cholesky(metric).T
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']];z=root@d
    source=torch.load(P/'SPARSE_ORTHOGONAL_CORE_V1_CENTERED_COMPACT.pt',weights_only=False,map_location='cuda');q=source['basis'].T.clone();baseline=json.loads((P/'SPARSE_ORTHOGONAL_CORE_V1_RESULT.json').read_text());total=q.new_tensor(baseline['native_centered_total'])
    state=fit(q,l,r,z,total,count=256,seconds=240)
    q=state['q'];i,j=state['edges'];a=l@q;b=r@q;scale=torch.where(i==j,torch.full_like(i,2,dtype=q.dtype),torch.full_like(i,2**.5,dtype=q.dtype));coeff=z@((a[:,i]*b[:,j]+a[:,j]*b[:,i])/scale)
    writer=torch.linalg.solve_triangular(root,coeff,upper=True);capture=float(coeff.square().sum()/total);replay=abs(capture-state['history'][-1]['captured_energy']);initial=abs(state['initial_capture']-baseline['centered_coefficient_capture'])
    writer_bridge=float((root@writer-coeff).norm()/coeff.norm());orth=max(v['orthogonality_error'] for v in state['history']);valid=max(replay,initial,writer_bridge,orth,state['maximum_score_decrease'])<=1e-10
    final=state['history'][-1];ap=P/'SPARSE_CORE_RCG_V1_CHECKPOINT.pt';assert not ap.exists()
    saved={k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in state.items()};saved.update(writer=writer.cpu(),unembedding_mean=mean.cpu(),config=dict(readers=128,edges=256,metric='centered_weight'),next_chunk=1)
    torch.save(saved,ap)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_converged':valid and state['converged'],'pred_c_centered_capture':valid and capture>=.05},initial_capture=state['initial_capture'],final=final,terminal_reason=state['terminal_reason'],maximum_score_decrease=state['maximum_score_decrease'],initial_bridge=initial,final_replay=replay,writer_bridge=writer_bridge,price=dict(body_forwards=0,centered_float_numbers=442368,edge_indices=512,additional_exact_common_numbers=664128,iterations=state['iteration'],full_core_selections=state['chunk_selections'],backtracks=state['chunk_backtracks'],direction_restarts=state['chunk_direction_restarts'],fit_seconds=state['chunk_seconds'],checkpoint_bytes=ap.stat().st_size),checkpoint_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,scope='Weight-only frameoptimization, exactconditionaloutput/edges. Commonchannelpreservedseparately. No globaloptimumorsemanticcircuitclaim.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
