#!/usr/bin/env python3
# BQGATE: 0forwards0seq; weight-only overlapping block manifold fit240s.
import os,sys,json,time,signal,hashlib,shutil
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from multioutput_quadratic_blocks_v1 import MultioutputQuadraticBlocks,MultioutputWeightObjective
from fit_multioutput_manifold_v1 import fit
from joint_quadratic_fit_v1 import product_cross
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    if isinstance(v,list):return [cpu(x) for x in v]
    return v
def main():
    binding=json.loads((P/'MULTIOUTPUT_MANIFOLD_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    for f in ['MULTIOUTPUT_MANIFOLD_V1_CONTROL.json','MULTIOUTPUT_BLOCK_ORTHOGONAL_GAUGE_V1_CONTROL.json']:assert json.loads((P/f).read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,seconds=240,groups=16,rank=16,outputs=4,penalty=.01)));return
    out=P/'MULTIOUTPUT_MANIFOLD_V1_RESULT.json';assert not out.exists();assert shutil.disk_usage(P).free>20_000_000
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']];total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    source=torch.load(P/'MULTIOUTPUT_BLOCK_ORTHOGONAL_GAUGE_V1_INITIAL.pt',weights_only=False,map_location='cuda')
    model=MultioutputQuadraticBlocks(1152,groups=16,rank=16,outputs=4,device='cuda');model.load_state_dict(source['model'])
    obj=MultioutputWeightObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    first=obj.terms(model);initial_bridge=abs(float(first[0].detach())-.914566897645157);initial_raw_bridge=abs(float(first[1].detach())-.9137697876842366)
    assert max(initial_bridge,initial_raw_bridge)<=1e-10
    state=fit(model,obj,seconds=240);final=state['history'][-1]
    replay=abs(float(obj.terms(model)[0].detach())-final['optimization_loss'])
    orth=max(max(v['orthogonality_error'],v['sphere_error'],v['gram_diagonal_error']) for v in state['history'])
    valid=max(initial_bridge,initial_raw_bridge,replay,orth,state['maximum_objective_increase'])<=1e-10
    ap=P/'MULTIOUTPUT_MANIFOLD_V1_CHECKPOINT.pt';assert not ap.exists()
    saved=cpu(state);saved.update(config=source['config'],next_chunk=1,metric='full_U',penalty=.01)
    torch.save(saved,ap)
    improvement=state['initial_loss']-final['optimization_loss'];retained=final['captured_energy_fraction']>=1-.9137697876842366-1e-6
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_converged':valid and state['converged'],'pred_c_improved_and_retained':valid and improvement>=1e-6 and retained},initial=state['history'][0],final=final,terminal_reason=state['terminal_reason'],objective_improvement=improvement,raw_capture_retained_clause=retained,maximum_objective_increase=state['maximum_objective_increase'],initial_bridge=initial_bridge,initial_raw_bridge=initial_raw_bridge,final_replay=replay,price=dict(body_forwards=0,float_numbers=model.storage_numbers()+1152*64,iterations=state['iteration'],objective_evaluations=state['chunk_evaluations'],backtracks=state['chunk_backtracks'],direction_restarts=state['chunk_direction_restarts'],fit_seconds=state['chunk_seconds'],inherited_fit_seconds=540.148,checkpoint_bytes=ap.stat().st_size),checkpoint_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,scope='Samefull-U lambda.01 overlapping16x16x4block representation. Newmanifoldcoordinates; projectedgradientnotnumericallyidenticaltooldrawgaugegradient. No data/globaloptimum/circuitclaim.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
