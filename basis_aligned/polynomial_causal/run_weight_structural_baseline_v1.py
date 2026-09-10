"""Native-weight structural baseline; no data access."""
import os,json,time,signal,hashlib,shutil
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from multioutput_quadratic_blocks_v1 import MultioutputQuadraticBlocks,MultioutputWeightObjective
from convergent_quadratic_fit_v2 import advance
from joint_quadratic_fit_v1 import product_cross
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(kind,registered_predictions):
    tag='WEIGHT_STRUCTURAL_BASELINE_V1_'+kind
    binding=json.loads((P/(tag+'_BINDING.json')).read_text());assert all(digest(p)==h for p,h in binding.items())
    for c in ['ENERGY_REGULARIZED_QUADRATIC_V1_CONTROL.json','MULTIOUTPUT_QUADRATIC_BLOCKS_V1_CONTROL.json','PENALIZED_CONVERGENCE_V1_CONTROL.json','OPTIMIZER_STALL_V2_CONTROL.json']:assert json.loads((P/c).read_text())['passed']
    count=589824 if kind=='square' else 377344
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,kind=kind,parameters=count,seconds=540)));return
    out=P/(tag+'_RESULT.json');assert not out.exists();assert shutil.disk_usage(P).free>=120_000_000,'Insufficient checkpoint headroom'
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    if kind=='square':model=QuadraticModel('square',1152,device='cuda');cls=EnergyRegularizedObjective;writers=256
    else:model=MultioutputQuadraticBlocks(1152,device='cuda');cls=MultioutputWeightObjective;writers=64
    assert model.storage_numbers()+1152*writers==count
    objective=cls(metric,total,l=l,r=r,d=d,penalty=.01)
    state=advance(model,objective,seconds=540)
    state['config']=dict(kind=kind,metric='weight',seed=0,penalty=.01);state['next_chunk']=1
    ap=P/(tag+'_CHECKPOINT.pt');assert not ap.exists();torch.save(state,ap)
    model.load_state_dict(state['best']['model'])
    replay=abs(float(objective.loss(model)[0].detach())-state['best']['diagnostics']['optimization_loss'])
    best=state['best']['diagnostics'];valid=replay<=1e-10 and best['squared_relative_error']>=-1e-8 and best['regularized_gram_condition']<=1e12
    result=dict(schema='weight.structural.baseline.v1',kind=kind,predictions={'pred_a_instrument':valid,'pred_b_converged':valid and state['converged'],'pred_c_capture_gain':valid and best['captured_energy_fraction']>=.09698912596},best=best,last=state['history'][-1],converged=state['converged'],terminal_reason=state['terminal_reason'],replay_absolute_error=replay,price=dict(body_forwards=0,parameter_numbers=count,closures=state['closures'],fit_seconds=state['chunk_seconds'],checkpoint_bytes=ap.stat().st_size),checkpoint_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,scope='Weight-only first seed/first chunk. Unequal capacities and penalty granularity. Not global optimum, stable identification, or circuit.')
    assert set(result['predictions'])==set(registered_predictions)
    result['registered_predictions']=registered_predictions
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
