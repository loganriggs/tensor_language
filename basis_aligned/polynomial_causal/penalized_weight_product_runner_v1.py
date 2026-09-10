"""Managed warm-start experiment for the explicitly penalized weight objective."""
import json,signal,time
from pathlib import Path
import torch
import torch.nn.functional as F
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from convergent_quadratic_fit_v2 import advance
from joint_quadratic_fit_v1 import product_cross
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def run(runner,binding_path,chunk=0,dry=False):
    binding=json.loads(Path(binding_path).read_text());assert all(digest(p)==h for p,h in binding.items())
    for name in ['ENERGY_REGULARIZED_QUADRATIC_V1_CONTROL.json','PENALIZED_CONVERGENCE_V1_CONTROL.json','OPTIMIZER_STALL_V2_CONTROL.json']:
        assert json.loads((P/name).read_text())['passed']
    config=dict(kind='product',metric='weight',penalty=.01,initialization='frozen_stalled_unit_gauge',seed=9116501)
    frozen=torch.load(P/'STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00_BEST.pt',map_location='cpu',weights_only=False)
    assert frozen['config']['kind']=='product' and frozen['config']['metric']=='weight'
    if dry:
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,config=config,chunk=chunk,max_fit_seconds=540,body_forwards=0)));return
    start=time.perf_counter();signal.alarm(900);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(config['seed'])
    out=P/f'PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_{chunk:02d}.json';assert not out.exists()
    checkpoint_path=P/'PENALIZED_WEIGHT_PRODUCT_V1_CHECKPOINT.pt'
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    model=QuadraticModel('product',1152,device='cuda')
    model.load_state_dict(frozen['best']['model'])
    with torch.no_grad():model.a.copy_(F.normalize(model.a,dim=1));model.b.copy_(F.normalize(model.b,dim=1))
    objective=EnergyRegularizedObjective(metric,total,l=l,r=r,d=d,penalty=config['penalty'])
    reference_run=json.loads((P/'STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00.json').read_text())
    fixed_path=json.loads((P/'STRUCTURED_FIT_FIXED_READER_PENALTY_V1_AUDIT.json').read_text())['records']
    reference_penalized=next(r['optimization_loss'] for r in fixed_path if r['penalty']==config['penalty'])
    initial=objective.diagnostics(model)[0]
    assert abs(initial['optimization_loss']-reference_penalized)<=1e-8
    previous=None
    if checkpoint_path.exists():
        previous=torch.load(checkpoint_path,map_location='cuda',weights_only=False)
        assert previous['next_chunk']==chunk and previous['config']==config and not previous['converged']
        model.load_state_dict(previous['model'])
    else:assert chunk==0
    state=advance(model,objective,previous,seconds=540)
    state.update(config=config,next_chunk=chunk+1)
    temporary=checkpoint_path.with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(checkpoint_path)
    best=state['best']['diagnostics'];last=state['history'][-1]
    restored=torch.load(checkpoint_path,map_location='cuda',weights_only=False)
    model.load_state_dict(restored['model']);replay=float(objective.loss(model)[0].detach())
    replay_error=abs(replay-last['optimization_loss'])
    valid=replay_error<=1e-10 and last['regularized_gram_condition']<=1e12
    reference_capture=frozen['best']['diagnostics']['captured_energy_fraction']
    useful=best['cancellation_ratio']<=10 and best['captured_energy_fraction']>=.75*reference_capture and reference_penalized-best['optimization_loss']>=1e-4
    result=dict(schema='penalized.weight.product.v1',config=config,chunk=chunk,best=best,last=last,
        predictions=dict(pred_a_instrument=bool(valid),pred_b_converged=bool(valid and state['converged']),pred_c_stable_useful=bool(valid and useful)),
        status='converged_local_fit' if state['converged'] else 'optimization_unfinished',terminal_reason=state['terminal_reason'],
        reference_unpenalized_capture=reference_capture,reference_penalized_objective=reference_penalized,initial=initial,checkpoint_sha256=digest(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,replay_absolute_error=replay_error,history=state['history'],
        price=dict(body_forwards=0,parameter_numbers=model.storage_numbers()+state['best']['writer'].numel(),
            closures=state['closures'],chunk_seconds=state['chunk_seconds'],warm_start_closures_to_best=frozen['best']['diagnostics']['closures'],
            warm_start_source_run_total_closures=reference_run['price']['optimizer_closures'],warm_start_source_run_seconds=reference_run['price']['chunk_seconds']),
        wall_seconds=time.perf_counter()-start,runner_sha256=digest(runner),binding_sha256=digest(binding_path),
        scope='Explicitly changed fitting bias and non-independent warm start. Stable/useful pass is numerical screening only, not circuit identification. Test data unopened.')
    atomic_create_json(out,result);print(json.dumps({k:v for k,v in result.items() if k!='history'}))
