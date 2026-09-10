"""V2 data runner: native Down_bias key repaired, same frozen objective/configs."""
import json,time,signal
from pathlib import Path
import torch
import torch.nn.functional as F
from structured_quadratic_models_v1 import QuadraticModel,QuadraticObjective
from convergent_quadratic_fit_v1 import advance
from joint_quadratic_fit_v1 import product_cross
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def run(config_name,chunk,runner,binding_path,dry=False):
    binding=json.loads(Path(binding_path).read_text());assert all(digest(p)==h for p,h in binding.items())
    configurations=json.loads((P/'STRUCTURED_QUADRATIC_CAMPAIGN_V1_CONFIGS.json').read_text());config=configurations[config_name]
    controls=json.loads((P/'STRUCTURED_QUADRATIC_MODELS_V1_CONTROL.json').read_text())['controls']
    assert all(c['dense_loss_absolute_error']<=1e-10 and c['gradient_absolute_error']<=1e-6 for c in controls.values())
    convergence_control=json.loads((P/'CONVERGENT_QUADRATIC_FIT_V1_CONTROL.json').read_text())
    assert convergence_control['unfinished_chunk_not_converged'] and convergence_control['checkpoint_resume_converged']
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    assert sd['transformer.h.17.mlp.Down_bias'].shape==(1152,)
    assert config['metric']=='data', 'V2 is only the data execution repair'
    integration=json.loads((P/'STRUCTURED_FIT_DATA_V2_INTEGRATION_CONTROL.json').read_text())
    assert integration['passed'] and integration['relative_replay_error']<=1e-3
    if dry:
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,config=config,chunk=chunk,max_fit_seconds=540,checkpoint_resume=True)));return
    out=P/f'STRUCTURED_FIT_V2_{config_name}_CHUNK_{chunk:02d}.json';assert not out.exists()
    checkpoint_path=P/f'STRUCTURED_FIT_V2_{config_name}_CHECKPOINT.pt';tic=time.perf_counter();signal.alarm(900)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(config['seed'])
    U=sd['lm_head.weight'].double().cuda();M=U.T@U;del U
    L,R,D=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    assert L.shape==R.shape==(4608,1152) and D.shape==(1152,4608) and M.shape==(1152,1152)
    model=QuadraticModel(config['kind'],1152,products=128,readers=64,groups=32,block_size=8,device='cuda')
    if config['metric']=='weight':
        gi=product_cross(L,R,L,R);go=D.T@M@D;total=(go*gi).sum()
        if config['initialization']=='native':
            indices=(gi.diag()*go.diag()).topk(128).indices
            assert config['kind']=='product'
            with torch.no_grad():model.a.copy_(F.normalize(L[indices],dim=1));model.b.copy_(F.normalize(R[indices],dim=1))
        del gi,go
        objective=QuadraticObjective(M,total,l=L,r=R,d=D)
    else:
        receipt=json.loads((P/'UNSUPERVISED_DATA_V2_RESULT.json').read_text());assert all(receipt['predictions'].values())
        ap=P/'UNSUPERVISED_DATA_V2_STATES.pt';assert digest(ap)==receipt['artifact_sha256']
        data=torch.load(ap,map_location='cpu',weights_only=True,mmap=True);rows=data['train_rows']
        x=(data['x'][rows].float()*data['x_scale'][rows]).reshape(-1,1152).double().cuda()
        y=(data['y'][rows].float()*data['y_scale'][rows]).reshape(-1,1152).double().cuda()-sd['transformer.h.17.mlp.Down_bias'].double().cuda()
        assert x.shape==y.shape==(51200,1152)
        total=((y@M)*y).sum();objective=QuadraticObjective(M,total,x=x,y=y)
        if config['initialization']=='native':
            power=((x@L.T)*(x@R.T)).square().mean(0)*(D.T@M@D).diag()
            indices=power.topk(128).indices
            with torch.no_grad():model.a.copy_(F.normalize(L[indices],dim=1));model.b.copy_(F.normalize(R[indices],dim=1))
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
    # Independent load confirms the actual checkpoint preserves its model/function.
    restored=torch.load(checkpoint_path,map_location='cuda',weights_only=False)
    model.load_state_dict(restored['model']);replay=float(objective.loss(model)[0].detach())
    replay_error=abs(replay-last['squared_relative_error']);valid=replay_error<=1e-10 and best['gram_condition']<=1e12
    result=dict(schema='structured.quadratic.fit.chunk.v2',config=config,chunk=chunk,
        predictions={'pred_a_instrument':bool(valid),'pred_b_converged':bool(valid and state['converged']),'pred_c_checkpoint_replay':bool(replay_error<=1e-10)},
        status='converged_local_fit' if state['converged'] else 'optimization_unfinished',best=best,last=last,
        checkpoint_sha256=digest(checkpoint_path),checkpoint_bytes=checkpoint_path.stat().st_size,replay_absolute_error=replay_error,
        history=state['history'],price=dict(body_forwards=0,input_parameter_numbers=model.storage_numbers(),output_writer_numbers=int(state['best']['writer'].numel()),optimizer_closures=state['closures'],chunk_seconds=state['chunk_seconds']),
        wall_seconds=time.perf_counter()-tic,runner_sha256=digest(runner),binding_sha256=digest(binding_path),
        scope='Unsupervised discovery fit, not a circuit. Nonconvergence is an unfinished optimizer, not evidence against structure. Test data unopened; no OOD or causal claims from this receipt.')
    atomic_create_json(out,result);print(json.dumps({k:v for k,v in result.items() if k!='history'}))
