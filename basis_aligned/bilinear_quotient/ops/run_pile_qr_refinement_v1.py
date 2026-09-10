#!/usr/bin/env python3
# BQGATE: 0forwards0seq; two240second nonlinear fits on51200cachedstates.
"""pred_a equivalent initial loss/gradient and best replay; pred_b QR converges
and gains>=1% training; pred_c QR validates>=1% better than matched normal solve.
No penalty or new fitting bias. Both start from exactly the same frozen model.
"""
import os,sys,json,time,signal,shutil
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(P)]
import torch
from structured_quadratic_models_v1 import QuadraticModel,QuadraticObjective
from stable_empirical_quadratic_v1 import StableEmpiricalObjective
from convergent_quadratic_fit_v2 import advance
from prepare_million_token_panel_v1 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
BIND=P/'PILE_QR_REFINEMENT_V1_BINDING.json';OUT=P/'PILE_QR_REFINEMENT_V1_RESULT.json'
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'STABLE_EMPIRICAL_QUADRATIC_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,training_states=51200,validation_states=7168,arms=['normal','qr'],seconds_per_arm=240,adam_steps=0,test_access=False)));return
    assert not OUT.exists();assert shutil.disk_usage(P).free>100000000
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    with torch.no_grad():
        u=sd['lm_head.weight'].double().cuda();m=u.T@u;del u
        l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
        data=torch.load(P/'MILLION_TOKEN_PANEL_V1_INPUTS.pt',weights_only=True,map_location='cpu',mmap=True);panels={}
        for split,n in [('train',51200),('validation',7168)]:
            ids=data[split+'_rows'];x=(data['x'][ids].float()*data['x_scale'][ids]).reshape(-1,1152).double().cuda();assert len(x)==n
            y=torch.cat([((x[i:i+1024]@l.T)*(x[i:i+1024]@r.T))@d.T for i in range(0,n,1024)])
            panels[split]=(x,y)
        del l,r,d
    x,y=panels['train'];xt,yt=panels['validation'];stable=StableEmpiricalObjective(m,x,y);normal=QuadraticObjective(m,((y@m)*y).sum(),x=x,y=y)
    frozen=torch.load(P/'STRUCTURED_FIT_V2_data_shared_reader_s0_CHUNK_00_BEST.pt',map_location='cuda',weights_only=False)
    def fresh():
        model=QuadraticModel('shared_reader',1152,device='cuda');model.load_state_dict(frozen['best']['model']);return model
    model=fresh();loss0,_,_=normal.loss(model);loss0.backward();gradient0=torch.cat([p.grad.flatten() for p in model.parameters()]);model.zero_grad();loss1,_,_=stable.loss(model);loss1.backward();gradient1=torch.cat([p.grad.flatten() for p in model.parameters()])
    equivalence=dict(loss_absolute_error=abs(float((loss0-loss1).detach())),gradient_relative_l2=float((gradient0-gradient1).norm()/gradient0.norm().clamp_min(1e-30)))
    valid=equivalence['loss_absolute_error']<=1e-8 and equivalence['gradient_relative_l2']<=1e-6;assert valid
    initial=float(loss1.detach());results={}
    for name,objective in [('normal',normal),('qr',stable)]:
        model=fresh();started=time.perf_counter()
        try:
            state=advance(model,objective,checkpoint={'phase':'lbfgs'},seconds=240,adam_steps=0)
            state.update(config=dict(kind='shared_reader',metric='data_pile',solver=name,source='MILLION_TOKEN_PANEL_V1',seed=91162135),next_chunk=1)
            checkpoint=P/f'PILE_QR_REFINEMENT_V1_{name}_CHECKPOINT.pt';assert not checkpoint.exists();torch.save(state,checkpoint)
            model.load_state_dict(state['best']['model'])
            with torch.no_grad():
                a,b,c=model.components();w=state['best']['writer'].cuda();pred=(((xt@a.T)*(xt@b.T))@c)@w.T;delta=pred-yt
                validation=float(((delta@m)*delta).sum()/((yt@m)*yt).sum())
                replay=float(stable.loss(model)[0]);replay_error=abs(replay-state['best']['diagnostics']['squared_relative_error'])
            valid=valid and replay_error<=1e-8
            result=dict(status='converged_local_fit' if state['converged'] else 'optimization_unfinished',converged=state['converged'],terminal_reason=state['terminal_reason'],best=state['best']['diagnostics'],last=state['history'][-1],validation_squared_relative_error=validation,replay_absolute_error=replay_error,closures=state['closures'],fit_seconds=state['chunk_seconds'],checkpoint_sha256=digest(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,wall_seconds=time.perf_counter()-started)
        except (ArithmeticError,RuntimeError) as error:
            valid=False;result=dict(status='optimizer_or_instrument_failure',error=repr(error),wall_seconds=time.perf_counter()-started)
        results[name]=result;atomic_create_json(P/f'PILE_QR_REFINEMENT_V1_{name}_RESULT.json',result);print(json.dumps(dict(arm=name,**result)),flush=True)
    qr,ng=results['qr'],results['normal'];both=all('best' in a for a in results.values())
    result=dict(schema='pile.qr.refinement.v1',predictions={'pred_a_instrument':bool(valid and both),'pred_b_qr_converged_improved':bool(valid and both and qr['converged'] and qr['best']['squared_relative_error']<=.99*initial),'pred_c_qr_validation_advantage':bool(valid and both and qr['validation_squared_relative_error']<=.99*ng['validation_squared_relative_error'])},initial_training_squared_error=initial,equivalence=equivalence,results=results,wall_seconds=time.perf_counter()-tic,binding_sha256=digest(BIND),runner_sha256=digest(RUNNER),scope='Matched240s fresh-LBFGS refinements, same frozen parameter gauge and Pile training states. No new bias, truncation, penalty or test access. Fixed arm order and single dependent starts limit ranking. Nonconvergence is unfinished optimization, not absent structure.')
    atomic_create_json(OUT,result);print(json.dumps(result))
if __name__=='__main__':main()
