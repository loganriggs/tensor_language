#!/usr/bin/env python3
# BQGATE: 0forwards0seq; 120s weight-only symmetric-product ALS, no data.
"""pred_a implicit-solve/monotonic/replay instrument; pred_b gradient+plateau
convergence; pred_c historical objective+1e-4 and cancellation<=10.
"""
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(P)]
import torch
import torch.nn.functional as F
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from symmetric_product_als_v1 import normal_operator,native_rhs,block_precondition,pcg
from joint_quadratic_fit_v1 import product_cross
from prepare_million_token_panel_v1 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
BIND=P/'WEIGHT_PRODUCT_ALS_V1_BINDING.json';OUT=P/'WEIGHT_PRODUCT_ALS_V1_RESULT.json'
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    for name in ['SYMMETRIC_PRODUCT_ALS_V1_CONTROL.json','PENALIZED_PRODUCT_ALS_V1_CONTROL.json','ENERGY_REGULARIZED_QUADRATIC_V1_CONTROL.json']:assert json.loads((P/name).read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,metric='weight',products=128,penalty=.01,fit_seconds=120,corpus_access=False)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    frozen=torch.load(P/'STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00_BEST.pt',weights_only=False,map_location='cuda')
    model=QuadraticModel('product',1152,device='cuda');model.load_state_dict(frozen['best']['model'])
    with torch.no_grad():model.a.copy_(F.normalize(model.a,dim=1));model.b.copy_(F.normalize(model.b,dim=1))
    objective=EnergyRegularizedObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    reference=json.loads((P/'PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_00.json').read_text())
    history=[];best=None;outer=0;inner=0;maximum_inner_residual=0.;maximum_increase=0.;converged=False
    def diagnose():
        nonlocal best,converged
        diag,w=objective.diagnostics(model);diag.update(outer_sweeps=outer,inner_cg_iterations=inner)
        history.append(diag)
        if best is None or diag['optimization_loss']<best['diagnostics']['optimization_loss']:
            best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},writer=w.cpu(),diagnostics=diag.copy())
        if len(history)>=5:
            change=abs(history[-5]['optimization_loss']-diag['optimization_loss'])/diag['captured_energy_fraction'];diag['relative_progress_five_checks']=change
            converged=change<=1e-5 and diag['relative_stationarity']<=1e-4 and diag['gradient_max_abs']<=1e-7
        print(json.dumps(diag),flush=True)
    diagnose();initial=history[0]['optimization_loss'];assert abs(initial-reference['initial']['optimization_loss'])<=1e-8
    started=time.perf_counter();reason='time_limit'
    while time.perf_counter()-started<120 and not converged:
        for operand in ['a','b']:
            with torch.no_grad():
                oldloss,w,_=objective.loss(model);a,b,_=model.components();variable,other=(a,b) if operand=='a' else (b,a)
                og=w.T@metric@w;regularized=og+.01*torch.diag(og.diag());rhs=native_rhs(l,r,d,other,w,metric)
                solution,diag=pcg(lambda z:normal_operator(z,other,regularized),rhs,lambda z:block_precondition(z,other,regularized),initial=variable,tolerance=1e-9,max_iterations=300)
                inner+=diag['iterations'];maximum_inner_residual=max(maximum_inner_residual,diag['true_relative_residual'])
                getattr(model,operand).copy_(F.normalize(solution,dim=1))
                newloss=objective.loss(model)[0];maximum_increase=max(maximum_increase,float(newloss-oldloss))
                if not diag['converged']:reason='inner_solve_unfinished';break
        outer+=1
        if reason=='inner_solve_unfinished':break
        if outer%5==0:diagnose()
    diagnose();fit_seconds=time.perf_counter()-started
    if converged:reason='plateau_and_stationarity'
    state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},best=best,history=history,converged=converged,next_chunk=1,config=dict(kind='product',metric='weight',penalty=.01,optimizer='implicit_als'),outer_sweeps=outer,inner_cg_iterations=inner)
    ap=P/'WEIGHT_PRODUCT_ALS_V1_CHECKPOINT.pt';assert not ap.exists();torch.save(state,ap)
    with torch.no_grad():replay=float(objective.loss(model)[0])
    replay_error=abs(replay-history[-1]['optimization_loss']);valid=replay_error<=1e-10 and maximum_increase<=1e-9 and maximum_inner_residual<=1e-9
    diag=best['diagnostics'];result=dict(schema='weight.product.als.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_converged':bool(valid and converged),'pred_c_reference_quality':bool(valid and diag['optimization_loss']<=reference['best']['optimization_loss']+1e-4 and diag['cancellation_ratio']<=10)},status='converged_local_fit' if converged else 'optimization_unfinished',terminal_reason=reason,initial_optimization_loss=initial,best=diag,last=history[-1],maximum_inner_true_relative_residual=maximum_inner_residual,maximum_half_step_objective_increase=maximum_increase,replay_absolute_error=replay_error,checkpoint_sha256=digest(ap),price=dict(body_forwards=0,parameter_numbers=442368,outer_sweeps=outer,inner_cg_iterations=inner,fit_seconds=fit_seconds,checkpoint_bytes=ap.stat().st_size,warm_start_source_seconds=540.0881669595838),wall_seconds=time.perf_counter()-tic,binding_sha256=digest(BIND),runner_sha256=digest(RUNNER),scope='Weight-only exact conditional ALS, explicitlambda.01 objective, dependent frozen initialization. Not data fitting, global optimum or circuit identification. Historical L-BFGS comparison has different optimizer iteration costs.')
    atomic_create_json(OUT,result);print(json.dumps(result))
if __name__=='__main__':main()
