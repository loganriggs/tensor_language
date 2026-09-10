#!/usr/bin/env python3
# BQGATE: 0forwards0seq; weight-only square stopping repair120s.
import os,sys,json,time,signal,hashlib,shutil
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from square_gauge_diagnostics_v1 import stationarity
from joint_quadratic_fit_v1 import product_cross
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    bind=json.loads((P/'WEIGHT_SQUARE_POLISH_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in bind.items())
    assert json.loads((P/'SQUARE_GAUGE_DIAGNOSTICS_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,seconds=120,parameters=589824)));return
    out=P/'WEIGHT_SQUARE_POLISH_V1_RESULT.json';assert not out.exists();assert shutil.disk_usage(P).free>15_000_000
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']];total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    frozen=torch.load(P/'WEIGHT_STRUCTURAL_BASELINE_V2_square_CHUNK_00_BEST.pt',weights_only=False,map_location='cuda');model=QuadraticModel('square',1152,device='cuda');model.load_state_dict(frozen['best']['model'])
    original_norms=model.a.detach().norm(dim=1,keepdim=True)
    with torch.no_grad():model.a.div_(original_norms)
    obj=EnergyRegularizedObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=20,history_size=8,tolerance_grad=1e-14,tolerance_change=1e-18,line_search_fn='strong_wolfe')
    history=[];closures=0;updates=0;zero=0;converged=False;reason='budget_limit'
    def diagnose():
        nonlocal converged
        diag,w=obj.diagnostics(model);diag.update(stationarity(model,diag['captured_energy_fraction'],original_norms));diag.update(closures=closures,updates=updates)
        assert torch.isfinite(torch.tensor(list(diag.values()))).all() and diag['regularized_gram_condition']<=1e12
        history.append(diag)
        if len(history)>=5:
            change=abs(history[-5]['optimization_loss']-diag['optimization_loss'])/diag['captured_energy_fraction'];diag['relative_progress_five_checks']=change
            converged=change<=1e-5 and max(diag['canonical_relative_stationarity'],diag['original_gauge_relative_stationarity'])<=1e-4 and max(diag['canonical_gradient_max'],diag['original_gauge_gradient_max'])<=1e-7
        print(json.dumps(diag),flush=True);return diag,w
    first,_=diagnose();old=frozen['best']['diagnostics'];initial_bridge=abs(first['optimization_loss']-old['optimization_loss']);station_bridge=abs(first['original_gauge_relative_stationarity']/old['relative_stationarity']-1)
    assert initial_bridge<=1e-10 and station_bridge<=1e-6
    start=time.perf_counter()
    while time.perf_counter()-start<120 and not converged:
        previous=model.a.detach().clone()
        def closure():
            nonlocal closures
            opt.zero_grad(set_to_none=True);loss,_,_=obj.loss(model);loss.backward();closures+=1;return loss
        opt.step(closure);updates+=1
        zero=zero+1 if torch.equal(model.a.detach(),previous) else 0
        if updates%5==0:diagnose()
        if zero>=25:reason='line_search_stalled';break
    final,w=diagnose();fit_seconds=time.perf_counter()-start
    if converged:reason='plateau_and_both_gauges_stationary'
    replay=abs(float(obj.loss(model)[0].detach())-final['optimization_loss'])
    valid=initial_bridge<=1e-10 and station_bridge<=1e-6 and replay<=1e-10
    ap=P/'WEIGHT_SQUARE_POLISH_V1_FINAL.pt';assert not ap.exists();torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},writer=w.cpu(),diagnostics=final,history=history,original_norms=original_norms.cpu(),optimizer_state_saved=False,config=frozen['config']),ap)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_converged':valid and converged,'pred_c_improved_and_retained':valid and first['optimization_loss']-final['optimization_loss']>=1e-10 and final['captured_energy_fraction']>=old['captured_energy_fraction']-1e-8},initial=first,final=final,terminal_reason=reason,initial_bridge=initial_bridge,initial_stationarity_bridge=station_bridge,final_replay_error=replay,price=dict(body_forwards=0,parameter_numbers=589824,closures=closures,lbfgs_actual_inner_iterations=int(opt.state[model.a].get('n_iter',0)),fit_seconds=fit_seconds,checkpoint_bytes=ap.stat().st_size),checkpoint_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,scope='Sameobjective stoppingrepair. Bothcoordinate gauges must converge; no globaloptimum/circuitclaim. Compactfunction saved, optimizerhistorynotretained.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
