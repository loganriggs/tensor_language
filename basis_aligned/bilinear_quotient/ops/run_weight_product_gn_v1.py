#!/usr/bin/env python3
# BQGATE: 0forwards0seq; weight-only joint GN120s, no corpus access.
import os,sys,json,time,signal,hashlib,shutil
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(P)]
import torch
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from fit_symmetric_product_gauss_newton_v1 import fit
from joint_quadratic_fit_v1 import product_cross
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    binding=json.loads((P/'WEIGHT_PRODUCT_GN_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    for n in ['SYMMETRIC_PRODUCT_GAUSS_NEWTON_V1_CONTROL.json','SYMMETRIC_PRODUCT_LM_V1_CONTROL.json']:assert json.loads((P/n).read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,seconds=120,products=128,penalty=.01)));return
    out=P/'WEIGHT_PRODUCT_GN_V1_RESULT.json';assert not out.exists();assert shutil.disk_usage(P).free>20_000_000
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum();c=torch.linalg.cholesky(metric).T
    frozen=torch.load(P/'STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00_BEST.pt',weights_only=False,map_location='cuda')
    model=QuadraticModel('product',1152,device='cuda');model.load_state_dict(frozen['best']['model'])
    objective=EnergyRegularizedObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    with torch.no_grad():
        _,w,_=objective.loss(model);a,b,_=model.components();target=(l,r,c@d/total.sqrt())
        pars,record=fit((a,b,c@w/total.sqrt()),target,total.new_tensor(1.),seconds=120)
        model.a.copy_(pars[0]);model.b.copy_(pars[1]);physical_writer=torch.linalg.solve_triangular(c,pars[2]*total.sqrt(),upper=True)
    final,w=objective.diagnostics(model)
    replay=abs(final['optimization_loss']-record['final'])
    writer_bridge=float((w-physical_writer).norm()/w.norm())
    reference=json.loads((P/'PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_00.json').read_text())
    initial_error=abs(record['initial']-reference['initial']['optimization_loss'])
    values=[record['initial']]+[row['objective'] for row in record['history']]
    increase=max([0.]+[b-a for a,b in zip(values,values[1:])])
    valid=writer_bridge<=1e-10 and replay<=1e-10 and initial_error<=1e-8 and increase<=1e-10
    plateau=abs(values[-1]-values[-5])/final['captured_energy_fraction'] if len(values)>=5 else float('inf')
    converged=plateau<=1e-5 and final['relative_stationarity']<=1e-4 and final['gradient_max_abs']<=1e-7
    ap=P/'WEIGHT_PRODUCT_GN_V1_CHECKPOINT.pt';assert not ap.exists()
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},best=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},writer=w.cpu(),diagnostics=final),record=record),ap)
    result=dict(schema='weight.product.gn.v1',predictions={'pred_a_instrument':valid,'pred_b_converged':valid and converged,'pred_c_reference_quality':valid and final['optimization_loss']<=.9141457259720446 and final['cancellation_ratio']<=10},final=final,initial=record['initial'],initial_bridge_error=initial_error,final_bridge_error=replay,writer_relative_bridge_error=writer_bridge,maximum_accepted_increase=increase,relative_progress_last_five=plateau,terminal_reason=record['terminal_reason'],price={k:record[k] for k in ['fit_seconds','accepted','rejected','inner_iterations','inexact_inner_solves']},checkpoint_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,scope='Weight-only joint GN with exact conditional writer projection. No global optimum or circuit claim.')
    result['price'].update(body_forwards=0,parameter_numbers=442368,warm_start_source_seconds=540.0881669595838,checkpoint_bytes=ap.stat().st_size)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
