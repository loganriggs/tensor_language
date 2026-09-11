#!/usr/bin/env python3
"""A source/target fidelity; B same-objective convergence; C unchanged concentration/capture screen."""
# BQGATE: 0forwards0seq; resume solver work on fixed native products and objectives.
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(P));sys.path.insert(0,str(RUNNER.parent))
import torch
from run_product_writer_redteam_v1 import digest,metrics,CK
from joint_quadratic_fit_v1 import product_cross
from legal_token_writer_admm_v2 import solve


def main():
    binding=json.loads((P/'PRODUCT_WRITER_REPAIR_V2_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'LEGAL_TOKEN_WRITER_ADMM_V2_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
            repair_only_unfinished_arms=True,maximum_seconds_per_arm=540,unchanged_objective=True)))
        return
    out=P/'PRODUCT_WRITER_REPAIR_V2_RESULT.json';assert not out.exists();signal.alarm(1500)
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    source=json.loads((P/'PRODUCT_WRITER_REDTEAM_V1_RESULT.json').read_text())
    cache=Path(source['cache']['path']);assert digest(cache)==source['cache']['sha256']
    saved=torch.load(cache,weights_only=True,map_location='cpu')
    root=saved['root'].cuda();pl=saved['left'].cuda();pr=saved['right'].cuda();scale=saved['scale']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda()
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    root_error=float((root@root.T-u.T@u).norm()/(u.T@u).norm())
    p=torch.linalg.solve_triangular(root,u.T,upper=False).T;f=p-p.mean(0)
    g=product_cross(pl,pr,pl,pr);c=root.T@down@product_cross(l,r,pl,pr)/scale
    total=99245061353.473/scale**2;arms=[];states={};final_writers={}
    for old in source['legal_arms']:
        key=str(old['penalty_multiplier']);z=saved['whitened_writers'][key].cuda()
        initial=metrics(p,z,g,c,total)
        metric_bridge=max(abs(initial[k]-old[k]) for k in ['full_coefficient_capture','median_top16_contrast_loading_energy'])
        if old['solver']['converged']:
            row={**old,'repair_skipped_already_converged':True,'initial_metric_bridge_error':metric_bridge}
        else:
            # V1 did not retain its multiplier. This is a documented primal warm start,
            # followed by full-state saving; it is not an exact V1 optimizer resume.
            state=dict(z=z,a=f@z,dual=torch.zeros((len(p),len(pl)),dtype=p.dtype,device=p.device),rho=1.,step=0,penalty=old['penalty'])
            z,state,report=solve(p,g,c,old['penalty'],state=state,max_steps=20000,tolerance=1e-5,seconds=540)
            row=dict(penalty_multiplier=old['penalty_multiplier'],penalty=old['penalty'],
                     **metrics(p,z,g,c,total),solver=report,initial_metric_bridge_error=metric_bridge,
                     initial_objective=old['solver']['final']['objective'])
            states[key]={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in state.items()}
        arms.append(row);final_writers[key]=z.cpu()
        print(json.dumps({k:v for k,v in row.items() if k!='solver'}|{'solver_final':row['solver']['final'],'converged':row['solver']['converged']}),flush=True)
    target=Path('/dev/shm/bilin18_product_writer_repair_v2.pt');assert not target.exists()
    torch.save(dict(states=states,whitened_writers=final_writers,root=root.cpu(),left=pl.cpu(),right=pr.cpu(),scale=scale,
        source_cache_sha256=source['cache']['sha256']),target)
    valid=source['predictions']['pred_a_instrument'] and max([root_error]+[a['initial_metric_bridge_error'] for a in arms])<=1e-8
    baseline=source['full_legal_unpenalized_baseline'];candidate=arms[1]
    result=dict(predictions={'pred_a_fidelity':valid,'pred_b_convergence':valid and all(a['solver']['converged'] for a in arms),
        'pred_c_concentration_at_retained_capture':valid and candidate['solver']['converged']
            and candidate['median_top16_contrast_loading_energy']>=1.25*baseline['median_top16_contrast_loading_energy']
            and candidate['full_coefficient_capture']>=.8*baseline['full_coefficient_capture']},
        full_legal_unpenalized_baseline=baseline,legal_arms=arms,root_weight_bridge_error=root_error,
        source_result_sha256=digest(P/'PRODUCT_WRITER_REDTEAM_V1_RESULT.json'),
        cache=dict(path=str(target),sha256=digest(target),bytes=target.stat().st_size,ephemeral=True),
        wall_seconds=time.perf_counter()-start,
        scope='Same convex fixed-product objectives, primal warm start from V1 with reset multiplier; adaptive rho and full-state checkpoint. Not joint input convergence or physical adoption.')
    with out.open('x') as fh:json.dump(result,fh,indent=2);fh.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='legal_arms'}),flush=True)


if __name__=='__main__':main()
