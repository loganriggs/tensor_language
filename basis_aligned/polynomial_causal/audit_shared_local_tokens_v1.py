"""Read-only exact coefficient-metric audit, including all vocabulary rows.

Without --program-g: optimal global baselines. With --program-g: score the frozen
compiled artifact and compare its individual token errors to that same baseline.
No semantic labels, model forwards, fitting or post-hoc group changes.
"""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross

P=Path(__file__).resolve().parent


def summary(error_squared,target_squared):
    positive=target_squared>0
    relative=(error_squared[positive]/target_squared[positive]).clamp_min(0).sqrt()
    return dict(full_coefficient_relative_error=float((error_squared.sum()/target_squared.sum()).sqrt()),
                median_token_relative_error=float(relative.median()),
                token_relative_error_quantiles={str(q):float(torch.quantile(relative,q)) for q in [.1,.5,.9,.95,.99]},
                fraction_tokens_error_below={str(q):float((relative<=q).double().mean()) for q in [.1,.25,.5,1.]},
                zero_target_rows=int((~positive).sum()),
                maximum_error_on_zero_target=float(error_squared[~positive].max()) if bool((~positive).any()) else 0.)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--program-g',type=int,choices=[64,128]);args=parser.parse_args()
    torch.set_num_threads(2);started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double()
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    metric=down@product_cross(l,r,l,r)@down.T
    root=torch.linalg.cholesky((metric+metric.T)/2)
    mean=u.mean(0);mean_function=mean@root
    x=(u-mean)@root;target_norm=(x+mean_function).square().sum(1)
    covariance=x.T@x;eigen,vectors=torch.linalg.eigh((covariance+covariance.T)/2)
    eigen=eigen.flip(0);vectors=vectors.flip(1)
    reference=json.loads((P/'FULLU_SHARED_LOCAL_FEASIBILITY_V1_RESULT.json').read_text())
    assert abs(float(target_norm.sum())/reference['coefficient_energy']-1)<1e-10
    baseline_errors={}
    baselines=[]
    for rank in [78,167]:
        residual=x-(x@vectors[:,:rank])@vectors[:,:rank].T
        e=residual.square().sum(1);baseline_errors[rank]=e
        spectral_error=abs(float(e.sum()/eigen[rank:].sum())-1)
        assert spectral_error<1e-10
        baselines.append(dict(rank=rank,**summary(e,target_norm),spectral_replay_error=spectral_error))
    result=dict(schema='shared.local.token.audit.v1',global_baselines=baselines,
                scope='Per-token quadratic coefficient errors; no language-model fidelity or semantic cluster claims.')
    if args.program_g is not None:
        g=args.program_g
        path=P/f'FULLU_SHARED_LOCAL_FIT_V1_G{g}_PROGRAM.pt'
        fit=json.loads((P/f'FULLU_SHARED_LOCAL_FIT_V1_G{g}_RESULT.json').read_text())
        program=torch.load(path,map_location='cpu',weights_only=True)
        global_part=program['global_codes'].double()@(program['global_reader'].double()@root)
        private_part=torch.zeros_like(global_part)
        labels=program['labels'].long();groups=[]
        for k,bank in enumerate(program['local_readers']):
            ix=labels==k
            private_part[ix]=program['local_codes'][ix].double()@(bank.double()@root)
        prediction=global_part+private_part+program['mean'].double()@root
        e=(prediction-x-mean_function).square().sum(1)
        baseline=baseline_errors[fit['matched_global_rank']]
        for k in range(len(program['local_readers'])):
            ix=labels==k
            groups.append(dict(group=k,tokens=int(ix.sum()),**summary(e[ix],target_norm[ix])) if bool(ix.any()) else dict(group=k,tokens=0))
        aggregate=summary(e,target_norm)
        replay=abs(aggregate['full_coefficient_relative_error']-fit['full_coefficient_relative_error'])
        assert replay<=fit['compiled_fp32_error']+1e-9
        centered_prediction=global_part+private_part
        result['program']=dict(global_width=g,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **aggregate,error_replay_absolute_difference=replay,groups=groups,
            fraction_tokens_improved_vs_matched_global=float((e<baseline).double().mean()),
            fraction_tokens_worsened_vs_matched_global=float((e>baseline).double().mean()),
            worst_squared_error_increase=float((e-baseline).max()),
            component_energy_over_combined=float((global_part.square().sum()+private_part.square().sum())/centered_prediction.square().sum()),
            exact_nonzero_global_codes=int((program['global_codes']!=0).sum()),
            exact_nonzero_private_codes=int((program['local_codes']!=0).sum()))
    result['wall_seconds']=time.perf_counter()-started
    stem='SHARED_LOCAL_GLOBAL_TOKEN_BASELINES_V1' if args.program_g is None else f'SHARED_LOCAL_TOKEN_AUDIT_V1_G{args.program_g}'
    with (P/(stem+'_RESULT.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='program'},indent=2))
    if 'program' in result:print(json.dumps({k:v for k,v in result['program'].items() if k!='groups'},indent=2))


if __name__=='__main__':main()
