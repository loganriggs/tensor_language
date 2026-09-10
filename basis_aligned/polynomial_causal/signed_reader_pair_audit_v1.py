"""Post-result natural coefficient scale, class effects and local gauge control."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from signed_reader_pair_v1 import writer,coefficient
from induction_context_transport_v2 import digest


def gauge_control():
    gen=torch.Generator().manual_seed(9111242);rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    def mlp(L,R,D,b):return SimpleNamespace(Left=SimpleNamespace(weight=L),Right=SimpleNamespace(weight=R),Down=SimpleNamespace(weight=D),Down_bias=b)
    L=rand(9,6);R=rand(9,6);D=rand(6,9);bias=rand(6);m=mlp(L,R,D,bias);S=torch.linalg.qr(rand(6,6)).Q
    transformed=mlp(L@S.T,R@S.T,S@D,S@bias);E=torch.linalg.qr(rand(6,2)).Q;u=rand(12,6);U=rand(17,6)
    v=writer(m,E[:,0],E[:,1]);v2=writer(transformed,S@E[:,0],S@E[:,1]);term=coefficient(u,E[:,0],E[:,1])[:,None]*v
    term2=coefficient(u@S.T,S@E[:,0],S@E[:,1])[:,None]*v2
    writer_error=float((v2-S@v).abs().max());term_error=float((term2-term@S.T).abs().max());reader_error=float((term2@(U@S.T).T-term@U.T).abs().max())
    assert max(writer_error,term_error,reader_error)<1e-10
    return dict(passed=True,writer_max_abs=writer_error,term_max_abs=term_error,reader_max_abs=reader_error,scope='Consistent orthogonal change of local MLP input/output and vocabulary-reader coordinates; not an untransformed native RoPE gauge or semantic identification')


def main():
    p=Path(__file__).parent;source=p/'SIGNED_READER_PAIR_V1_RESULT.json';r=json.loads(source.read_text());old=json.loads((p/'CALIBRATION_STABILITY_CONTEXT_V1_RESULT.json').read_text());rng=np.random.default_rng(9111242)
    component=torch.load(p/'SIGNED_READER_PAIR_V1_COMPONENT.pt',weights_only=True,map_location='cpu');w=torch.load(p/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt',weights_only=True,map_location='cpu')['w'];scale=float(w@component['writer']/(w@w));reports={}
    for cohort in ('FW_HOLDOUT','PILE_SHIFT'):
        rows=[x for x in r['per_row'] if x['split']==cohort];n=len(rows);draw=rng.integers(n,size=(4000,n));stats={}
        for cls in ('all','frequent','rare'):
            count=np.array([256 if cls=='all' else x[cls+'_count'] for x in rows]);effects={arm:np.array([x['ce_sums'][arm][cls]-x['ce_sums']['native'][cls] for x in rows]) for arm in ('full','calibration','complement')};effects['interaction']=effects['full']-effects['calibration']-effects['complement'];byclass={}
            for arm,delta in effects.items():
                estimate=float(delta.sum()/count.sum())
                if arm!='interaction':assert abs(estimate-r['reports'][cohort]['ce_effects'][arm][cls])<1e-12
                byclass[arm]=dict(estimate=estimate,ci95=np.quantile(delta[draw].sum(1)/count[draw].sum(1),[.025,.975]).tolist(),positive_rows=int((delta>0).sum()),rows=n)
            stats[cls]=byclass
        abmean=r['reports'][cohort]['ab_mean'];absd=r['reports'][cohort]['ab_sd'];qmean=old['reports'][cohort]['q_mean'];qsd=old['reports'][cohort]['q_sd']
        stats['natural_scale']=dict(ab_mean=abmean,ab_sd=absd,constructed_witness_abs_ab=288.,witness_ab_over_natural_sd=288/absd,pair_q_mean=scale*abmean,old_q_mean=qmean,pair_q_RMS_over_old_q_RMS=abs(scale)*(abmean**2+absd**2)**.5/(qmean**2+qsd**2)**.5,scope='RMS ratio is not additive variance share; large continuous witness coordinates need not be typical native activations')
        reports[cohort]=stats
    out=dict(experiment='signed_reader_pair_audit_v1',model_forwards=0,post_result=True,registered_gates_unchanged=True,seed=9111242,bootstrap_draws=4000,reports=reports,projection_scalar_coefficient=scale,gauge_control=gauge_control(),uncertainty_scope='Row bootstrap; FineWeb document grouping unknown, Pile one row per document',source_sha256=digest(__file__),input_result_sha256=digest(source))
    with (p/'SIGNED_READER_PAIR_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'gauge_control':out['gauge_control'],'reports':{c:{'all':v['all'],'natural_scale':v['natural_scale']} for c,v in reports.items()}},indent=2))


if __name__=='__main__':main()
