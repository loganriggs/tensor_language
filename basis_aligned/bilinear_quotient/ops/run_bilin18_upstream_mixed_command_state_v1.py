"""Fixed block8 mixed-command state cut and downstream read/output mediation.

A: parent read stats abs1e-4/relative1e-5, exact algebra abs1e-9/relative1e-10,
FP32 preserved modes abs1e-4/relative1e-5, removed mode1e-5, causal order,
controls/authorities/counts/restoration. B: mixed read retention<=.1 per phase.
C: output mixed retention<=.1 and mean/single mode errors<=.01 per phase.
64 forwards/256 sequences/384 extra local contractions; no fitting or updates.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_upstream_read_mediation pred_c_selective_output_mediation
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import run_bilin18_routed_shared_command_interaction_v1 as N
import mixed_command_state_intervention as S
from circuit_fast_screen_managed_runner import atomic_create_json

P=N.P;C=N.D.C;R=N.R;M=N.M;POLY=P.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_UPSTREAM_MIXED_COMMAND_STATE_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_UPSTREAM_MIXED_COMMAND_STATE_V1_RESULT.json'
FILES={'prior':PRIOR,'parent':N.OUT,'parent_runner':Path(N.__file__),'primitive':Path(S.__file__)}
EXPECTED={'prior': 'c8125da05fd21ac74df7ba12f826364f426556752d1c7563eb28191d392088a0', 'parent': 'b10ef0d68c7e635bcb00bbd6d62ff195dfdbe21ef0700ac26ca7ffb5f83d5581', 'parent_runner': 'c37eb7776bc41d7c3d08d2acb4bed8dbedf02c208e6e324f6f9aaf4767ca963a', 'primitive': 'f24768ca40941e198c69e57bbb9b8dd9e1a48e38e8f02ba7c1b077b12e219cf7'}


def agreement(x,y,atol=1e-4,rtol=1e-5):
    delta=x.double()-y.double();absolute=float(delta.abs().max())
    relative=float(delta.square().mean().sqrt())/max(float(y.double().square().mean().sqrt()),1e-6)
    finite=bool(x.isfinite().all() and y.isfinite().all())
    return {'max_abs':absolute,'relative_rms':relative,'finite':finite,'passed':finite and absolute<=atol and relative<=rtol}


def read_records(reads,endpoints):
    records=[];future=0.
    for layer,hs in reads.items():
        for head,r in hs.items():
            u=r['value'].double()@r['weight'].double().T
            payload={ab:u[i] for i,(_,ab,_) in enumerate(endpoints)}
            for role in P.ROLES:
                pos=endpoints[0][2][role+'_position']-1
                patterns={ab:r['pattern'][i,pos:pos+1].double() for i,(_,ab,_) in enumerate(endpoints)}
                d=N.D.decompose(patterns,payload)
                if role=='temporal':future=max(future,float(d['mixed'].abs().max()))
                records.append({'layer':layer,'head':head,'role':role,'entries':d['mixed'].numel(),
                    'native_mean_square':sum(float(x.square().sum()) for x in d['native'].values())/4,
                    'mixed_square':float(d['mixed'].square().sum()),'crossed_error_square':float((d['crossed']-d['mixed']).square().sum()),
                    **{name+'_square':float(x.square().sum()) for name,x in d['terms'].items()},'crossed_square':float(d['crossed'].square().sum())})
    return records,future


def main():
    observed={k:P.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:P.sha(p) for k,p in N.FILES.items()}==N.EXPECTED
    assert {k:P.sha(p) for k,p in P.FILES.items()}==P.EXPECTED
    previous=json.loads(N.OUT.read_text());assert previous['predictions']['pred_a_instrument'] and previous['predictions']['pred_b_material_read_interaction']
    rows=P.parent.candidate.build_rows();assert len(rows)==32 and rows==P.parent.candidate.build_rows()
    for row in rows:
        es=list(row['endpoints'].values());assert len(es)==4 and len({len(e['ids']) for e in es})==1
        assert all(len({e[role+'_position'] for e in es})==1 for role in P.ROLES)
    dry={'dryrun':True,'model_loaded':False,'gpu_accessed':False,'worlds':32,'model_forwards':64,'sequence_evaluations':256,'extra_local_contractions':384,'authority_sha256':observed}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);started=time.perf_counter()
    backend=P.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    assert {k:getattr(backend.model.config,k) for k in P.parent.EXPECTED_CONFIG}==P.parent.EXPECTED_CONFIG
    controls=S.controls();assert controls['passed'];records=[];deployed=[];exact=[];causal=[];removed=0.;future=0.;forwards=0;layers=0;finite=True;restored=True
    with torch.inference_mode():
        for row in rows:
            endpoints,_=P.parent.endpoint_bank([row]);cells=[ab for _,ab,_ in endpoints];tokens=torch.tensor([e['ids'] for _,_,e in endpoints],device='cuda')
            outputs={};reads={};state=[]
            for arm in ('native','edited'):
                with R.capture(backend.model,P.ARMS['AB']) as raw:
                    if arm=='native':outputs[arm],_=P.parent._forward(backend,tokens)
                    else:
                        with S.at_boundary(backend.model,8,cells,state):outputs[arm],_=P.parent._forward(backend,tokens)
                forwards+=1;layers+=len(raw);reads[arm],f=read_records(raw,endpoints);future=max(future,f)
                finite=finite and bool(outputs[arm].isfinite().all())
            restored=restored and not backend.model.transformer.h[8]._forward_hooks and all('squared_attention' not in backend.model.transformer.h[l].attn.__dict__ for l in P.ARMS['AB'])
            assert len(state)==1
            before,after=state[0];bm=C.modes({ab:before[i].double() for i,ab in enumerate(cells)});am=C.modes({ab:after[i].double() for i,ab in enumerate(cells)})
            ideal=S.remove_mixed(before.double(),cells);im=C.modes({ab:ideal[i] for i,ab in enumerate(cells)})
            for uv in ('00','10','01'):
                deployed.append(agreement(am[uv],bm[uv]));exact.append(M.correspondence(im[uv]+bm['00'],bm[uv]+bm['00']))
            exact.append(M.correspondence(im['11']+bm['00'],bm['00']));removed=max(removed,float(am['11'].abs().max()))
            temporal=endpoints[0][2]['temporal_position']-1
            causal.append(agreement(outputs['edited'][:,temporal],outputs['native'][:,temporal]))
            modes={arm:C.modes({ab:P.center(logits[i].double()) for i,ab in enumerate(cells)}) for arm,logits in outputs.items()}
            panels={}
            for role in ('all',)+P.ROLES:
                pos=None if role=='all' else endpoints[0][2][role+'_position']-1
                panel={}
                for uv in C.CELLS:
                    n=modes['native'][uv] if pos is None else modes['native'][uv][pos]
                    e=modes['edited'][uv] if pos is None else modes['edited'][uv][pos]
                    panel[uv]={'entries':n.numel(),'native_square':float(n.square().sum()),'edited_square':float(e.square().sum()),'error_square':float((e-n).square().sum())}
                panels[role]=panel
            lp=outputs['native'].double().log_softmax(-1);kl=(lp.exp()*(lp-outputs['edited'].double().log_softmax(-1))).sum(-1).flatten().cpu().tolist()
            records.append({'row_id':row['row_id'],'phase':row['phase'],'template_id':row['template_id'],'reads':reads,'outputs':panels,'kl':kl})
    summaries={};replays=[]
    for phase in ('FIT','HOLDOUT'):
        pr=[r for r in records if r['phase']==phase]
        for template in ['ALL']+sorted({r['template_id'] for r in pr}):
            rs=[r for r in pr if template=='ALL' or r['template_id']==template];read_summary={}
            for role in P.ROLES:
                arms={arm:N.summarize([entry for r in rs for entry in r['reads'][arm] if entry['role']==role]) for arm in ('native','edited')}
                read_summary[role]={**arms,'mixed_retention':arms['edited']['mixed_rms']/max(arms['native']['mixed_rms'],1e-6)}
                for field in ('native_shared_rms','mixed_rms','crossed_relative_error'):
                    replays.append(agreement(torch.tensor(arms['native'][field],dtype=torch.float64),torch.tensor(previous['summaries'][phase+'/'+template][role][field],dtype=torch.float64)))
            out_summary={}
            for role in ('all',)+P.ROLES:
                out_summary[role]={}
                for uv in C.CELLS:
                    vs=[r['outputs'][role][uv] for r in rs];count=sum(v['entries'] for v in vs);rms=lambda k:(sum(v[k] for v in vs)/count)**.5
                    n=rms('native_square');e=rms('edited_square');error=rms('error_square')
                    out_summary[role][uv]={'native_rms':n,'edited_rms':e,'retention':e/max(n,1e-6),'relative_change':error/max(n,1e-6),'denominator_floored':n<1e-6}
            kl=[x for r in rs for x in r['kl']]
            summaries[phase+'/'+template]={'reads':read_summary,'outputs':out_summary,'kl':{'mean':float(np.mean(kl)),'p99':float(np.quantile(kl,.99)),'max':float(np.max(kl))}}
        summaries[phase+'/heads']={f'L{l}H{h}/{role}':{arm:N.summarize([e for r in pr for e in r['reads'][arm] if e['layer']==l and e['head']==h and e['role']==role]) for arm in ('native','edited')}
            for l,hs in P.ARMS['AB'].items() for h in hs for role in P.ROLES}
    a=finite and restored and forwards==64 and layers==192 and len(records)==32 and removed<=1e-5 and future<=1e-8 and all(x['passed'] for x in deployed+exact+causal+replays)
    b=all(summaries[p+'/ALL']['reads']['iswas']['mixed_retention']<=.1 for p in ('FIT','HOLDOUT'))
    c=all(summaries[p+'/ALL']['outputs']['iswas']['11']['retention']<=.1 and all(summaries[p+'/ALL']['outputs']['iswas'][uv]['relative_change']<=.01 for uv in ('00','10','01')) for p in ('FIT','HOLDOUT'))
    maxima=lambda xs:{key:max(x[key] for x in xs) for key in ('max_abs','relative_rms')}
    result={'scope':'Opened cross-cell residual intervention, not natural token edit or structurally reduced variable.',
        'predictions':{'pred_a_instrument':a,'pred_b_upstream_read_mediation':b,'pred_c_selective_output_mediation':c},
        'terminal':'invalid' if not a else 'upstream_joint_variable_screen_pass' if b and c else 'upstream_joint_variable_null',
        'audits':{'parent_replay':maxima(replays),'deployed_preserved_modes':maxima(deployed),'exact_mode_algebra':maxima(exact),'earlier_query':maxima(causal),
                  'deployed_removed_mode_max_abs':removed,'future_mixed_read_max_abs':future,'restored':restored},
        'controls':controls,'summaries':summaries,'records':records,'authority_sha256':observed,'runner_sha256':P.sha(RUNNER),
        'price':{'model_forwards':forwards,'sequence_evaluations':forwards*4,'extra_local_contractions':layers*2,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'fit_parameters':0,'model_updates':0},
        'wall_seconds':time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','records')},indent=2))
    print(json.dumps({p:summaries[p+'/ALL'] for p in ('FIT','HOLDOUT')},indent=2))

if __name__=='__main__':main()
