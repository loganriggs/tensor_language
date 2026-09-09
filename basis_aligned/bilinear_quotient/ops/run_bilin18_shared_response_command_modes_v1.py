"""Full-response command-invariance falsifier on the opened dual-command cube.

A: parent KL replay1e-5, FP64 transform abs1e-9/relative1e-10, future-bit
zero1e-8, controls/shape/finite/counts. B: minimum command-invariant response
relative RMS error<=.01 in each phase/all-token/command-query panel.
64 forwards/256 sequences; no fits, updates, new replacement or subset rescue.
"""
# C names the already registered future-bit instrument check for runner metadata.
# BQGATE: EXPERIMENT pred_a_instrument pred_b_command_independent_response pred_c_causal_order
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import run_bilin18_value_component_factorial_v1 as V
import command_response_modes as C
from circuit_fast_screen_managed_runner import atomic_create_json

P=V.P;POLY=P.POLY;RUNNER=Path(__file__).resolve()
sys.path.insert(0,str(POLY))
import field_intervention_metrics as M
PRIOR=POLY/'BILIN18_SHARED_RESPONSE_COMMAND_MODES_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_SHARED_RESPONSE_COMMAND_MODES_V1_RESULT.json'
FILES={'prior':PRIOR,'parent':V.OUT,'parent_runner':Path(V.__file__),'transform':Path(C.__file__)}
EXPECTED={'prior':'97168f04bd150ef35e16deef0cc098dfdf714e7b46205532c0b36af542d5a80e',
          'parent':'bbffa1dd0665455e84958fbf4b26ecf50623d53923f3eaaf26e3ce8e3ab25a54',
          'parent_runner':'610eb0c2ba033050b34962417d6850c62c100b83498d2a3264102403834cfa56',
          'transform':'f619ada77f991a6c96def2cccee831f1a79668b3c75d84c1b0aee01046bf844c'}

def main():
    observed={k:P.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:P.sha(p) for k,p in V.FILES.items()}==V.EXPECTED
    assert {k:P.sha(p) for k,p in P.FILES.items()}==P.EXPECTED
    previous=json.loads(V.OUT.read_text());assert previous['predictions']['pred_a_instrument']
    rows=P.parent.candidate.build_rows();assert len(rows)==32 and rows==P.parent.candidate.build_rows()
    for row in rows:
        es=list(row['endpoints'].values())
        assert len({len(e['ids']) for e in es})==1
        assert all(len({e[role+'_position'] for e in es})==1 for role in P.ROLES)
    dryrun={'dryrun':True,'model_loaded':False,'gpu_accessed':False,'worlds':32,'model_forwards':64,'sequence_evaluations':256,'authority_sha256':observed}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dryrun));return
    assert not OUT.exists();signal.alarm(600);started=time.perf_counter()
    backend=P.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    checks=C.controls();assert checks['passed'];records=[];groups={};audits=[];parseval=[];forwards=0;finite=True;future=0.
    with torch.inference_mode():
        for row in rows:
            endpoints,_=P.parent.endpoint_bank([row]);tokens=torch.tensor([e['ids'] for _,_,e in endpoints],device='cuda')
            native,_=P.parent._forward(backend,tokens);forwards+=1
            with V.S.remove_components(backend.model,P.ARMS['AB'],shared=True):changed,_=P.parent._forward(backend,tokens)
            forwards+=1;finite=finite and bool(torch.isfinite(native).all() and torch.isfinite(changed).all())
            effect=P.center(changed.double()-native.double());cells={cell:effect[i] for i,(_,cell,_) in enumerate(endpoints)}
            coeff=C.modes(cells);replay=C.reconstruct(coeff)
            for i,(_,cell,_) in enumerate(endpoints):audits.append(M.correspondence(replay[cell]+native[i],cells[cell]+native[i]))
            parseval.append(M.correspondence(sum(x.square().sum() for x in cells.values()),4*sum(x.square().sum() for x in coeff.values())))
            positions={role:endpoints[0][2][role+'_position']-1 for role in P.ROLES}
            future=max(future,*(float(coeff[uv][positions['temporal']].abs().max()) for uv in ('01','11')))
            panels={}
            for label,pos in [('all',None)]+list(positions.items()):
                selected={uv:x if pos is None else x[pos] for uv,x in coeff.items()}
                panels[label]={'entries_per_cell':selected['00'].numel(),'mode_square':{uv:float(x.square().sum()) for uv,x in selected.items()}}
            lp=native.double().log_softmax(-1);kl=(lp.exp()*(lp-changed.double().log_softmax(-1))).sum(-1).flatten().cpu().tolist()
            record={'row_id':row['row_id'],'phase':row['phase'],'template_id':row['template_id'],'panels':panels};records.append(record)
            for key in (row['phase']+'/ALL',row['phase']+'/'+row['template_id']):
                group=groups.setdefault(key,{'records':[],'kl':[]});group['records'].append(record);group['kl'].extend(kl)
    summaries={};replays=[]
    for key,g in groups.items():
        panels={}
        for label in ('all',)+P.ROLES:
            ps=[r['panels'][label] for r in g['records']];count=sum(p['entries_per_cell'] for p in ps)
            squares={uv:sum(p['mode_square'][uv] for p in ps) for uv in C.CELLS};total=sum(squares.values())
            erms=(total/count)**.5;residual=(sum(v for uv,v in squares.items() if uv!='00')/count)**.5
            panels[label]={'effect_rms':erms,'minimum_invariant_residual_rms':residual,'minimum_relative_error':residual/max(erms,1e-6),
                'denominator_floored':erms<1e-6,'mode_rms':{uv:(value/count)**.5 for uv,value in squares.items()}}
        kl={'mean':float(np.mean(g['kl'])),'p99':float(np.quantile(g['kl'],.99)),'max':float(np.max(g['kl']))}
        summaries[key]={'panels':panels,'kl':kl}
        if key.endswith('/ALL'):replays.extend(abs(value-previous['summaries'][key]['kl']['without_S'][name]) for name,value in kl.items())
    a=finite and forwards==64 and len(records)==32 and all(x['passed'] for x in audits+parseval) and future<=1e-8 and max(replays)<=1e-5
    b=all(p['minimum_relative_error']<=.01 for phase in ('FIT','HOLDOUT') for p in summaries[phase+'/ALL']['panels'].values())
    result={'scope':'Opened response-transport lower bound for a command-independent effect, not native-logit invariance, KL lower bound or structural saving.',
        'predictions':{'pred_a_instrument':a,'pred_b_command_independent_response':b,'pred_c_causal_order':future<=1e-8},'terminal':'invalid' if not a else 'invariant_response_screen' if b else 'command_invariant_response_null',
        'parent_kl_replay_max_abs':max(replays),'reconstruction_max_abs':max(x['max_abs'] for x in audits),'parseval_max_abs':max(x['max_abs'] for x in parseval),
        'future_command_mode_max_abs':future,'controls':checks,'summaries':summaries,'records':records,
        'authority_sha256':observed,'runner_sha256':P.sha(RUNNER),'price':{'model_forwards':forwards,'sequence_evaluations':forwards*4,'native_parameters':545902902,'fit_parameters':0,'model_updates':0},
        'wall_seconds':time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','records')},indent=2))
    print(json.dumps({p:summaries[p+'/ALL'] for p in ('FIT','HOLDOUT')},indent=2))

if __name__=='__main__':main()
