"""Native routing/payload command convolution at fixed shared-value consumers.

A: native abs1e-3/relative1e-5, exact convolution abs1e-9/relative1e-10,
payload mixed zero1e-9, future mixed zero1e-8, controls/authorities/counts.
B: mixed/shared RMS>=.01 at late query in both phases. C: crossed-only
mixed-write relative error<=.01 in both phases. No head/query selection.
32 full forwards,128 sequences,192 extra local contractions; no fit/update.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_material_read_interaction pred_c_crossed_terms_suffice
import json
import os
from pathlib import Path
import signal
import sys
import time
import run_bilin18_shared_first_value_only_v1 as P
import shared_value_read_capture as R
import shared_read_command_convolution as D
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve();POLY=P.POLY
sys.path.insert(0,str(POLY))
import field_intervention_metrics as M
PRIOR=POLY/'BILIN18_ROUTED_SHARED_COMMAND_INTERACTION_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_ROUTED_SHARED_COMMAND_INTERACTION_V1_RESULT.json'
FILES={'prior':PRIOR,'incidence':POLY/'BILIN18_SHARED_VALUE_PRODUCER_INCIDENCE_V1_RESULT.json',
       'capture':Path(R.__file__),'convolution':Path(D.__file__),'transform':Path(D.C.__file__),
       'native':P.ROOT.parents[1]/'jacclust/tt_model.py'}
EXPECTED={'prior': '4f276b91d90e7853fa42f775cc66ce1ed6048e1c6f09998c0ce211571a068620', 'incidence': '42b4c9d2e6be5d6ddbacd6be46594967957850507db9be07da993bd4bf47ed05', 'capture': '9f7056b572a4feae9974fc5330a6a3cbedff8c223971a56e8b0a4bc30d10f919', 'convolution': '98d1b6bb983e31544df32131f3a88f93599a6ef266a26e3c9f892ca5398cef5d', 'transform': 'f619ada77f991a6c96def2cccee831f1a79668b3c75d84c1b0aee01046bf844c', 'native': '49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2'}


def native_agreement(actual,expected):
    delta=actual.double()-expected.double()
    absolute=float(delta.abs().max());relative=float(delta.square().mean().sqrt())/max(float(expected.double().square().mean().sqrt()),1e-6)
    finite=bool(actual.isfinite().all() and expected.isfinite().all())
    return {'max_abs':absolute,'relative_rms':relative,'finite':finite,'passed':finite and absolute<=1e-3 and relative<=1e-5}


def summarize(records):
    count=sum(r['entries'] for r in records);rms=lambda key:(sum(r[key] for r in records)/count)**.5
    mixed=rms('mixed_square');native=rms('native_mean_square');error=rms('crossed_error_square')
    return {'entries':count,'native_shared_rms':native,'mixed_rms':mixed,'mixed_over_native':mixed/max(native,1e-6),
            'crossed_error_rms':error,'crossed_relative_error':error/max(mixed,1e-6),'mixed_denominator_floored':mixed<1e-6,
            'term_rms':{name:rms(name+'_square') for name in ('joint_router','router_I_payload_T','router_T_payload_I','joint_payload','crossed')}}


def main():
    observed={k:P.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:P.sha(p) for k,p in P.FILES.items()}==P.EXPECTED
    incidence=json.loads(FILES['incidence'].read_text());assert all(incidence['predictions'].values())
    rows=P.parent.candidate.build_rows();assert len(rows)==32 and rows==P.parent.candidate.build_rows()
    for row in rows:
        es=list(row['endpoints'].values());assert len(es)==4 and len({len(e['ids']) for e in es})==1
        assert all(len({e[role+'_position'] for e in es})==1 for role in P.ROLES)
    dry={'dryrun':True,'model_loaded':False,'gpu_accessed':False,'worlds':32,'model_forwards':32,'sequence_evaluations':128,
         'extra_local_contractions':192,'authority_sha256':observed}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);started=time.perf_counter()
    backend=P.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    assert {k:getattr(backend.model.config,k) for k in P.parent.EXPECTED_CONFIG}==P.parent.EXPECTED_CONFIG
    controls={'capture':R.controls(),'convolution':D.controls()};assert all(c['passed'] for c in controls.values())
    records=[];native_audits=[];convolutions=[];payload_zero=0.;future=0.;forwards=0;layers=0;finite=True;restored=True
    with torch.inference_mode():
        for row in rows:
            endpoints,_=P.parent.endpoint_bank([row]);tokens=torch.tensor([e['ids'] for _,_,e in endpoints],device='cuda')
            with R.capture(backend.model,P.ARMS['AB']) as reads:logits,_=P.parent._forward(backend,tokens)
            forwards+=1;layers+=len(reads);finite=finite and bool(logits.isfinite().all())
            restored=restored and all('squared_attention' not in backend.model.transformer.h[l].attn.__dict__ for l in P.ARMS['AB'])
            assert set(reads)==set(P.ARMS['AB'])
            for layer,heads in reads.items():
                assert set(heads)==set(P.ARMS['AB'][layer])
                for head,r in heads.items():
                    native_audits.append(native_agreement(r['shared_read']+r['context_read'],r['native_read']))
                    native_audits.append(native_agreement(r['pattern']@r['value'],r['shared_read']))
                    u=r['value'].double()@r['weight'].double().T
                    payload={ab:u[i] for i,(_,ab,_) in enumerate(endpoints)}
                    for role in P.ROLES:
                        pos=endpoints[0][2][role+'_position']-1
                        patterns={ab:r['pattern'][i,pos:pos+1].double() for i,(_,ab,_) in enumerate(endpoints)}
                        d=D.decompose(patterns,payload);total=sum(d['terms'].values());live=d['native']['00']
                        convolutions.append(M.correspondence(total+live,d['mixed']+live))
                        for i,(_,ab,_) in enumerate(endpoints):
                            reference=r['shared_read'][i,pos:pos+1]@r['weight'].T
                            native_audits.append(native_agreement(d['native'][ab],reference))
                        payload_zero=max(payload_zero,float(d['payload_mixed'].abs().max()))
                        if role=='temporal':future=max(future,float(d['mixed'].abs().max()))
                        record={'row_id':row['row_id'],'phase':row['phase'],'template_id':row['template_id'],'layer':layer,'head':head,'role':role,
                            'entries':d['mixed'].numel(),'native_mean_square':sum(float(x.square().sum()) for x in d['native'].values())/4,
                            'mixed_square':float(d['mixed'].square().sum()),'crossed_error_square':float((d['crossed']-d['mixed']).square().sum()),
                            **{name+'_square':float(x.square().sum()) for name,x in d['terms'].items()},'crossed_square':float(d['crossed'].square().sum())}
                        records.append(record)
    summaries={}
    for phase in ('FIT','HOLDOUT'):
        selected=[r for r in records if r['phase']==phase]
        for template in ['ALL']+sorted({r['template_id'] for r in selected}):
            rs=[r for r in selected if template=='ALL' or r['template_id']==template]
            summaries[phase+'/'+template]={role:summarize([r for r in rs if r['role']==role]) for role in P.ROLES}
        summaries[phase+'/heads']={f'L{l}H{h}/{role}':summarize([r for r in selected if r['layer']==l and r['head']==h and r['role']==role])
            for l,hs in P.ARMS['AB'].items() for h in hs for role in P.ROLES}
    a=finite and restored and forwards==32 and layers==96 and len(records)==256 and payload_zero<=1e-9 and future<=1e-8 and all(x['passed'] for x in native_audits+convolutions)
    b=all(summaries[p+'/ALL']['iswas']['mixed_over_native']>=.01 for p in ('FIT','HOLDOUT'))
    c=all(summaries[p+'/ALL']['iswas']['crossed_relative_error']<=.01 for p in ('FIT','HOLDOUT'))
    result={'scope':'Native read interaction on opened command cube; no final-effect sufficiency, independent extraction or structural saving.',
        'predictions':{'pred_a_instrument':a,'pred_b_material_read_interaction':b,'pred_c_crossed_terms_suffice':c},
        'terminal':'invalid' if not a else 'crossed_read_screen_pass' if b and c else 'native_read_interaction_screen_null',
        'native_oracle_max_abs':max(x['max_abs'] for x in native_audits),'native_oracle_max_relative':max(x['relative_rms'] for x in native_audits),
        'convolution_max_abs':max(x['max_abs'] for x in convolutions),'convolution_max_relative':max(x['relative_rms'] for x in convolutions),
        'payload_mixed_max_abs':payload_zero,'future_mixed_read_max_abs':future,'restored':restored,'controls':controls,'summaries':summaries,'records':records,
        'authority_sha256':observed,'runner_sha256':P.sha(RUNNER),
        'price':{'model_forwards':forwards,'sequence_evaluations':forwards*4,'extra_local_contractions':layers*2,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'fit_parameters':0,'model_updates':0},
        'wall_seconds':time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','records')},indent=2))
    print(json.dumps({p:summaries[p+'/ALL'] for p in ('FIT','HOLDOUT')},indent=2))

if __name__=='__main__':main()
