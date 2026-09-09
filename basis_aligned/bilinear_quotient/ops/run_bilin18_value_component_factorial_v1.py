"""Fixed shared/contextual consumer deletion factorial; attribution, not replacement.

A: parent replay<=1e-5, direct-head-zero abs<=1e-4/relative<=1e-5,
controls/counts/finite pass. B: shared/context effects cosine<=-.5 and
shared-effect/native RMS>=.1 in each command/phase. C: joint contribution
relative error<=.01 in each command/phase. No head/scale/subset rescue.
129 forwards/516 sequences, one load, no fits or model updates.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_opposing_effects pred_c_independent_contributions
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import run_bilin18_shared_first_value_only_v1 as P
import shared_value_component_intervention as S
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT=P.ROOT;POLY=P.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_VALUE_COMPONENT_FACTORIAL_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_VALUE_COMPONENT_FACTORIAL_V1_RESULT.json'
FILES={'prior':PRIOR,'parent_result':P.OUT,'parent_runner':Path(P.__file__),'primitive':Path(S.__file__)}
EXPECTED={'prior':'86544d8a82bcc5343a01f6e7333e422820b359650d554644fb98e5c4e8862f6b',
          'parent_result':'447e72dbe5b032976c6aaac902fd8aea587eb3a4c2cc9fa91ca382975401014c',
          'parent_runner':'ba85983fc445adb1f858f857af9a71827f1fcff52a956ed9e82edc1677506b63',
          'primitive':'467732cbdab7d37c5d31ca9fbc93fc43d769b1b85be76e2fe183362d0926e1dc'}
ARMS={'native':(False,False),'without_C':(True,False),'without_S':(False,True),'without_both':(True,True)}
def rms(x):return float(np.sqrt(np.mean(np.square(x))))

def main():
    observed={k:P.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:P.sha(p) for k,p in P.FILES.items()}==P.EXPECTED
    previous=json.loads(P.OUT.read_text());assert previous['predictions']['pred_a_instrument']
    rows=P.parent.candidate.build_rows();assert len(rows)==32 and rows==P.parent.candidate.build_rows()
    assert max(len(e['ids']) for r in rows for e in r['endpoints'].values())==27
    dryrun={'dryrun':True,'gpu_accessed':False,'model_loaded':False,'rows':32,'sequence_evaluations':516,
            'model_forwards':129,'selection':P.ARMS['AB'],'arms':ARMS,'authority_sha256':observed,'max_analysis_logit_bytes':4*27*50304*8}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dryrun));return
    assert not OUT.exists();signal.alarm(600);started=time.perf_counter()
    backend=P.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    controls={'parent':P.S.controls(),'components':S.controls()};assert all(c['passed'] for c in controls.values())
    assert {k:getattr(backend.model.config,k) for k in P.parent.EXPECTED_CONFIG}==P.parent.EXPECTED_CONFIG
    records=[];groups={};forwards=0;finite=True;oracle={}
    with torch.inference_mode():
        for row_index,row in enumerate(rows):
            endpoints,lookup=P.parent.endpoint_bank([row]);maximum=max(len(e['ids']) for _,_,e in endpoints)
            tokens=torch.tensor([e['ids']+[50256]*(maximum-len(e['ids'])) for _,_,e in endpoints],device='cuda')
            valid=torch.arange(maximum,device='cuda')[None,:]<torch.tensor([len(e['ids']) for _,_,e in endpoints],device='cuda')[:,None]
            logits={}
            for arm,(contextual,shared) in ARMS.items():
                with S.remove_components(backend.model,P.ARMS['AB'],contextual=contextual,shared=shared):
                    logits[arm],_=P.parent._forward(backend,tokens)
                forwards+=1;finite=finite and bool(torch.isfinite(logits[arm]).all())
            if row_index==0:
                handles=[]
                for layer,heads in P.ARMS['AB'].items():
                    def zero(_module,args,heads=heads):
                        value=args[0].clone()
                        for h in heads:value[...,h*128:(h+1)*128]=0
                        return (value,)+args[1:]
                    handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(zero))
                try:direct,_=P.parent._forward(backend,tokens);forwards+=1
                finally:
                    for handle in handles:handle.remove()
                error=logits['without_both'].double()-direct.double()
                oracle={'max_abs':float(error.abs().max()),'relative_rms':float(error.square().mean().sqrt())/max(float(direct.double().square().mean().sqrt()),1e-6)}
            lp=logits['native'].double().log_softmax(-1);prob=lp.exp()
            kl={a:(prob*(lp-logits[a].double().log_softmax(-1))).sum(-1)[valid].cpu().numpy() for a in ARMS if a!='native'}
            effect=P.center(logits['without_both'].double()-logits['native'].double())[valid]
            interaction=P.center(logits['without_both'].double()-logits['without_C'].double()-logits['without_S'].double()+logits['native'].double())[valid]
            commands={}
            for role in P.ROLES:
                pairs=P.parent.pair_indices(endpoints,lookup,role);batch=torch.arange(4,device='cuda')
                pos=torch.tensor([e[role+'_position']-1 for _,_,e in endpoints],device='cuda')
                margins={a:P.parent.margins(v,endpoints,role) for a,v in logits.items()}
                commands[role]={'paired_changes':{a:(m[pairs]-m).tolist() for a,m in margins.items()},
                    'argmax':{a:v[batch,pos].argmax(-1).cpu().tolist() for a,v in logits.items()},
                    'answers':[e[role+'_answer_id'] for _,_,e in endpoints]}
            record={'row_id':row['row_id'],'phase':row['phase'],'template_id':row['template_id'],'commands':commands,
                'token_count':int(valid.sum()),'logit_count':int(effect.numel()),'effect_square':float(effect.square().sum()),'interaction_square':float(interaction.square().sum())}
            records.append(record)
            for key in (row['phase']+'/ALL',row['phase']+'/'+row['template_id']):
                g=groups.setdefault(key,{'records':[],'kl':{a:[] for a in kl}});g['records'].append(record)
                for a,v in kl.items():g['kl'][a].extend(v.tolist())
    summaries={};replays=[]
    for key,g in groups.items():
        rs=g['records'];commands={}
        for role in P.ROLES:
            cr=[r['commands'][role] for r in rs];d={a:np.asarray([x for r in cr for x in r['paired_changes'][a]]) for a in ARMS}
            e={a:v-d['native'] for a,v in d.items()};ec=e['without_C'];es=e['without_S'];ej=e['without_both'];native_rms=rms(d['native'])
            denominator=float(np.linalg.norm(ec)*np.linalg.norm(es));cos=float(ec@es/denominator) if denominator else None
            arg={a:np.asarray([x for r in cr for x in r['argmax'][a]]) for a in ARMS};answers=np.asarray([x for r in cr for x in r['answers']])
            commands[role]={'native_paired_rms':native_rms,'opposition_cosine':cos,'shared_effect_over_native':rms(es)/max(native_rms,1e-6),
                'contextual_effect_rms':rms(ec),'shared_effect_rms':rms(es),'joint_effect_rms':rms(ej),
                'composition_relative_error':rms(ej-ec-es)/max(rms(ej),1e-6),
                'signed_recovery':{a:float(v@d['native'])/max(float(d['native']@d['native']),1e-12) for a,v in d.items()},
                'parent_replay':{'native_paired_rms':native_rms,'paired_relative_error':rms(ec)/max(native_rms,1e-6),
                    'native_argmax_agreement':float(np.mean(arg['without_C']==arg['native'])),'native_accuracy':float(np.mean(arg['native']==answers)),
                    'AB_accuracy':float(np.mean(arg['without_C']==answers))}}
        logit_count=sum(r['logit_count'] for r in rs);erms=(sum(r['effect_square'] for r in rs)/logit_count)**.5;irms=(sum(r['interaction_square'] for r in rs)/logit_count)**.5
        summaries[key]={'kl':{a:{'mean':float(np.mean(v)),'p99':float(np.quantile(v,.99)),'max':float(np.max(v))} for a,v in g['kl'].items()},
            'commands':commands,'all_token_effect_rms':erms,'all_token_interaction_rms':irms,'all_token_composition_relative_error':irms/max(erms,1e-6)}
        if key.endswith('/ALL'):
            old=previous['summaries'][key]
            replays.extend(abs(v-old['kl']['AB'][k]) for k,v in summaries[key]['kl']['without_C'].items())
            for role in P.ROLES:replays.extend(abs(v-old['commands'][role][k]) for k,v in commands[role]['parent_replay'].items())
    selected=[c for phase in ('FIT','HOLDOUT') for c in summaries[phase+'/ALL']['commands'].values()]
    a=finite and forwards==129 and len(records)==32 and max(replays)<=1e-5 and oracle['max_abs']<=1e-4 and oracle['relative_rms']<=1e-5
    b=all(c['opposition_cosine'] is not None and c['opposition_cosine']<=-.5 and c['shared_effect_over_native']>=.1 for c in selected)
    c=all(c['composition_relative_error']<=.01 for c in selected)
    result={'scope':'Opened full-model consumer-local attribution; no new replacement or structural saving.',
        'predictions':{'pred_a_instrument':a,'pred_b_opposing_effects':b,'pred_c_independent_contributions':c},
        'terminal':'invalid' if not a else 'opposing_value_effects' if b else 'opposing_value_hypothesis_null',
        'parent_replay_max_abs':max(replays),'whole_head_oracle':oracle,'controls':controls,'summaries':summaries,'records':records,
        'authority_sha256':observed,'runner_sha256':P.sha(RUNNER),'price':{'model_forwards':forwards,'sequence_evaluations':forwards*4,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'fit_parameters':0,'model_updates':0},
        'wall_seconds':time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','records')},indent=2))
    print(json.dumps({p:summaries[p+'/ALL'] for p in ('FIT','HOLDOUT')},indent=2))

if __name__=='__main__':main()
