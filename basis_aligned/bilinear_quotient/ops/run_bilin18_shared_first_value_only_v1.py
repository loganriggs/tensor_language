"""Registered token-only payload replacement, full-output screen and composition.

A: complete authorities, control, counts and finite output. B: AB teacher KL
mean<=.001/p99<=.01, command argmax>=.99 and paired-effect error<=.01 in each
phase. C: centered joint interaction/effect<=.01. Only ABC licenses follow-up;
no A/B subset rescue. 128 forwards,512 sequence evaluations, no fits/updates.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_shared_payload_fidelity pred_c_composition
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import circuit_fast_screen_producer as producer
import run_temporal_iswas_dual_command_shared_head_module_factorial_v1 as parent
import shared_first_value_only as S
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT=Path(__file__).resolve().parents[1];POLY=ROOT.parent/'polynomial_causal'
RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_SHARED_FIRST_VALUE_ONLY_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_SHARED_FIRST_VALUE_ONLY_V1_RESULT.json'
AUTHORITY=ROOT/'circuits/followups/temporal_iswas_four_head_union_joint_composition_v1_result.json'
FILES={'prior':PRIOR,'parent':AUTHORITY,'builder':ROOT/'ops/circuit_candidate_temporal_iswas_dual_command_v2.py',
       'forward':Path(parent.__file__),'primitive':Path(S.__file__)}
EXPECTED={'prior':'6239c6e3aa4304447f403e65c5e5930ed8e72a6a7ab601c98bc9dbd64230e298',
          'parent':'a79a172f23a28befcb4ba0b78f1a5ca4420c5f49323d2c51d912b46f93bb611d',
          'builder':'72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0',
          'forward':'61cc52e73d51441512b073b75eef86f41ff5941c8f97770b7a062446c47ef849',
          'primitive':'ec209a3d65e3d8b51f237dbbf2dcf85ab973bb0074735bb64704f57682e0f79c'}
ARMS={'native':{},'A':{9:(1,4)},'B':{11:(3,),15:(5,)},'AB':{9:(1,4),11:(3,),15:(5,)}}
ROLES=('temporal','iswas')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def center(t):return t-t.mean(-1,keepdim=True)

def main():
    observed={k:sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    authority=json.loads(AUTHORITY.read_text());assert authority['terminal']=='four_head_joint_program_licensed'
    rows=parent.candidate.build_rows();assert len(rows)==32
    assert rows==parent.candidate.build_rows()
    lengths=[len(e['ids']) for r in rows for e in r['endpoints'].values()]
    assert len(lengths)==128 and max(lengths)==27 and 4*max(lengths)*50304*8<256*1024**2
    dryrun={'dryrun':True,'gpu_accessed':False,'model_loaded':False,'rows':32,'sequence_evaluations':512,
            'model_forwards':128,'arms':ARMS,'max_analysis_logit_bytes':4*27*50304*8,'authority_sha256':observed}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps(dryrun));return
    assert not OUT.exists();signal.alarm(600);started=time.perf_counter()
    backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch
    torch.set_num_threads(2);checks=S.controls();assert checks['passed']
    config={k:getattr(backend.model.config,k) for k in parent.EXPECTED_CONFIG};assert config==parent.EXPECTED_CONFIG
    groups={};records=[];forwards=0;finite=True
    with torch.inference_mode():
        for row in rows:
            endpoints,lookup=parent.endpoint_bank([row]);maximum=max(len(e['ids']) for _,_,e in endpoints)
            tokens=torch.tensor([e['ids']+[50256]*(maximum-len(e['ids'])) for _,_,e in endpoints],device='cuda')
            valid=torch.arange(maximum,device='cuda')[None,:]<torch.tensor([len(e['ids']) for _,_,e in endpoints],device='cuda')[:,None]
            logits={}
            for arm,selection in ARMS.items():
                with S.remove_context_values(backend.model,selection):logits[arm],_=parent._forward(backend,tokens)
                forwards+=1;finite=finite and bool(torch.isfinite(logits[arm]).all())
            lp=logits['native'].double().log_softmax(-1);prob=lp.exp()
            kl={a:(prob*(lp-logits[a].double().log_softmax(-1))).sum(-1)[valid].cpu().numpy() for a in ('A','B','AB')}
            effect=center(logits['AB'].double()-logits['native'].double())[valid]
            interaction=center(logits['AB'].double()-logits['A'].double()-logits['B'].double()+logits['native'].double())[valid]
            commands={}
            for role in ROLES:
                pos=torch.tensor([e[role+'_position']-1 for _,_,e in endpoints],device='cuda');batch=torch.arange(4,device='cuda')
                pairs=parent.pair_indices(endpoints,lookup,role)
                margins={a:parent.margins(v,endpoints,role) for a,v in logits.items()}
                native_delta=margins['native'][pairs]-margins['native'];candidate_delta=margins['AB'][pairs]-margins['AB']
                arg={a:v[batch,pos].argmax(-1).cpu().numpy() for a,v in logits.items()}
                answer=np.asarray([e[role+'_answer_id'] for _,_,e in endpoints])
                commands[role]={'paired_error_square':float(np.square(candidate_delta-native_delta).sum()),
                    'paired_native_square':float(np.square(native_delta).sum()),'paired_count':4,
                    'argmax_agree_count':int((arg['AB']==arg['native']).sum()),
                    'native_correct_count':int((arg['native']==answer).sum()),'AB_correct_count':int((arg['AB']==answer).sum())}
            record={'row_id':row['row_id'],'phase':row['phase'],'template_id':row['template_id'],
                'token_count':int(valid.sum()),'logit_count':int(effect.numel()),
                'effect_square':float(effect.square().sum()),'interaction_square':float(interaction.square().sum()),
                'zero_effect_tokens':int((effect.square().sum(-1)==0).sum()),'commands':commands}
            records.append(record)
            for key in (row['phase']+'/ALL',row['phase']+'/'+row['template_id']):
                group=groups.setdefault(key,{'records':[],'kl':{a:[] for a in kl}});group['records'].append(record)
                for a,values in kl.items():group['kl'][a].extend(values.tolist())
    summaries={}
    for key,group in groups.items():
        rs=group['records'];total=lambda field:sum(r[field] for r in rs)
        erms=(total('effect_square')/total('logit_count'))**.5;irms=(total('interaction_square')/total('logit_count'))**.5
        commands={}
        for role in ROLES:
            cr=[r['commands'][role] for r in rs];count=sum(c['paired_count'] for c in cr)
            nrms=(sum(c['paired_native_square'] for c in cr)/count)**.5
            commands[role]={'count':count,'native_paired_rms':nrms,
                'paired_relative_error':(sum(c['paired_error_square'] for c in cr)/count)**.5/max(nrms,1e-6),
                **{out:sum(c[field] for c in cr)/count for out,field in (('native_argmax_agreement','argmax_agree_count'),('native_accuracy','native_correct_count'),('AB_accuracy','AB_correct_count'))}}
        summaries[key]={'kl':{a:{'mean':float(np.mean(v)),'p99':float(np.quantile(v,.99)),'max':float(np.max(v))} for a,v in group['kl'].items()},
            'effect_rms':erms,'interaction_rms':irms,'composition_relative_error':irms/max(erms,1e-6),
            'zero_effect_token_fraction':total('zero_effect_tokens')/total('token_count'),'commands':commands,'token_count':total('token_count')}
    selected=[summaries[p+'/ALL'] for p in ('FIT','HOLDOUT')]
    a=finite and checks['passed'] and forwards==128 and len(records)==32
    b=all(s['kl']['AB']['mean']<=.001 and s['kl']['AB']['p99']<=.01 and all(c['native_argmax_agreement']>=.99 and c['paired_relative_error']<=.01 for c in s['commands'].values()) for s in selected)
    c=all(s['composition_relative_error']<=.01 for s in selected)
    result={'scope':'Opened dual-command basic screen on actual bilin18; fresh/OOD/extraction untested.',
        'predictions':{'pred_a_instrument':a,'pred_b_shared_payload_fidelity':b,'pred_c_composition':c},
        'terminal':'invalid' if not a else 'shared_payload_screen_pass' if b and c else 'shared_payload_null',
        'summaries':summaries,'records':records,'controls':checks,'authority_sha256':observed,'runner_sha256':sha(RUNNER),
        'price':{'model_forwards':forwards,'sequence_evaluations':forwards*4,'fit_parameters':0,'model_updates':0,
                 'native_parameter_count':sum(p.numel() for p in backend.model.parameters()),'hypothetically_omittable_value_weights':589824,'realized_weight_savings':0},
        'value_lambdas':{str(l):float(backend.model.transformer.h[l].attn.lamb) for l in ARMS['AB']},
        'wall_seconds':time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','records')},indent=2))
    print(json.dumps({p:summaries[p+'/ALL'] for p in ('FIT','HOLDOUT')},indent=2))

if __name__=='__main__':main()
