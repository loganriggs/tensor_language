"""Registered direction/gain factorial on the overlapping backward join."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(BASE)]
import torch
import exact_source_edit_reference as E
import join_contribution_context_reference as J
import overlapping_join_normalizer_reference as N
import source_port_interchange_reference as P
import key_direction_gain_reference as K
import field_intervention_metrics as M
ARMS={'native':(False,False),'direction':(True,False),'gain':(False,True),'full':(True,True)}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def center(x): return x-x.mean(-1,keepdim=True)
def rms(x): return float(x.square().mean().sqrt())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    source=BASE/'OVERLAPPING_JOIN_PORTS_V1_ROWS.pt';out=BASE/'OVERLAPPING_JOIN_KEY_GAIN_V1_RESULT.json';rows=BASE/'OVERLAPPING_JOIN_KEY_GAIN_V1_ROWS.pt'
    assert digest(source)=='9816f643cb7f42c578ab9a7db1006fdeedb94a4670e81a2cd508660352228678'
    assert not out.exists() and not rows.exists();previous=torch.load(source,map_location='cpu',weights_only=True)
    controls=K.controls();assert controls['passed']
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT);model,_=load('attn4-rms-seed0');model=model.double().eval()
    audits=[];replays=[];reuse=[];results={};saved={};support_valid=True
    with torch.inference_mode():
        for pop,worlds in N.populations().items():
            logits={arm:[] for arm in ARMS};meta=[];ratios=[]
            for world in worlds:
                tok=world['tokens'];context=program.prepare(tok);x=context['x']
                write=J.contributions(program.background,tok,world['masks'],world['heads'])[1]
                reuse.append(M.correspondence(write,write[:1].expand_as(write)))
                ratio=(K.gains(model.layers[-1],x-write)/K.gains(model.layers[-1],x)).squeeze(-1)
                support=write.ne(0).any(-1);support_valid &= bool((ratio[~support]==1).all()) and int(support.sum())==16
                ratios.append(ratio[support].reshape(8,2).clone())
                for arm,(d,g) in ARMS.items():
                    actual=P.execute(program,context,K.ports(program,context,write,d,g))
                    native=K.native(model,x,write,d,g);audits.append(M.correspondence(actual,native))
                    logits[arm].append(actual[:,-1].clone())
                meta.extend(world['metadata'])
            assert meta==previous[pop]['metadata'];logits={k:torch.cat(v) for k,v in logits.items()}
            replays.extend((M.correspondence(logits['native'],previous[pop]['query_logits']['0']),
                            M.correspondence(logits['full'],previous[pop]['query_logits']['3'])))
            panels={};groups={}
            for query in (0,1):
                groups[str(query)]={}
                for hop in range(4):
                    sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta])
                    factorial={arm:M.panel(logits,arm,meta,sel) for arm in ARMS}
                    full=factorial['full']['gold_probability_change'];direction=factorial['direction']['gold_probability_change'];gain=factorial['gain']['gold_probability_change']
                    eligible=abs(full)>=.001;dratio=direction/full if eligible else None;gratio=abs(gain/full) if eligible else None
                    effect=center(logits['full'][sel]-logits['native'][sel]);interaction=center(logits['full'][sel]-logits['direction'][sel]-logits['gain'][sel]+logits['native'][sel])
                    den=max(rms(effect),1e-6)
                    groups[str(query)][str(hop)]={'factorial':factorial,'full_effect_rms':rms(effect),
                        'direction_full_relative_error':rms(center(logits['direction'][sel]-logits['full'][sel]))/den,
                        'gain_full_relative_error':rms(center(logits['gain'][sel]-logits['full'][sel]))/den,
                        'interaction_rms':rms(interaction),'interaction_over_full_rms':rms(interaction)/den,'effect_denominator_floored':rms(effect)<1e-6}
                    if hop==3:panels[str(query)]={'full_gold_change':full,'direction_gold_change':direction,'gain_gold_change':gain,
                        'direction_ratio':dratio,'absolute_gain_ratio':gratio,'denominator_valid':eligible,
                        'direction_dominance_passed':eligible and .9<=dratio<=1.1 and gratio<=.1}
            ratios=torch.cat(ratios);results[pop]={'nomination':panels,'groups':groups,
                'source_gain_ratio':{'min':float(ratios.min()),'mean':float(ratios.mean()),'max':float(ratios.max())}}
            saved[pop]={'metadata':meta,'query_logits':logits,'source_gain_ratios':ratios}
    predictions={'pred_a_instrument':support_valid and all(a['passed'] for a in audits+replays+reuse),
        'pred_b_direction_dominance':all(g['direction_dominance_passed'] for p in results.values() for g in p['nomination'].values())}
    torch.save(saved,rows);result={'scope':'Opened direction/gain diagnosis; no sufficient source-field replacement or structural reduction.',
        'predictions':predictions,'controls':controls,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'saved_replay_max_abs':max(a['max_abs'] for a in replays),'reuse_max_abs':max(a['max_abs'] for a in reuse),
        'source_support_valid':support_valid,'independent_worlds':32,'requests':512,'opaque_export_constants':program.independent_constant_count(),
        'source_sha256':digest(source),'rows_sha256':digest(rows),'runner_sha256':digest(Path(__file__)),
        'prereg_sha256':digest(BASE/'OVERLAPPING_JOIN_KEY_GAIN_V1_PREREGISTRATION.md'),
        'reference_sha256':digest(BASE/'key_direction_gain_reference.py'),'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('populations','controls')},indent=2))
    print(json.dumps({p:{'nomination':v['nomination'],'hop3':{q:g['3'] for q,g in v['groups'].items()},'gain_ratio':v['source_gain_ratio']} for p,v in results.items()},indent=2))


if __name__=='__main__':main()
