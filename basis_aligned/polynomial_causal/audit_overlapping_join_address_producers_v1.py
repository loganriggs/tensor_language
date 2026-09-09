"""Fixed E/origin-path/remainder split of a shared backward join's key use."""
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
import join_value_producer_reference as V
import join_origin_writer_reference as O
import overlapping_join_normalizer_reference as N
import source_port_interchange_reference as P
import key_direction_gain_reference as K
import field_intervention_metrics as M
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def center(x): return x-x.mean(-1,keepdim=True)
def rms(x): return float(x.square().mean().sqrt())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    source=BASE/'OVERLAPPING_JOIN_KEY_GAIN_V1_ROWS.pt';out=BASE/'OVERLAPPING_JOIN_ADDRESS_PRODUCERS_V1_RESULT.json';rows=BASE/'OVERLAPPING_JOIN_ADDRESS_PRODUCERS_V1_ROWS.pt'
    assert digest(source)=='83f2ac14394f78c7b6e9e084328c617f9b631bd0c865fb347c29bcdf2fda23a7'
    assert not out.exists() and not rows.exists();previous=torch.load(source,map_location='cpu',weights_only=True)
    controls={'producer':V.controls(),'origin_path':O.controls()};assert all(c['passed'] for c in controls.values())
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT);model,_=load('attn4-rms-seed0');model=model.double().eval()
    audits=[];replays=[];reuse=[];quadratic=[];results={};saved={}
    with torch.inference_mode():
        for pop,worlds in N.populations().items():
            logits={str(s):[] for s in range(8)};meta=[]
            for world in worlds:
                tok=world['tokens'];context=program.prepare(tok);mask=world['masks'][1]
                full=J.contributions(program.background,tok,{1:mask},{1:2})[1]
                value_mask=mask.clone();value_mask[:,:,:48:2]=False
                direct=V.producer_writes(program.background,tok,{1:value_mask},{1:2})[0][1]['E']
                origin=O.origin_write(program.background,tok,{1:mask[0]},{1:2})[1]
                parts=(direct,origin,full-direct-origin)
                for part in parts:reuse.append(M.correspondence(part,part[:1].expand_as(part)))
                for subset in range(8):
                    write=sum((p for bit,p in enumerate(parts) if subset&(1<<bit)),torch.zeros_like(full))
                    actual=P.execute(program,context,K.ports(program,context,write,True,False))
                    native=K.native(model,context['x'],write,True,False);audits.append(M.correspondence(actual,native))
                    logits[str(subset)].append(actual[:,-1].clone())
                meta.extend(world['metadata'])
            assert meta==previous[pop]['metadata'];logits={k:torch.cat(v) for k,v in logits.items()}
            replays.extend((M.correspondence(logits['0'],previous[pop]['query_logits']['native']),
                            M.correspondence(logits['7'],previous[pop]['query_logits']['direction'])))
            quadratic.append(M.correspondence(logits['7']+logits['1']+logits['2']+logits['4'],logits['3']+logits['5']+logits['6']+logits['0']))
            panels={};groups={}
            for query in (0,1):
                groups[str(query)]={}
                for hop in range(4):
                    sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta])
                    factorial={s:M.panel({'native':logits['0'],s:v},s,meta,sel) for s,v in logits.items()}
                    full_effect=center(logits['7'][sel]-logits['0'][sel]);den=max(rms(full_effect),1e-6)
                    rel=rms(center(logits['3'][sel]-logits['7'][sel]))/den
                    lp=logits['7'][sel].log_softmax(-1);kl=(lp.exp()*(lp-logits['3'][sel].log_softmax(-1))).sum(-1).clamp_min(0)
                    interactions={f'{a}_{b}':rms(center(logits[str(a|b)][sel]-logits[str(a)][sel]-logits[str(b)][sel]+logits['0'][sel])) for a,b in ((1,2),(1,4),(2,4))}
                    groups[str(query)][str(hop)]={'factorial':factorial,'full_effect_rms':rms(full_effect),'two_path_full_relative_error':rel,
                        'two_path_query_mean_kl':float(kl.mean()),'two_path_query_p99_kl':float(torch.quantile(kl,.99)),
                        'two_path_query_max_kl':float(kl.max()),'pair_interaction_rms':interactions,
                        'sufficiency_passed':rel<=.01 and float(kl.mean())<=.001 and float(torch.quantile(kl,.99))<=.01}
                    if hop==3:
                        full_gold=factorial['7']['gold_probability_change'];valid=abs(full_gold)>=.001
                        ratios={n:factorial[s]['gold_probability_change']/full_gold if valid else None for n,s in (('E','1'),('O','2'),('R','4'))}
                        primary='E' if query==0 else 'O';other='O' if query==0 else 'E'
                        panels[str(query)]={'full_gold_change':full_gold,'ratios':ratios,'denominator_valid':valid,
                            'two_address_nomination_passed':valid and .8<=ratios[primary]<=1.2 and abs(ratios[other])<=.2 and abs(ratios['R'])<=.2}
            results[pop]={'nomination':panels,'groups':groups};saved[pop]={'metadata':meta,'query_logits':logits}
    predictions={'pred_a_instrument':all(a['passed'] for a in audits+replays+reuse+quadratic),
        'pred_b_two_address_split':all(g['two_address_nomination_passed'] for p in results.values() for g in p['nomination'].values()),
        'pred_c_two_path_sufficiency':all(g['sufficiency_passed'] for p in results.values() for q in p['groups'].values() for g in q.values())}
    torch.save(saved,rows);result={'scope':'Opened within-head producer split at fixed native source gain; no enlarged origin field or model reduction.',
        'predictions':predictions,'controls':controls,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'saved_replay_max_abs':max(a['max_abs'] for a in replays),'reuse_max_abs':max(a['max_abs'] for a in reuse),
        'three_way_closure_max_abs':max(a['max_abs'] for a in quadratic),'independent_worlds':32,'requests':512,
        'opaque_export_constants':program.independent_constant_count(),'source_sha256':digest(source),'rows_sha256':digest(rows),
        'runner_sha256':digest(Path(__file__)),'prereg_sha256':digest(BASE/'OVERLAPPING_JOIN_ADDRESS_PRODUCERS_V1_PREREGISTRATION.md'),
        'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('populations','controls')},indent=2))
    print(json.dumps({p:{'nomination':v['nomination'],'hop3':{q:{k:x for k,x in g['3'].items() if k!='factorial'} for q,g in v['groups'].items()}} for p,v in results.items()},indent=2))


if __name__=='__main__':main()
