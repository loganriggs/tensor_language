"""Exact entity/task versus entity/document query interactions, fixed hypotheses."""
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
import overlapping_join_normalizer_reference as N
import join_origin_writer_reference as O
import forward_endpoint_program_reference as F
import frozen_payload_lineage_reference as L
import frozen_query_read_reference as R
import key_payload_interaction_reference as I
import query_direction_reference as Q
import field_intervention_metrics as M
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def center(x):return x-x.mean(-1,keepdim=True)
def rms(x):return float(x.square().mean().sqrt())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    out=BASE/'ENDPOINT_MIXED_QUERY_INPUTS_V1_RESULT.json';rows=BASE/'ENDPOINT_MIXED_QUERY_INPUTS_V1_ROWS.pt'
    source=BASE/'ENDPOINT_ROUTE_QUERY_LINEAGE_V1_ROWS.pt'
    assert not out.exists() and not rows.exists()
    assert digest(source)=='f790aa9c396b12e57d207f821be9217413255d1ac170b553999426163687b048'
    previous=torch.load(source,map_location='cpu',weights_only=True)
    prior=json.loads((BASE/'ENDPOINT_ROUTE_QUERY_LINEAGE_V1_RESULT.json').read_text())
    assert prior['predictions']['pred_a_instrument'] and prior['reference_sha256']==digest(BASE/'query_direction_reference.py')
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT);model,_=load('attn4-rms-seed0');model=model.double().eval()
    audits=[];replays=[];lineage=[];partitions=[];results={};saved={}
    with torch.inference_mode():
        for pop,worlds in N.populations().items():
            routes={r:{a:[] for a in ('L','H','D','LH','LD')} for r in ('I','J')};meta=[]
            for w in worlds:
                tok=w['tokens'];context=program.prepare(tok);cache=L.prepare(program.background,tok);embed=cache['embedding'];roots={}
                for name,positions in (('L',[49]),('H',[48,50]),('D',list(range(48)))):
                    root=torch.zeros_like(embed);root[:,positions]=embed[:,positions]
                    roots[name]=L.propagate(program.background,root,cache)[:,-1]
                lineage.append(M.correspondence(sum(roots.values()),context['x'][:,-1]))
                states={**roots,'LH':roots['L']+roots['H'],'LD':roots['L']+roots['D']}
                key=O.origin_write(program.background,tok,{1:w['masks'][1][0]},{1:2})[1]
                mask=w['masks'][0].clone();mask[:,:,:48:2]=False;payload=F.messages(program.background,tok,mask)
                for name,raw in states.items():
                    changed=Q.context(program,context,raw);term=I.interaction(program,changed,key,payload);total=R.read(program,changed,payload)
                    if name!='L':
                        with Q.native_hooks(model,context['x'],raw):oracle=I.native_arms(model,context['x'],key,payload)
                        a,b,c,d=(oracle[k] for k in ((False,False),(True,False),(False,True),(True,True)))
                        audits.extend((M.correspondence(a+d,b+c+term),M.correspondence(total+c[:,-1],a[:,-1])))
                    routes['I'][name].append(term[:,-1].clone());routes['J'][name].append((total-term[:,-1]).clone())
                meta.extend(w['metadata'])
            assert meta==previous[pop]['metadata'];routes={r:{a:torch.cat(v) for a,v in arms.items()} for r,arms in routes.items()}
            native=previous[pop]['native_query_logits'];mixed={};groups={};panels={}
            for name,route in routes.items():
                full=previous[pop]['routes'][name]['mixed']
                replays.append(M.correspondence(route['L']+native,previous[pop]['routes'][name]['local']+native))
                mixed[name]={'task':route['LH']-route['L']-route['H'],'document':route['LD']-route['L']-route['D']}
                partitions.append(M.correspondence(sum(mixed[name].values())+native,full+native))
                groups[name]={};panels[name]={}
                for query in (0,1):
                    groups[name][str(query)]={}
                    for hop in range(4):
                        sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta])
                        logits={'native':native,'full':native-full,**{k:native-v for k,v in mixed[name].items()}}
                        factorial={k:M.panel(logits,k,meta,sel) for k in ('full','task','document')};den=max(rms(center(full[sel])),1e-6)
                        groups[name][str(query)][str(hop)]={'factorial':factorial,'mixed_effect_rms':rms(center(full[sel])),
                            'task_full_relative_error':rms(center(mixed[name]['task'][sel]-full[sel]))/den,
                            'document_full_relative_error':rms(center(mixed[name]['document'][sel]-full[sel]))/den}
                        if query==0 and hop==3:
                            fg=factorial['full']['gold_probability_change'];valid=abs(fg)>=.001
                            for selected,other in (('task','document'),('document','task')):
                                ratio=factorial[selected]['gold_probability_change']/fg if valid else None
                                contrast=abs(factorial[other]['gold_probability_change']/fg) if valid else None
                                panels[name][selected]={'full_gold_change':fg,'selected_ratio':ratio,'complement_absolute_ratio':contrast,
                                    'denominator_valid':valid,'nomination_passed':valid and .8<=ratio<=1.2 and contrast<=.2}
            results[pop]={'nomination':panels,'groups':groups};saved[pop]={'metadata':meta,'native_query_logits':native,'mixed_query_terms':mixed}
    predictions={'pred_a_instrument':all(a['passed'] for a in audits+replays+lineage+partitions),
        **{f'pred_{letter}_{hypothesis}':all(r[hypothesis]['nomination_passed'] for p in results.values() for r in p['nomination'].values()) for letter,hypothesis in (('b','task'),('c','document'))}}
    torch.save(saved,rows);result={'scope':'Opened exact input partition of the entity/complement query interaction; no root reselection or structural reduction.',
        'predictions':predictions,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'local_replay_max_abs':max(a['max_abs'] for a in replays),'lineage_max_abs':max(a['max_abs'] for a in lineage),
        'mixed_partition_max_abs':max(a['max_abs'] for a in partitions),'independent_worlds':32,'requests':512,
        'opaque_export_constants':program.independent_constant_count(),'source_sha256':digest(source),'rows_sha256':digest(rows),
        'runner_sha256':digest(Path(__file__)),'prereg_sha256':digest(BASE/'ENDPOINT_MIXED_QUERY_INPUTS_V1_PREREGISTRATION.md'),
        'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='populations'},indent=2))
    print(json.dumps({p:v['nomination'] for p,v in results.items()},indent=2))


if __name__=='__main__':main()
