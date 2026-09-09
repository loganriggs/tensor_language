"""Fixed query-entity lineage versus complement for the exact I/J endpoint routes."""
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
    out=BASE/'ENDPOINT_ROUTE_QUERY_LINEAGE_V1_RESULT.json';rows=BASE/'ENDPOINT_ROUTE_QUERY_LINEAGE_V1_ROWS.pt'
    source=BASE/'ORIGIN_ENDPOINT_INTERACTION_V1_ROWS.pt'
    assert not out.exists() and not rows.exists()
    assert digest(source)=='4e25c055bc7822349261b88db9ade9bb0cad7c9caaf6d525bc56344285f82f8b'
    previous=torch.load(source,map_location='cpu',weights_only=True);controls=Q.controls();assert controls['passed']
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT);model,_=load('attn4-rms-seed0');model=model.double().eval()
    audits=[];replays=[];lineage=[];results={};saved={};zero=True
    with torch.inference_mode():
        for pop,worlds in N.populations().items():
            routes={r:{a:[] for a in ('full','local','complement','zero')} for r in ('I','J')};meta=[];native=[]
            for w in worlds:
                tok=w['tokens'];context=program.prepare(tok);cache=L.prepare(program.background,tok)
                lineage.append(M.correspondence(L.propagate(program.background,cache['embedding'],cache),context['x']))
                root=torch.zeros_like(cache['embedding']);root[:,49]=cache['embedding'][:,49]
                local=L.propagate(program.background,root,cache)[:,-1];full=context['x'][:,-1]
                query_states={'full':full,'local':local,'complement':full-local,'zero':full*0}
                key=O.origin_write(program.background,tok,{1:w['masks'][1][0]},{1:2})[1]
                mask=w['masks'][0].clone();mask[:,:,:48:2]=False;payload=F.messages(program.background,tok,mask)
                for arm,raw in query_states.items():
                    changed=Q.context(program,context,raw);term=I.interaction(program,changed,key,payload)
                    total=R.read(program,changed,payload)
                    with Q.native_hooks(model,context['x'],raw):oracle=I.native_arms(model,context['x'],key,payload)
                    a,b,c,d=(oracle[k] for k in ((False,False),(True,False),(False,True),(True,True)))
                    audits.extend((M.correspondence(a+d,b+c+term),M.correspondence(total+c[:,-1],a[:,-1])))
                    routes['I'][arm].append(term[:,-1].clone());routes['J'][arm].append((total-term[:,-1]).clone())
                    if arm=='zero':zero &= float(term[:,-1].abs().max())<1e-12 and float(total.abs().max())<1e-12
                meta.extend(w['metadata']);native.append(context['logits'][:,-1].clone())
            assert meta==previous[pop]['metadata'];native=torch.cat(native)
            routes={r:{a:torch.cat(v) for a,v in arms.items()} for r,arms in routes.items()}
            for route in routes.values():route['mixed']=route['full']-route['local']-route['complement']
            old=previous[pop]['query_logits'];replays.extend((M.correspondence(native,old['native']),
                M.correspondence(native-routes['I']['full'],old['without_interaction']),
                M.correspondence(native-routes['I']['full']-routes['J']['full'],old['value_cut'])))
            panels={};groups={}
            for query in (0,1):
                groups[str(query)]={}
                for hop in range(4):
                    sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta]);reported={}
                    for name,route in routes.items():
                        selected='local' if name=='I' else 'complement'
                        terms={**route,'remainder':route['full']-route[selected]}
                        logits={'native':native,**{k:native-v for k,v in terms.items()}}
                        factorial={k:M.panel(logits,k,meta,sel) for k in terms}
                        den=max(rms(center(route['full'][sel])),1e-6)
                        reported[name]={'factorial':factorial,'selected_query_part':selected,
                            'selected_full_relative_error':rms(center(route[selected][sel]-route['full'][sel]))/den,
                            'mixed_over_full_rms':rms(center(route['mixed'][sel]))/den}
                        if query==0 and hop==3:
                            full_effect=factorial['full']['gold_probability_change'];valid=abs(full_effect)>=.001
                            ratio=factorial[selected]['gold_probability_change']/full_effect if valid else None
                            remaining=abs(factorial['remainder']['gold_probability_change']/full_effect) if valid else None
                            panels[name]={'full_gold_change':full_effect,'selected_gold_change':factorial[selected]['gold_probability_change'],
                                'selected_ratio':ratio,'remainder_absolute_ratio':remaining,'denominator_valid':valid,
                                'nomination_passed':valid and .8<=ratio<=1.2 and remaining<=.2}
                    groups[str(query)][str(hop)]=reported
            results[pop]={'nomination':panels,'groups':groups};saved[pop]={'metadata':meta,'native_query_logits':native,'routes':routes}
    predictions={'pred_a_instrument':zero and all(a['passed'] for a in audits+replays+lineage),
        'pred_b_query_lineage':all(g['nomination_passed'] for p in results.values() for g in p['nomination'].values())}
    torch.save(saved,rows);result={'scope':'Opened query-lineage route decomposition at fixed native gain; not token-input intervention or structural reduction.',
        'predictions':predictions,'controls':controls,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'saved_replay_max_abs':max(a['max_abs'] for a in replays),'lineage_max_abs':max(a['max_abs'] for a in lineage),
        'zero_query_zero_route':zero,'independent_worlds':32,'requests':512,'opaque_export_constants':program.independent_constant_count(),
        'rows_sha256':digest(rows),'source_sha256':digest(source),'runner_sha256':digest(Path(__file__)),
        'reference_sha256':digest(BASE/'query_direction_reference.py'),'prereg_sha256':digest(BASE/'ENDPOINT_ROUTE_QUERY_LINEAGE_V1_PREREGISTRATION.md'),
        'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('populations','controls')},indent=2))
    print(json.dumps({p:{'nomination':v['nomination'],'forward_hop3':{r:{k:x for k,x in q.items() if k!='factorial'} for r,q in v['groups']['0']['3'].items()}} for p,v in results.items()},indent=2))


if __name__=='__main__':main()
