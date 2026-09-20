#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_pair_closure pred_c_frozen_two_pairs
"""Native semantic port screen, not a standalone circuit.
24 prefix and 168 suffix batch calls (8 baseline + 160 interventions), 0 fits.
Null: direct recurrence cannot explain A; computed evidence is distributed.
All original weights and native contexts remain charged. New frozen constructions; same nouns, no fitting. Eight independent full native controls.
"""
import os,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/semantic_port_pairs_fresh_v1_result.json'
PREDICTIONS=dict(pred_a_instrument='grouped A/B replay abs<=1e-4; closed embedding recurrence relative<=1e-5; exact call counts',pred_b_pair_closure='all three pair Mobius corrections predict B within .10 every cell; baseline capability>=.90',pred_c_frozen_two_pairs='frozen pair23+pair34 plus individual effects predicts B within .10 every cell; baseline capability>=.90')
PLAN=dict(prefix_calls=24,suffix_calls=168,full_native_calls=8,fits=0,rows_per_panel=48,arms=['embedding_recurrence','early_writes_0_3','middle_writes_4_7','mlp_8','mlp_10','A','B','pair23','pair24','pair34'],predictions=PREDICTIONS,scope='Native causal source screen, full model and all context generation charged')

def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    assert not OUT.exists()
    torch.set_grad_enabled(False);torch.set_num_threads(4);start=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    dec=json.loads((POLY/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm())
    coefficient=1.
    for block in model.transformer.h[:12]:coefficient=float(block.lambdas[0])*coefficient+float(block.lambdas[1])
    data={};counts=dict(prefix=0,suffix=0,full_native=0);replay=[];closure=[];normcheck=[]
    panels=[('opposite','696','SEMANTIC_PORT_FRESH_OPPOSITE_ROWS.json','1869db8a18d4c89c28fa81a6bb57a19497e6f8692831eb4391f5d23c84556d73'),('congruent','697','SEMANTIC_PORT_FRESH_CONGRUENT_ROWS.json','bfba23ab066beb14f5cd469edce137db7ec1e8ccc26bfc36b66f896426b17511')]
    arms={name:[i] for i,name in enumerate(graph.PORTS)};arms.update(A=[0,1],B=[2,3,4],pair23=[2,3],pair24=[2,4],pair34=[3,4])
    for panel,version,filename,digest in panels:
        rf=POLY/filename;assert hashlib.sha256(rf.read_bytes()).hexdigest()==digest
        rows=json.loads(rf.read_text());data[panel]={role:{arm:{} for arm in arms} for role in ['subject','attractor']}
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template]
            tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda')
            read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda')
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float()
            raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            base=graph._suffix_margin(model,raw,x0,first,read,answers,torch,F);counts['suffix']+=1
            native=graph.native_margin(model,tokens,read,answers,torch,F);counts['full_native']+=1
            replay.append(float((base-native).abs().max()))
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit
                removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float()
                normcheck.append(float(((removed[batch,pos].double().norm(dim=1)-v.norm(dim=1)).abs()/v.norm(dim=1)).max()))
                _,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                ds=[(pe[i]-pb[i])[batch,pos] for i in range(5)]
                predicted=coefficient*(removed[batch,pos].double()-initial[batch,pos].double())
                closure.append(float((predicted-ds[0]).norm()/ds[0].norm().clamp_min(1e-30)))
                for arm,indices in arms.items():
                    edited=raw.clone();edited[batch,pos]+=sum(ds[i] for i in indices).float()
                    margin=graph._suffix_margin(model,edited,x0,first,read,answers,torch,F);counts['suffix']+=1
                    effects=(base-margin).tolist()
                    for i,r in enumerate(entries):
                        cell=data[panel][role][arm].setdefault(r['family'],dict(effects=[],margins=[],base=[]))
                        cell['effects'].append(effects[i]);cell['margins'].append(float(margin[i]));cell['base'].append(float(base[i]))
    sufficiency=[];additive=[]
    for panel,roles in data.items():
        for role,armdata in roles.items():
            for group,indices in [('A',[0,1]),('B',[2,3,4])]:
                for family,cell in armdata[group].items():
                    target=np.array(cell['effects']);den=max(np.linalg.norm(target),1e-30)
                    pred=sum(np.array(armdata[graph.PORTS[i]][family]['effects']) for i in indices)
                    additive.append(dict(panel=panel,role=role,group=group,family=family,relative_error=float(np.linalg.norm(pred-target)/den)))
                    if group=='A':sufficiency.append(dict(panel=panel,role=role,family=family,relative_error=float(np.linalg.norm(np.array(armdata[graph.PORTS[0]][family]['effects'])-target)/den)))
    signs={}
    for arm in graph.PORTS:
        stats={}
        for panel in data:
            for role in data[panel]:
                e=np.concatenate([c['effects'] for c in data[panel][role][arm].values()]);stats[panel+'_'+role]=dict(damage_fraction=float(np.mean(e>1e-4)),rms=float(np.sqrt(np.mean(e**2))))
        signs[arm]=stats
    winners=[arm for arm in graph.PORTS[1:] if all(signs[arm][p+'_subject']['damage_fraction']>=.75 for p in data) and signs[arm]['congruent_attractor']['damage_fraction']>=.75 and signs[arm]['opposite_attractor']['damage_fraction']<=.25]
    a=max(replay)<=1e-4 and max(closure)<=1e-5 and max(normcheck)<=1e-5 and counts==dict(prefix=24,suffix=168,full_native=8)
    pair_records=[]
    for panel,roles in data.items():
        for role,armdata in roles.items():
            for family,cell in armdata['B'].items():
                target=np.array(cell['effects']);den=max(np.linalg.norm(target),1e-30)
                singles={i:np.array(armdata[graph.PORTS[i]][family]['effects']) for i in [2,3,4]}
                additive_prediction=sum(singles.values())
                terms={name:np.array(armdata[name][family]['effects'])-singles[ij[0]]-singles[ij[1]] for name,ij in [('pair23',[2,3]),('pair24',[2,4]),('pair34',[3,4])]}
                pair_prediction=additive_prediction+sum(terms.values())
                pair_records.append(dict(panel=panel,role=role,family=family,target=target.tolist(),additive=additive_prediction.tolist(),pair_terms={k:v.tolist() for k,v in terms.items()},pair_prediction=pair_prediction.tolist(),third_order=(target-pair_prediction).tolist(),all_pair_error=float(np.linalg.norm(pair_prediction-target)/den),single_pair_errors={k:float(np.linalg.norm(additive_prediction+v-target)/den) for k,v in terms.items()}))
    for c in pair_records:
        y=np.array(c['target']);pred=np.array(c['additive'])+np.array(c['pair_terms']['pair23'])+np.array(c['pair_terms']['pair34'])
        c['frozen_two_pair_error']=float(np.linalg.norm(pred-y)/max(np.linalg.norm(y),1e-30))
    capabilities=[float(np.mean(np.array(cell['base'])>0)) for roles in data.values() for armdata in roles.values() for cell in armdata['B'].values()]
    shared_pairs=['pair23','pair34']
    result=dict(baseline_capabilities=capabilities,pair_records=pair_records,shared_pairs=shared_pairs,plan=PLAN,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and min(capabilities)>=.90 and max(c['all_pair_error'] for c in pair_records)<=.10),bool(a and min(capabilities)>=.90 and max(c['frozen_two_pair_error'] for c in pair_records)<=.10)])),counts=counts,embedding_coefficient=coefficient,max_embedding_closure_relative=max(closure),max_group_replay_absolute=max(replay),max_norm_error=max(normcheck),embedding_sufficiency=sufficiency,additive_baseline=additive,sign_roles=signs,screen_selected_ports=winners,data=data,seconds=time.perf_counter()-start)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps({k:result[k] for k in ['predictions','max_embedding_closure_relative','max_group_replay_absolute','screen_selected_ports','seconds']}))
if __name__=='__main__':main()
