#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_embedding_sufficiency pred_c_computed_sign_role
"""Native semantic port screen, not a standalone circuit.
24 prefix and 120 suffix batch calls (8 baseline + 112 interventions), 0 fits.
Null: direct recurrence cannot explain A; computed evidence is distributed.
All original weights and native contexts remain charged. Opened v696/v697 rows.
"""
import os,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/semantic_port_census_v1_result.json'
PREDICTIONS=dict(pred_a_instrument='grouped A/B replay abs<=1e-4; closed embedding recurrence relative<=1e-5; exact call counts',pred_b_embedding_sufficiency='embedding port predicts A within relative .10 every panel/role/family',pred_c_computed_sign_role='some computed port: subject damage >=.75 both panels; congruent attractor damage>=.75, opposite<=.25')
PLAN=dict(prefix_calls=24,suffix_calls=120,fits=0,rows_per_panel=48,arms=['embedding_recurrence','early_writes_0_3','middle_writes_4_7','mlp_8','mlp_10','A','B'],predictions=PREDICTIONS,scope='Native causal source screen, full model and all context generation charged')

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
    data={};counts=dict(prefix=0,suffix=0);replay=[];closure=[];normcheck=[]
    panels=[('opposite','696','SUBJECT_ATTRACTOR_CONTROL_V696_ROWS.json','8f5dd61488ce13d8eea3de10e832b9e12ec5c6a3bbb0d4a6c7232d119e690be8'),('congruent','697','SUBJECT_CONGRUENT_ATTRACTOR_V697_ROWS.json','e1196c12c8df57ac1aae15ab0ce6bbf484312bdf61e73b6fab926b64048e019d')]
    arms={name:[i] for i,name in enumerate(graph.PORTS)};arms.update(A=[0,1],B=[2,3,4])
    for panel,version,filename,digest in panels:
        rf=POLY/filename;assert hashlib.sha256(rf.read_bytes()).hexdigest()==digest
        rows=json.loads(rf.read_text());data[panel]={role:{arm:{} for arm in arms} for role in ['subject','attractor']}
        prior=json.loads((OUT.parent/f'subject_attention_freeze_v{version}_result.json').read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template]
            tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda')
            read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda')
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float()
            raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            base=graph._suffix_margin(model,raw,x0,first,read,answers,torch,F);counts['suffix']+=1
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
        for role in ['subject','attractor']:
            for arm in ['A','B']:
                old=prior['reuse_reports' if role=='subject' else 'attractor_reports'][arm]
                for family,cell in data[panel][role][arm].items():replay.append(float(np.max(np.abs(np.array(cell['effects'])-old[family]['target']))))
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
    a=max(replay)<=1e-4 and max(closure)<=1e-5 and max(normcheck)<=1e-5 and counts==dict(prefix=24,suffix=120)
    result=dict(plan=PLAN,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and max(c['relative_error'] for c in sufficiency)<=.10),bool(a and winners)])),counts=counts,embedding_coefficient=coefficient,max_embedding_closure_relative=max(closure),max_group_replay_absolute=max(replay),max_norm_error=max(normcheck),embedding_sufficiency=sufficiency,additive_baseline=additive,sign_roles=signs,screen_selected_ports=winners,data=data,seconds=time.perf_counter()-start)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps({k:result[k] for k in ['predictions','max_embedding_closure_relative','max_group_replay_absolute','screen_selected_ports','seconds']}))
if __name__=='__main__':main()
