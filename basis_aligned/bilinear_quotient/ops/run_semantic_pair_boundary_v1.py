#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_suffix_dominance pred_c_boundary_dominance
"""Locate B-port pair interactions across block11, full native costs retained.
24 prefix,120 block11,184 post11-suffix batches;0 fits. Opened96 rows.
Competing nulls: interactions arise after block11, or in block11 and are transported.
Ordered decomposition, not unique attribution. Native capability failures retained.
"""
import os,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'semantic_pair_boundary_v1_result.json'
PREDICTIONS=dict(pred_a_instrument='prior effects replay abs1e-4 and counts24prefix120block11184suffix',pred_b_suffix_dominance='suffix-only interaction predicts total within20% every material cell',pred_c_boundary_dominance='boundary-induced interaction predicts total within20% every material cell')
PLAN=dict(prefix=24,block11=120,suffix=184,fits=0,material_threshold=.01,predictions=PREDICTIONS,scope='full model and native generators charged; no capability promotion')
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    decoder=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(decoder['threshold'])/float(axis.norm())
    counts=dict(prefix=0,block11=0,suffix=0);data={};replay=[]
    prior=json.loads((A/'semantic_port_pairs_fresh_v1_result.json').read_text())
    arms={graph.PORTS[i]:[i] for i in [2,3,4]};arms.update(pair23=[2,3],pair24=[2,4],pair34=[3,4],B=[2,3,4])
    def block11(raw,first):
        counts['block11']+=1;b=model.transformer.h[11];at,_=b.attn(F.rms_norm(raw,(raw.shape[-1],)),first);x=raw+at
        return x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
    def suffix(x,x0,first,read,answers):
        counts['suffix']+=1
        for b in model.transformer.h[12:]:
            x=b.lambdas[0]*x+b.lambdas[1]*x0;at,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+at;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
        batch=torch.arange(len(x),device=x.device)
        logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30)
        selected=logits.gather(1,answers)
        return (selected[:,0]-selected[:,1]).double()
    for panel,digest in [('opposite','1869db8a18d4c89c28fa81a6bb57a19497e6f8692831eb4391f5d23c84556d73'),('congruent','bfba23ab066beb14f5cd469edce137db7ec1e8ccc26bfc36b66f896426b17511')]:
        path=P/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        rows=json.loads(path.read_text());data[panel]={role:{} for role in ['subject','attractor']}
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda')
            read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda')
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            bstate=block11(raw,first);base=suffix(bstate,x0,first,read,answers)
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit
                removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float()
                _,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                ds={i:(pe[i]-pb[i])[batch,pos] for i in [2,3,4]};single_states={}
                for arm,indices in arms.items():
                    edited=raw.clone();edited[batch,pos]+=sum(ds[i] for i in indices).float();state=block11(edited,first)
                    actual=base-suffix(state,x0,first,read,answers)
                    if len(indices)==1:single_states[indices[0]]=state-bstate
                    summed=bstate+sum(single_states[i] for i in indices) if len(indices)>1 else state
                    hypothetical=base-suffix(summed,x0,first,read,answers) if len(indices)>1 else actual
                    for i,row in enumerate(entries):
                        c=data[panel][role].setdefault(row['family'],{}).setdefault(arm,dict(actual=[],post11_sum=[]))
                        c['actual'].append(float(actual[i]));c['post11_sum'].append(float(hypothetical[i]))
    cells=[]
    for panel,roles in data.items():
        for role,families in roles.items():
            for family,armsdata in families.items():
                for arm,c in armsdata.items():replay.append(float(np.max(np.abs(np.array(c['actual'])-prior['data'][panel][role][arm][family]['effects']))))
                den=max(np.linalg.norm(armsdata['B']['actual']),1e-30)
                for pair,indices in [('pair23',[2,3]),('pair24',[2,4]),('pair34',[3,4])]:
                    primitive=sum(np.array(armsdata[graph.PORTS[i]]['actual']) for i in indices)
                    total=np.array(armsdata[pair]['actual'])-primitive;late=np.array(armsdata[pair]['post11_sum'])-primitive;early=total-late
                    jnorm=max(np.linalg.norm(total),1e-30)
                    cells.append(dict(panel=panel,role=role,family=family,pair=pair,total=total.tolist(),suffix_only=late.tolist(),boundary_induced=early.tolist(),total_over_B=float(jnorm/den),boundary_over_B=float(np.linalg.norm(early)/den),suffix_over_B=float(np.linalg.norm(late)/den),suffix_prediction_error=float(np.linalg.norm(early)/jnorm),boundary_prediction_error=float(np.linalg.norm(late)/jnorm),material=bool(jnorm/den>=.01)))
    material=[c for c in cells if c['material']]
    a=max(replay)<=1e-4 and counts==dict(prefix=24,block11=120,suffix=184)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and material and all(c['suffix_prediction_error']<=.2 for c in material)),bool(a and material and all(c['boundary_prediction_error']<=.2 for c in material))])),max_replay_absolute=max(replay),cells=cells,data=data,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_replay_absolute','seconds']}))
if __name__=='__main__':main()
