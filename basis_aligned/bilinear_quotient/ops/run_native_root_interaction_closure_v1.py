#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_combined_prediction pred_c_root_closure
"""Exact mixed edge replay and finite removal; opened composition diagnosis."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'native_root_interaction_closure_v1_result.json'
PLAN=dict(prefix=12,suffix=68,additive_readout=32)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tiktoken
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    sys.path.insert(0,str(P))
    from native_source_observables import source_observables,norm64
    from attention_mixed_edge_core import compile_core,mixed
    from native_source_observables import attention_write64,mlp_write64
    from two_site_composition_scoring import key,score,control
    binding_path=P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json';binding=json.loads(binding_path.read_text())
    assert hashlib.sha256((P/'two_axis_state_readout.py').read_bytes()).hexdigest()==binding['implementation_sha256']
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    prior_path=A/'source_ood_v2_result.json';assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==binding['source_sha256']
    prior=json.loads(prior_path.read_text());prior_records={(r['panel'],r['role'],r['family'],r['arm']):r for r in prior['records']}
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    assert all(not b.mlp.config.gated for b in model.transformer.h[11:])
    decoder=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(decoder['threshold'])/float(axis.norm())
    enc=tiktoken.get_encoding('gpt2');words=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(w)==1 for w in words)
    modal=torch.tensor([w[0] for w in words],device='cuda').reshape(3,2)
    counts={k:0 for k in PLAN};groups=[];checks=[]
    census=json.loads((A/'finite_interaction_census_v1_result.json').read_text());census_groups={(g['panel'],g['template']):g for g in census['groups']}
    composition=json.loads((A/'two_site_composition_v1r1_result.json').read_text())
    previous={(g['panel'],g['template']):g for g in composition['groups']}
    for context in binding['contexts']:
        panel,template=context['panel'],context['template']
        rows=json.loads((P/f'SOURCE_OOD_V2_{panel.upper()}_ROWS.json').read_text());entries=[r for r in rows if r['template']==template]
        tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda')
        answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
        initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
        directions=torch.zeros(*raw.shape,2,device='cuda',dtype=torch.float64)
        for index,(role,position) in enumerate([('subject','subject_position'),('attractor','control_position')]):
            pos=torch.tensor([r[position] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit
            removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float()
            changed,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
            original=torch.stack([(pe[i]-pb[i])[batch,pos].float().double() for i in range(5)],dim=1)
            delta=(changed.double()-raw.double())[batch,pos];six=torch.cat([original,(delta-original.sum(1))[:,None,:]],dim=1)
            amplitudes=torch.tensor(context['amplitudes'][role],device='cuda',dtype=torch.float64);assert bool((amplitudes[:,1]==0).all())
            directions[batch,pos,:,index]=torch.einsum('bi,bid->bd',amplitudes,six)
        states={};values={}
        def readout(x):
            logits=30*torch.tanh((norm64(x)[:,None,None,:]*model.lm_head.weight[pairs].double()).sum(-1)/30)
            return logits[:,:,0]-logits[:,:,1]
        for coordinate in [[0,0]]+binding['coordinates']:
            label=key(coordinate);z=torch.tensor(coordinate,device='cuda',dtype=torch.float64)
            x=raw.double()+torch.einsum('btdp,p->btd',directions,z)
            for layer in range(11,18):
                block=model.transformer.h[layer]
                if layer>11:x=block.lambdas[0].double()*x+block.lambdas[1].double()*x0.double()
                x=x+attention_write64(block,x,first);x=x+mlp_write64(block,x)
            states[label]=x[batch,read];values[label]=readout(states[label]);counts['suffix']+=1
        old=previous[panel,template];census_group=census_groups[panel,template]
        replay=max(float((values['0,0']-values[label]-torch.tensor(old['double'][label],device='cuda',dtype=torch.float64)).abs().max()) for label in values if label!='0,0')
        records=[];closure=0.
        for s,t in binding['coordinates']:
            if not s or not t:continue
            label=key([s,t]);si=key([s,0]);ti=key([0,t]);base=values['0,0']
            additive=readout(states[si]+states[ti]-states['0,0']);counts['additive_readout']+=1
            generated=values[si]+values[ti]-base-additive
            transported=additive-values[label]
            total=values[si]+values[ti]-base-values[label]
            closure=max(closure,float((generated+transported-total).abs().max()))
            for family in dict.fromkeys(old['families']):
                ids=[i for i,f in enumerate(old['families']) if f==family]
                rr=[r for r in census_group['records'] if r['family']==family and r['coordinate']==[s,t]];assert len(rr)==14
                summed=torch.tensor(np.asarray([r['effect'] for r in rr]).sum(0),device='cuda',dtype=torch.float64)
                combined=summed+generated[ids];target=total[ids];norms=target.norm(dim=0)
                records.append(dict(family=family,coordinate=[s,t],target=target.tolist(),generated=generated[ids].tolist(),transported=transported[ids].tolist(),sum_module_effect=summed.tolist(),target_norms=norms.tolist(),errors={mode:(v-target).norm(dim=0).tolist() for mode,v in [('modules_only',summed),('root_plus_modules',combined),('root_only',generated[ids])]}))
        groups.append(dict(panel=panel,template=template,records=records));checks.append(dict(saved_replay=replay,root_closure=closure))
    records=[r for g in groups for r in g['records']]
    instrument=counts==PLAN and max(c['saved_replay'] for c in checks)<=1e-10
    combined=all(r['errors']['root_plus_modules'][0]<=max(1e-8,.1*r['target_norms'][0]) for r in records)
    result=dict(plan=PLAN,counts=counts,checks=checks,groups=groups,predictions=dict(pred_a_instrument=instrument,pred_b_combined_prediction=instrument and combined,pred_c_root_closure=instrument and max(c['root_closure'] for c in checks)<=1e-10),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps(dict(predictions=result['predictions'],counts=counts,seconds=result['seconds'])))
if __name__=='__main__':main()
