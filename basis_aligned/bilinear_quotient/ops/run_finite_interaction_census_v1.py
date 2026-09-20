#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_single_module
"""Exact mixed edge replay and finite removal; opened composition diagnosis."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'finite_interaction_census_v1_result.json'
PLAN=dict(prefix=12,native_trajectory=68,additive_write=448,intervention_suffix=448,zero_removal_suffix=56)
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
        modules=[(layer,kind) for layer in range(11,18) for kind in ['attention','mlp']]
        def write(index,x):
            layer,kind=modules[index];block=model.transformer.h[layer]
            return attention_write64(block,x,first) if kind=='attention' else mlp_write64(block,x)
        def logits(x):
            z=30*torch.tanh((norm64(x[batch,read])[:,None,None,:]*model.lm_head.weight[pairs].double()).sum(-1)/30)
            return z[:,:,0]-z[:,:,1]
        def tail(index,x):
            for j in range(index+1,len(modules)):
                layer,kind=modules[j]
                if kind=='attention':
                    block=model.transformer.h[layer];x=block.lambdas[0].double()*x+block.lambdas[1].double()*x0.double()
                x=x+write(j,x)
            return logits(x)
        values={};trajectories={}
        for coordinate in [[0,0]]+binding['coordinates']:
            label=key(coordinate);z=torch.tensor(coordinate,device='cuda',dtype=torch.float64)
            x=raw.double()+torch.einsum('btdp,p->btd',directions,z);trajectory=[]
            for j,(layer,kind) in enumerate(modules):
                if kind=='attention' and layer>11:
                    block=model.transformer.h[layer];x=block.lambdas[0].double()*x+block.lambdas[1].double()*x0.double()
                w=write(j,x);trajectory.append((x,w));x=x+w
            trajectories[label]=trajectory;values[label]=logits(x);counts['native_trajectory']+=1
        old=previous[panel,template]
        replay=max(float((values['0,0']-values[label]-torch.tensor(old['double'][label],device='cuda',dtype=torch.float64)).abs().max()) for label in values if label!='0,0')
        zero_replay=0.
        for j,(x,w) in enumerate(trajectories['0,0']):
            zero_replay=max(zero_replay,float((tail(j,x+w)-values['0,0']).abs().max()));counts['zero_removal_suffix']+=1
        records=[];closure=0.
        for s,t in binding['coordinates']:
            if not s or not t:continue
            label=key([s,t]);si=key([s,0]);ti=key([0,t])
            original=values[si]+values[ti]-values['0,0']-values[label]
            for j,(layer,kind) in enumerate(modules):
                (xb,wb),(xs,ws),(xt,wt),(xj,wj)=[trajectories[k][j] for k in ['0,0',si,ti,label]]
                additive=xs+xt-xb;incoming=xj-additive
                wa=write(j,additive);counts['additive_write']+=1
                generated=wa-ws-wt+wb;transported=wj-wa
                outgoing=(xj+wj)-(xs+ws)-(xt+wt)+(xb+wb)
                closure=max(closure,float((outgoing-incoming-generated-transported).abs().max()))
                removed=tail(j,xj+wj-generated);counts['intervention_suffix']+=1
                remaining=values[si]+values[ti]-values['0,0']-removed
                effect=removed-values[label]
                for family in dict.fromkeys(old['families']):
                    ids=[i for i,f in enumerate(old['families']) if f==family]
                    norm=float(original[ids,0].norm());after=float(remaining[ids,0].norm())
                    records.append(dict(family=family,coordinate=[s,t],layer=layer,kind=kind,original_number_norm=norm,remaining_number_norm=after,remaining_ratio=after/max(norm,1e-30),effect=effect[ids].tolist(),original_interaction=original[ids].tolist(),remaining_interaction=remaining[ids].tolist(),state_norms={k:float(v[ids].norm()) for k,v in [('incoming',incoming),('generated',generated),('transported',transported)]}))
        groups.append(dict(panel=panel,template=template,records=records));checks.append(dict(saved_replay=replay,zero_removal_replay=zero_replay,closure=closure))
    failed={(r['panel'],r['family'],tuple(r['coordinate'])) for r in composition['records'] if r['mode']=='state_program' and r['kind']=='attractor_increment' and (r['number_error']>.1 or r['modal_error']>.05)}
    table=[]
    for layer,kind in modules:
        selected=[r for g in groups for r in g['records'] if r['layer']==layer and r['kind']==kind and (g['panel'],r['family'],tuple(r['coordinate'])) in failed]
        assert len(selected)==13
        table.append(dict(layer=layer,kind=kind,count=13,halved=sum(r['remaining_ratio']<=.5 for r in selected),median_remaining=float(np.median([r['remaining_ratio'] for r in selected])),worst_remaining=max(r['remaining_ratio'] for r in selected)))
    instrument=counts==PLAN and max(max(c.values()) for c in checks)<=1e-8
    result=dict(plan=PLAN,counts=counts,checks=checks,groups=groups,failed_cells_summary=table,predictions=dict(pred_a_instrument=instrument,pred_b_single_module=instrument and any(r['halved']==13 for r in table)),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps(dict(predictions=result['predictions'],counts=counts,summary=table,seconds=result['seconds'])))
if __name__=='__main__':main()
