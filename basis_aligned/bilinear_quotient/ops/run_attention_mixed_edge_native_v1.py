#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_edge_localization pred_c_causal_attenuation
"""Exact mixed edge replay and finite removal; opened composition diagnosis."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'attention_mixed_edge_native_v1_result.json'
PLAN=dict(prefix=12,suffix=104,attention=68,reader_gradients=16)
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
    counts=dict(prefix=0,suffix=0,attention=0,reader_gradients=0);groups=[];checks=[]
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
        subject=torch.tensor([r['subject_position'] for r in entries],device='cuda')
        attractor=torch.tensor([r['control_position'] for r in entries],device='cuda')
        query=torch.maximum(subject,attractor);keypos=torch.minimum(subject,attractor)
        assert bool((query>keypos).all())
        query_axis=0 if bool((subject>attractor).all()) else 1
        assert bool(((subject>attractor)==(query_axis==0)).all())
        block=model.transformer.h[11];att=block.attn
        def suffix(post):
            counts['suffix']+=1
            x=post+mlp_write64(block,post)
            for layer in range(12,18):
                b=model.transformer.h[layer];x=b.lambdas[0].double()*x+b.lambdas[1].double()*x0.double()
                x=x+attention_write64(b,x,first);x=x+mlp_write64(b,x)
            logits=30*torch.tanh((norm64(x[batch,read])[:,None,None,:]*model.lm_head.weight[pairs].double()).sum(-1)/30)
            return logits[:,:,0]-logits[:,:,1]
        coordinates=[[0,0]]+binding['coordinates'];writes={};values={}
        for coordinate in coordinates:
            z=torch.tensor(coordinate,device='cuda',dtype=torch.float64)
            state=raw.double()+torch.einsum('btdp,p->btd',directions,z)
            label=key(coordinate);writes[label]=attention_write64(block,state,first);counts['attention']+=1
            values[label]=suffix(state+writes[label])
        with torch.enable_grad():
            post=(raw.double()+writes['0,0']).detach().requires_grad_();base=suffix(post)
            readers=torch.stack([torch.autograd.grad(base[:,o].sum(),post,retain_graph=o<3)[0][batch,query] for o in range(4)],dim=1)
            counts['reader_gradients']+=4
        cos,sin=att.rotary(torch.zeros(len(entries),raw.shape[1],att.n_head,att.head_dim,device='cuda',dtype=torch.float32))
        weights={k:getattr(att,n).weight.double() for k,n in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
        core=compile_core(weights,raw[batch,query].double(),directions[batch,query,:,query_axis],raw[batch,keypos].double(),directions[batch,keypos,:,1-query_axis],readers,first.reshape_as(raw)[batch,keypos].double(),att.lamb.double(),att.n_head,cos[0,query,0].double(),sin[0,query,0].double(),cos[0,keypos,0].double(),sin[0,keypos,0].double(),torch.finfo(torch.float32).eps,torch.finfo(torch.float32).eps)
        old=previous[panel,template]
        # Use explicit float64 in the independent saved-output comparison.
        replay=max(float((values['0,0']-values[label]-torch.tensor(old['double'][label],device='cuda',dtype=torch.float64)).abs().max()) for label in values if label!='0,0')
        records=[]
        for s,t in binding['coordinates']:
            if not s or not t:continue
            label=key([s,t]);si=key([s,0]);ti=key([0,t])
            delta=writes[label]-writes[si]-writes[ti]+writes['0,0']
            selected=delta[batch,query];outside=delta.clone();outside[batch,query]=0
            local=torch.einsum('bod,bd->bo',readers,selected)
            folded=mixed(core,s if query_axis==0 else t,t if query_axis==0 else s)
            state=raw.double()+torch.einsum('btdp,p->btd',directions,torch.tensor([s,t],device='cuda',dtype=torch.float64))
            removed=suffix(state+writes[label]-delta)
            actual_i=values['0,0']-values[label]-(values['0,0']-values[si])-(values['0,0']-values[ti])
            remaining_i=values['0,0']-removed-(values['0,0']-values[si])-(values['0,0']-values[ti])
            for family in dict.fromkeys(old['families']):
                ids=[i for i,f in enumerate(old['families']) if f==family]
                original=float(actual_i[ids,0].norm());remaining=float(remaining_i[ids,0].norm())
                records.append(dict(family=family,coordinate=[s,t],original_interaction_norm=original,remaining_interaction_norm=remaining,remaining_ratio=remaining/max(original,1e-30),fixed_reader_error=float(((removed-values[label])-(-local))[ids,0].norm()),native_removal_norm=float((removed-values[label])[ids,0].norm())))
            checks.append(dict(edge_error=float((local-folded).abs().max()),outside_error=float(outside.abs().max()),saved_replay=replay))
        groups.append(dict(panel=panel,template=template,query_axis=query_axis,coefficients_per_context=sum(v.numel() for v in core.values())//len(entries),records=records))
    failed={(r['panel'],r['family'],tuple(r['coordinate'])) for r in composition['records'] if r['mode']=='state_program' and r['kind']=='attractor_increment' and (r['number_error']>.1 or r['modal_error']>.05)}
    selected=[r for g in groups for r in g['records'] if (g['panel'],r['family'],tuple(r['coordinate'])) in failed]
    assert len(selected)==13
    instrument=counts==PLAN and max(c['saved_replay'] for c in checks)<1e-10
    localization=max(c['edge_error'] for c in checks)<1e-10 and max(c['outside_error'] for c in checks)<1e-10
    result=dict(plan=PLAN,counts=counts,checks=checks,groups=groups,failed_cells=selected,predictions=dict(pred_a_instrument=instrument,pred_b_edge_localization=instrument and localization,pred_c_causal_attenuation=instrument and localization and all(r['remaining_ratio']<=.5 for r in selected)),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps(dict(predictions=result['predictions'],counts=counts,failed_remaining_ratios=[r['remaining_ratio'] for r in selected],seconds=result['seconds'])))
if __name__=='__main__':main()
