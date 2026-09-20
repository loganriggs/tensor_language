#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_replay pred_b_effect_prediction pred_c_capable_position_transfer
"""Composition of projected attention12 with the exported width8 MLP response chain.

48 newly frozen noun/syntax-transfer sequences plus matched random site edits, eight prefix calls and twenty-four suffix calls, zero fits.
One exact baseline, true edited suffix, frozen edited suffix, frozen baseline replay.
Six baseline attention-write ports remain charged. Not standalone extraction.
Null: small secant attributions conceal material downstream feedback.
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_attention_freeze_v671_result.json'
PREDICTIONS=dict(pred_a_exact_replay='baseline and independent joint versus shared chain replay max absolute error <=1e-4',
 pred_b_effect_prediction='A and B source effects relative error<=.10 and modal prediction error<=.05 of own target norm in every cell',
 pred_c_capable_position_transfer='post11 superposition error<=.10 eachcell; cross-effect error<=.05 of sum effect and better than additive baseline')


def main():
    plan=dict(prefix_calls=8,suffix_calls=56,sequences_per_arm=48,fits=0,backwards=0,
              frozen_attention_layers=list(range(12,18)),native_background_ports=6,
              random_edit_seeds=list(range(66300,66308)),modal_readers=[['can','will'],['may','might'],['should','could']],additional_block11_calls=8,additional_post11_suffix_calls=12,additional_native_attention_calls=24,execution_policy='managed_queue_only',predictions=PREDICTIONS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    decoder=json.loads((POLY/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64)
    unit=axis/axis.norm();target_projection=float(decoder['threshold'])/float(axis.norm())
    import hashlib
    rowfile=POLY/'SUBJECT_JOINT_READER_FRESH_V667_ROWS.json'
    assert hashlib.sha256(rowfile.read_bytes()).hexdigest()=='6f1f1ee75784999f42622cdb2d6d5fd08069bb7d8980ceaf343fdf98ab0b0c71'
    rows=json.loads(rowfile.read_text());batches=[]
    for template in dict.fromkeys(r['template'] for r in rows):
        batches.append([(r['token_ids'],r['subject_position'],r['answer_ids'],r['family'],r['readout_position'])
                        for r in rows if r['template']==template])
    assert sum(map(len,batches))==48
    import sys
    sys.path.insert(0,str(POLY))
    from projected_two_qk_attention import compile_attention,execute as folded_attention
    from shared_response_runtime import execute as run_shared,step as shared_step
    from projected_bilinear_response import prepare_context,prepare_readout,evaluate_prepared,readout_prepared
    package=torch.load(OUT.parent/'subject_response_v665_program.pt',map_location='cpu',weights_only=True)
    def gpu(value):
        if isinstance(value,torch.Tensor):return value.cuda()
        if isinstance(value,dict):return {k:gpu(v) for k,v in value.items()}
        if isinstance(value,list):return [gpu(v) for v in value]
        return value
    runtime=gpu(package['runtime'])
    programs=gpu(package['producer']['programs'])
    output_encoders=gpu(package['producer']['output_encoders'])
    U=gpu(package['producer']['readout'])
    final_basis=gpu(package['producer']['final_decoder'])
    from sparse_response_pairs import prune_pairs
    from conditional_attention_mlp_runtime import execute as run_dag
    input_encoders=[gpu(package['producer']['initial_encoder'])]+output_encoders[:-1]
    output_decoders=[program['input_basis'] for program in programs[1:]]+[final_basis]
    sparse_runtime,_=prune_pairs(runtime,input_encoders,output_decoders,18)
    reuse_reports={name:{} for name in ['A','B','sum']};reuse_counts=dict(block11=0,post11_suffix=0)
    boundary_gaps=[]
    chain_checks=[]
    P=package['producer']['programs'][0]['input_basis'].cuda()
    Q=package['producer']['initial_encoder'].cuda()
    attn=model.transformer.h[12].attn
    weights={key:getattr(attn,name).weight.double() for key,name in
             [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
    attention_checks=[];compiled_prices=[]
    import tiktoken
    encoding=tiktoken.get_encoding('gpt2')
    control_tokens=['can','will','may','might','should','could']
    encoded=[encoding.encode(' '+word) for word in control_tokens]
    assert all(len(ids)==1 for ids in encoded)
    control_ids=[ids[0] for ids in encoded]
    Ucontrol=model.lm_head.weight[control_ids].double()
    native_collateral=[];predicted_collateral=[];target_effects=[];random_effects=[[] for _ in range(8)]
    context={};eps=torch.finfo(torch.float32).eps
    cells={};replay=[];counts=dict(prefix_calls=0,suffix_calls=0)

    def suffix(raw,x0,first,pos,answers,frozen=None,mode="freeze"):
        counts['suffix_calls']+=1;x=raw;writes={}
        batch=torch.arange(len(x),device=x.device)
        if frozen is None:
            context['mlp_contexts']=[];context['all_mlp_contexts']=[];context['all_attention']=[];context['all_zero']=[];context['all_background']=[]
        for layer in range(11,18):
            block=model.transformer.h[layer]
            if layer>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
            if layer>=12 and frozen is None:
                attention=block.attn;heads=attention.n_head;hd=attention.head_dim
                b,t,d=x.shape
                cos_all,sin_all=attention.rotary(torch.zeros(b,t,heads,hd,device=x.device))
                weights_all={key:getattr(attention,name).weight.double() for key,name in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
                decoder=programs[layer-12]['input_basis']
                encoder=Q if layer==12 else output_encoders[layer-13]
                attention_program=compile_attention(weights_all,x.double(),decoder,encoder,first.reshape(b,t,heads,hd).double(),attention.lamb.double(),heads,cos_all[0,:,0].double(),sin_all[0,:,0].double(),eps,eps)
                context['all_attention'].append(attention_program)
                context['all_zero'].append(folded_attention(attention_program,torch.zeros(b,t,8,device=x.device,dtype=torch.float64)))
                context['all_background'].append(x.double().clone())
            if layer==12 and frozen is None:
                b,t,d=x.shape;heads=attn.n_head;hd=attn.head_dim
                cos,sin=attn.rotary(torch.zeros(b,t,heads,hd,device=x.device))
                context['background']=x.double().clone()
                context['program']=compile_attention(weights,x.double(),P,Q,first.reshape(b,t,heads,hd).double(),
                    attn.lamb.double(),heads,cos[0,:,0].double(),sin[0,:,0].double(),eps,eps)
                context['zero']=folded_attention(context['program'],torch.zeros(b,t,P.shape[1],device=x.device,dtype=torch.float64))
            if frozen is not None and layer==12 and mode=='folded':
                z=(x.double()-context['background'])@Q
                projected=folded_attention(context['program'],z)-context['zero']
                context['folded_coordinates']=projected
                context['initial_coordinates']=z/float(block.lambdas[0])
                a=frozen[layer]+(projected@P.T).float()
            elif frozen is not None and layer in frozen and not(layer==12 and mode=='native12'):
                a=frozen[layer]
            else:
                a,first=block.attn(F.rms_norm(x,(x.shape[-1],)),first)
            x=x+a
            if frozen is None and layer>=12:
                h=x[batch,pos].double();mlp=block.mlp
                m0=(((h@mlp.Left.weight.double().T)*(h@mlp.Right.weight.double().T))@mlp.Down.weight.double().T/(h.square().mean(-1,keepdim=True)+eps))@output_encoders[layer-12]
                context['mlp_contexts'].append(prepare_context(programs[layer-12],h,m0,eps))
                h_all=x.double()
                m_all=(((h_all@mlp.Left.weight.double().T)*(h_all@mlp.Right.weight.double().T))@mlp.Down.weight.double().T/(h_all.square().mean(-1,keepdim=True)+eps))@output_encoders[layer-12]
                context['all_mlp_contexts'].append(prepare_context(programs[layer-12],h_all,m_all,eps))
            x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
            if layer==11 and frozen is None:context['baseline_after11']=x.clone()
            if layer>=12:writes[layer]=a
        batch=torch.arange(len(x),device=x.device)
        if frozen is None:
            context['final_fixed'],context['final_context']=prepare_readout(x[batch,pos].double(),final_basis,U,eps)
            context['control_fixed'],context['control_context']=prepare_readout(x[batch,pos].double(),final_basis,Ucontrol,eps)
            context['control_base']=30*torch.tanh(F.rms_norm(x[batch,pos],(x.shape[-1],))@Ucontrol.float().T/30)
        logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,pos],(x.shape[-1],)))/30)
        selected=logits.gather(1,answers).double()
        return selected[:,0]-selected[:,1],writes

    def native_selected(raw,x0,first,readpos,start_layer=11):
        x=raw
        for layer in range(start_layer,18):
            block=model.transformer.h[layer]
            if layer>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
            a,first=block.attn(F.rms_norm(x,(x.shape[-1],)),first)
            x=x+a;x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
        normed=F.rms_norm(x[torch.arange(len(x),device=x.device),readpos],(x.shape[-1],))
        selected=model.lm_head(normed)[:,[318,389]+control_ids]
        return (30*torch.tanh(selected/30)).double()

    for entries in batches:
        tokens=torch.tensor([e[0] for e in entries],device='cuda')
        pos=torch.tensor([e[1] for e in entries],device='cuda')
        readpos=torch.tensor([e[4] for e in entries],device='cuda')
        answers=torch.tensor([e[2] for e in entries],device='cuda')
        batch=torch.arange(len(entries),device='cuda')
        initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float()
        subject=initial[batch,pos].double();projection=subject@unit
        orthogonal=subject-projection[:,None]*unit
        scale=((subject.square().sum(1)-target_projection**2)/orthogonal.square().sum(1)).sqrt()
        removed=initial.clone();removed[batch,pos]=(target_projection*unit+scale[:,None]*orthogonal).float()
        raw,x0,first,pb,_=graph._capture(model,initial,torch,F)
        _,_,_,pr,_=graph._capture(model,removed,torch,F);counts['prefix_calls']+=2
        edited=raw.clone();edited[batch,pos]+=sum((y-x)[batch,pos] for x,y in zip(pb,pr)).float()
        base,background=suffix(raw,x0,first,readpos,answers)
        native_logits=native_selected(edited,x0,first,readpos);counts['suffix_calls']+=1
        direction_now=torch.where(answers[:,0]==318,1.,-1.)
        true=(native_logits[:,0]-native_logits[:,1])*direction_now
        true_controls=native_logits[:,2:]
        delta=edited[batch,pos]-raw[batch,pos]
        for seed_index in range(8):
            generator=torch.Generator(device='cuda').manual_seed(66300+seed_index)
            noise=torch.randn(delta.shape,generator=generator,device='cuda')
            noise=noise/noise.norm(dim=-1,keepdim=True)*delta.norm(dim=-1,keepdim=True)
            random_raw=raw.clone();random_raw[batch,pos]+=noise
            random_margin=graph._suffix_margin(model,random_raw,x0,first,readpos,answers,torch,F);counts['suffix_calls']+=1
            random_effects[seed_index].extend((base-random_margin).tolist())
        frozen,_=suffix(edited,x0,first,readpos,answers,background)
        restored,_=suffix(edited,x0,first,readpos,answers,background,mode='native12')
        folded,_=suffix(edited,x0,first,readpos,answers,background,mode='folded')
        case=dict(contexts=context['mlp_contexts'],final_context=context['final_context'],direction=torch.where(answers[:,0]==318,1.,-1.))
        initial_all=context['initial_coordinates'].clone()
        initial_z=initial_all[batch,readpos]
        innovation=context['folded_coordinates'][batch,readpos]
        plain=run_shared(runtime,initial_z,case['contexts'],case['final_context'])
        direct=initial_z
        for block,program,c in zip(runtime['blocks'],programs,case['contexts']):
            direct=evaluate_prepared(program,direct*block['scale'],c)
        reference=readout_prepared(context['final_fixed'],direct,case['final_context'])
        chain_checks.append(float((plain-reference).abs().max()))
        zero=run_shared(runtime,torch.zeros_like(initial_z),case['contexts'],case['final_context'])
        chain_checks.append(float(((zero[:,0]-zero[:,1])*case['direction']-base).abs().max()))
        composed=run_shared(runtime,initial_z,case['contexts'],case['final_context'],
                            [innovation,None,None,None,None,None])
        z_all=initial_all
        for l,(rb,ap,mlpc) in enumerate(zip(runtime['blocks'],context['all_attention'],context['all_mlp_contexts'])):
            zin=z_all*rb['scale']
            projected_attention=folded_attention(ap,zin)
            delta_attention=projected_attention-context['all_zero'][l]
            proposed=context['all_background'][l]+zin@programs[l]['input_basis'].T
            native_attention,_=model.transformer.h[l+12].attn(F.rms_norm(proposed.float(),(proposed.shape[-1],)),first)
            encoder=Q if l==0 else output_encoders[l-1]
            native_projection=native_attention.double()@encoder
            attention_checks.append(float((projected_attention-native_projection).norm()/native_projection.norm().clamp_min(1e-20)))
            z_all=shared_step(rb,z_all,mlpc,delta_attention)
        all_dynamic=readout_prepared(runtime['readout'],z_all[batch,readpos],case['final_context'])
        def tensor_values(value):
            if isinstance(value,torch.Tensor):return value.numel()
            if isinstance(value,dict):return sum(tensor_values(v) for v in value.values())
            if isinstance(value,list):return sum(tensor_values(v) for v in value)
            return 0
        compiled_prices.append(dict(sequence_length=tokens.shape[1],batch=len(entries),attention_values=tensor_values(context['all_attention']),mlp_context_values=tensor_values(context['all_mlp_contexts'])))
        direction=case['direction']
        plain_margin=(plain[:,0]-plain[:,1])*direction
        composed_margin=(all_dynamic[:,0]-all_dynamic[:,1])*direction
        controls=readout_prepared(context['control_fixed'],z_all[batch,readpos],context['control_context'])
        cb=context['control_base'].double()
        control_native=((cb-true_controls)[:,0::2]-(cb-true_controls)[:,1::2])
        control_prediction=((cb-controls)[:,0::2]-(cb-controls)[:,1::2])
        native_collateral.extend(control_native.tolist());predicted_collateral.extend(control_prediction.tolist())
        target_effects.extend((base-true).tolist())
        source_deltas=[]
        for indices in [[0,1],[2,3,4]]:
            source_raw=raw.clone()
            source_raw[batch,pos]+=sum((pr[j]-pb[j])[batch,pos] for j in indices).float()
            block=model.transformer.h[11]
            source_attention,_=block.attn(F.rms_norm(source_raw,(source_raw.shape[-1],)),first)
            source_state=source_raw+source_attention
            source_state=source_state+block.mlp(F.rms_norm(source_state,(source_state.shape[-1],)))
            source_deltas.append(source_state-context['baseline_after11']);reuse_counts['block11']+=1
        per_batch={}
        for source_name,delta in [('A',source_deltas[0]),('B',source_deltas[1]),('sum',source_deltas[0]+source_deltas[1])]:
            state=context['baseline_after11']+delta
            selected=native_selected(state,x0,first,readpos,start_layer=12);reuse_counts['post11_suffix']+=1
            native_effect=base-(selected[:,0]-selected[:,1])*direction
            source_z=delta.double()@Q
            predicted,final_z=run_dag(sparse_runtime,context['all_attention'],source_z,context['all_mlp_contexts'],case['final_context'],readpos)
            predicted_effect=base-(predicted[:,0]-predicted[:,1])*direction
            ctrl=readout_prepared(context['control_fixed'],final_z,context['control_context'])
            native_modal=((cb-selected[:,2:])[:,0::2]-(cb-selected[:,2:])[:,1::2])
            predicted_modal=((cb-ctrl)[:,0::2]-(cb-ctrl)[:,1::2])
            per_batch[source_name]=native_effect
            for i,e in enumerate(entries):
                cell=reuse_reports[source_name].setdefault(e[3],dict(target=[],prediction=[],native_modal=[],predicted_modal=[]))
                cell['target'].append(float(native_effect[i]));cell['prediction'].append(float(predicted_effect[i]))
                cell['native_modal'].append(native_modal[i].tolist());cell['predicted_modal'].append(predicted_modal[i].tolist())
        boundary_gaps.extend((per_batch['sum']-(base-true)).tolist())
        replayed,_=suffix(raw,x0,first,readpos,answers,background,mode='folded')
        replay.append(float((replayed-base).abs().max()))
        for i,e in enumerate(entries):
            c=cells.setdefault(e[3],dict(target=[],prediction=[],native_margin=[],frozen_prediction=[],restored_prediction=[],dense_folded_prediction=[],reduced_frozen_prediction=[]))
            c['native_margin'].append(float(base[i]));c['target'].append(float(base[i]-true[i]));c['prediction'].append(float(base[i]-composed_margin[i]));c['dense_folded_prediction'].append(float(base[i]-folded[i]));c['reduced_frozen_prediction'].append(float(base[i]-plain_margin[i]));c['frozen_prediction'].append(float(base[i]-frozen[i]));c['restored_prediction'].append(float(base[i]-restored[i]))
        print('batch',len(entries),'baseline replay',replay[-1],flush=True)
    for c in cells.values():
        y=np.asarray(c['target']);p=np.asarray(c['prediction']);yn=np.linalg.norm(y);pn=np.linalg.norm(p)
        dense=np.asarray(c['dense_folded_prediction'])
        c['versus_dense_layer12_relative_error']=float(np.linalg.norm(p-dense)/max(np.linalg.norm(dense),1e-30))
        c['reduced_frozen_relative_error']=float(np.linalg.norm(np.asarray(c['reduced_frozen_prediction'])-y)/max(yn,1e-30))
        c['frozen_relative_error']=float(np.linalg.norm(np.array(c['frozen_prediction'])-y)/max(yn,1e-30))
        c['restored_relative_error']=float(np.linalg.norm(np.array(c['restored_prediction'])-y)/max(yn,1e-30))
        c.update(native_accuracy=float(np.mean(np.asarray(c['native_margin'])>0)),relative_l2=float(np.linalg.norm(p-y)/max(yn,1e-30)),
                 cosine=float(p@y/max(yn*pn,1e-30)),target_rms=float(np.sqrt(np.mean(y*y))),
                 absolute_error_rms=float(np.sqrt(np.mean((p-y)**2))))
    a=max(replay)<=1e-4 and max(chain_checks)<=1e-4 and counts==dict(prefix_calls=8,suffix_calls=56)
    b=a and len(attention_checks)==24 and max(attention_checks)<=1e-4
    target_norm=float(np.linalg.norm(target_effects))
    native_control=np.asarray(native_collateral);predicted_control=np.asarray(predicted_collateral)
    collateral_ratios=np.linalg.norm(native_control,axis=0)/target_norm
    collateral_prediction_errors=np.linalg.norm(predicted_control-native_control,axis=0)/target_norm
    random_norms=np.linalg.norm(np.asarray(random_effects),axis=1)
    specificity=target_norm/float(np.median(random_norms))
    selectivity=dict(readers=[['can','will'],['may','might'],['should','could']],
        native_collateral_over_target=collateral_ratios.tolist(),prediction_error_over_target=collateral_prediction_errors.tolist(),
        target_norm=target_norm,random_effect_norms=random_norms.tolist(),target_over_median_random=specificity,
        native_collateral=native_collateral,predicted_collateral=predicted_collateral,target_effects=target_effects,random_effects=random_effects,
        scope='Number-invariant modal contrasts and equal-L2 raw-pre11 same-site edits; limited control family, not universal preservation')
    for report in reuse_reports.values():
        for cell in report.values():
            y=np.asarray(cell['target']);pred=np.asarray(cell['prediction']);den=max(np.linalg.norm(y),1e-20)
            cell['relative_error']=float(np.linalg.norm(pred-y)/den)
            cell['modal_error']=(np.linalg.norm(np.asarray(cell['predicted_modal'])-np.asarray(cell['native_modal']),axis=0)/den).tolist()
    cross={}
    for family in reuse_reports['sum']:
        A,B,S=(reuse_reports[n][family] for n in ['A','B','sum'])
        target=np.asarray(S['target'])-np.asarray(A['target'])-np.asarray(B['target'])
        pred=np.asarray(S['prediction'])-np.asarray(A['prediction'])-np.asarray(B['prediction'])
        den=max(np.linalg.norm(S['target']),1e-20)
        cross[family]=dict(target=target.tolist(),prediction=pred.tolist(),
            error_over_sum=float(np.linalg.norm(pred-target)/den),additive_error_over_sum=float(np.linalg.norm(target)/den))
    a=a and len(attention_checks)==24 and max(attention_checks)<=1e-4 and reuse_counts==dict(block11=8,post11_suffix=12)
    b=a and all(cell['relative_error']<=.10 and max(cell['modal_error'])<=.05 for name in ['A','B'] for cell in reuse_reports[name].values())
    c=a and all(cell['relative_error']<=.10 for cell in reuse_reports['sum'].values()) and all(x['error_over_sum']<=.05 and x['error_over_sum']<x['additive_error_over_sum'] for x in cross.values())

    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,cells=cells,
                predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),selectivity=selectivity,reuse_reports=reuse_reports,cross_effects=cross,reuse_counts=reuse_counts,pre11_joint_vs_post11_sum_gap=boundary_gaps,
                max_baseline_replay_absolute_error=max(replay),max_exported_chain_replay_error=max(chain_checks),max_attention_projection_relative_error=max(attention_checks),additional_attention_calls=len(attention_checks),compiled_prices=compiled_prices,
                price=dict(mlp_runtime_values=2582,mlp_producer_values=577536,rank=8,mlp_context_per_example=505,attention_context='native sequence background, exact compiler and first-value ports charged; T^2 r^2 score storage'),wall_seconds=time.perf_counter()-tic,
                scope='Post11 source-write reuse and superposition, not pre11 joint composition. 48 opened noun/syntax-transfer rows, frozen joint-observable v665 bases and unchanged v661 execution; new modal and matched-random controls. Object relative clauses and subject relatives with final adverbs; vocabulary disjoint from calibration and v660/v662, historical vocabulary reuse not excluded. Eight template-number cells; no outcome-based selection. All attention12-17 and MLP12-17 responses dynamic in reduced full-sequence states. Native baseline context, first-value and initial post11 response retained as charged ports. No fresh generalization, selective removal or port-closed circuit claim.')
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'subject freeze')
    OUT.write_text(payload);print(json.dumps(result['predictions']));assert a


if __name__=='__main__':main()
