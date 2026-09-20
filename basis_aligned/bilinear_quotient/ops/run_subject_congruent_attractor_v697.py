#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_replay pred_b_effect_prediction pred_c_capable_position_transfer
"""Congruent-attractor number-removal test paired with v696 opposite-number rows.

48 opened rows; 8prefix+56suffix+24attention+8block11+12post11suffix calls.
No fits. Oracle uses native final states, not a legal predictive circuit.
Null: propagation, rather than the final response subspace, causes failed controls.
All baseline/context dependencies retained and charged as in v671.
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_attention_freeze_v697_result.json'
PREDICTIONS=dict(pred_a_exact_replay='original instrument plusattractor normpreservation1e-5 and4prefix/8block11/8suffixcalls',
 pred_b_effect_prediction='congruent attractor removal harms>=75% rows andcongruent-minus-opposite effect positive>=90%pairedrows A/B; editedcapability>=.90',
 pred_c_capable_position_transfer='attractor prediction errors<=.10 number/.05 modal relative tosubject effect norm everycell')


def main():
    plan=dict(prefix_calls=8,suffix_calls=56,sequences_per_arm=48,fits=0,backwards=0,attractor_prefix_calls=4,attractor_block11_calls=8,attractor_suffix_calls=8,extra_attention_pullbacks=8,extra_reference_forwards=16,extra_token_local_mlp_pullbacks=24,extra_fulltoken_mlp_pullbacks=24,analytic_reverse_passes=4,attention_pullbacks=24,reference_attention_forwards=48,additional_tangent_suffix_calls=8,additional_initial_interface_suffix_calls=12,
              dynamic_attention_layers=list(range(12,18)),native_background_ports=6,
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
    rowfile=POLY/'SUBJECT_CONGRUENT_ATTRACTOR_V697_ROWS.json'
    assert hashlib.sha256(rowfile.read_bytes()).hexdigest()=='e1196c12c8df57ac1aae15ab0ce6bbf484312bdf61e73b6fab926b64048e019d'
    rows=json.loads(rowfile.read_text());batches=[]
    for template in dict.fromkeys(r['template'] for r in rows):
        batches.append([(r['token_ids'],r['subject_position'],r['answer_ids'],r['family'],r['readout_position'],r['control_position'])
                        for r in rows if r['template']==template])
    assert sum(map(len,batches))==48
    import sys
    sys.path.insert(0,str(POLY))
    from projected_two_qk_attention import compile_attention,execute as folded_attention
    from full_suffix_readers import secant_readers
    from selected_suffix_readers import earliest_attention_reader,baseline_reader
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
    boundary_gaps=[];oracle_checks=[];initial_counts=0
    attractor_counts=dict(prefix=0,block11=0,suffix=0);attractor_norm_checks=[]
    attractor_reports={name:{} for name in ['A','B']}
    optimized_checks=[]
    tangent_checks=[];tangent_calls=0;reader_prices=[]
    tangent_reports={mode:{name:{} for name in ['A','B','sum']} for mode in ['linear','corrected','pruned']}
    initial_reports={name:{} for name in ['A','B','sum']}
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
            context['baseline_states']=[];context['baseline_premlp']=[]
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
                context['baseline_premlp'].append(h_all.clone())
                m_all=(((h_all@mlp.Left.weight.double().T)*(h_all@mlp.Right.weight.double().T))@mlp.Down.weight.double().T/(h_all.square().mean(-1,keepdim=True)+eps))@output_encoders[layer-12]
                context['all_mlp_contexts'].append(prepare_context(programs[layer-12],h_all,m_all,eps))
            x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
            if frozen is None:context['baseline_states'].append(x.double().clone())
            if layer==11 and frozen is None:context['baseline_after11']=x.clone()
            if layer>=12:writes[layer]=a
        batch=torch.arange(len(x),device=x.device)
        if frozen is None:
            context['baseline_final']=x[batch,pos].double().clone()
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
        context['last_native_final']=x[torch.arange(len(x),device=x.device),readpos].double().clone()
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
        apos=torch.tensor([e[5] for e in entries],device='cuda')
        assert bool((apos!=pos).all()) and bool((apos<readpos).all())
        av=initial[batch,apos].double();aorth=av-(av@unit)[:,None]*unit
        ascale=((av.square().sum(1)-target_projection**2)/aorth.square().sum(1)).sqrt()
        ar=initial.clone();ar[batch,apos]=(target_projection*unit+ascale[:,None]*aorth).float()
        attractor_norm_checks.append(float(((ar[batch,apos].double().norm(dim=1)-av.norm(dim=1)).abs()/av.norm(dim=1)).max()))
        _,_,_,aports,_=graph._capture(model,ar,torch,F);attractor_counts['prefix']+=1
        edited=raw.clone();edited[batch,pos]+=sum((y-x)[batch,pos] for x,y in zip(pb,pr)).float()
        base,background=suffix(raw,x0,first,readpos,answers)
        b,t,d=raw.shape
        blocks=[]
        for module in model.transformer.h[12:18]:
            am=module.attn;cos,sin=am.rotary(torch.zeros(b,t,am.n_head,am.head_dim,device='cuda'))
            blocks.append(dict(left=module.mlp.Left.weight.double(),right=module.mlp.Right.weight.double(),down=module.mlp.Down.weight.double(),
                lambdas=module.lambdas.double(),attention={key:getattr(am,name).weight.double() for key,name in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]},
                mixture=am.lamb.double(),heads=am.n_head,head_eps=eps,cos=cos[0,:,0].double(),sin=sin[0,:,0].double()))
        tr=dict(states=context['baseline_states'],raw_attention_inputs=context['all_background'],mlp_inputs=context['baseline_premlp'])
        reader_pairs=torch.cat([U,Ucontrol]).reshape(4,2,d)
        q0=secant_readers(blocks,tr,tr,first.reshape(b,t,9,128).double(),reader_pairs,readpos,eps)[0]
        qp=earliest_attention_reader(blocks,tr,first.reshape(b,t,9,128).double(),reader_pairs[1:],readpos,eps)
        qp_control=baseline_reader(blocks,tr,first.reshape(b,t,9,128).double(),reader_pairs[1:],readpos,eps,set(range(6)),{0})
        optimized_checks.append(float((qp-qp_control).abs().max()))
        reader_prices.append(qp.numel())
        generator=torch.Generator(device='cuda').manual_seed(686)
        small=torch.randn(context['baseline_after11'].shape,generator=generator,device='cuda',dtype=torch.float64)
        small*=.001*context['baseline_after11'].double().norm(dim=-1,keepdim=True)/small.norm(dim=-1,keepdim=True)
        plus=native_selected((context['baseline_after11'].double()+small).float(),x0,first,readpos,start_layer=12)
        minus=native_selected((context['baseline_after11'].double()-small).float(),x0,first,readpos,start_layer=12);tangent_calls+=2
        finite=((plus[:,0::2]-plus[:,1::2])-(minus[:,0::2]-minus[:,1::2]))/2
        analytical=(q0*small).sum((-1,-2)).T
        tangent_checks.append(float((finite-analytical).norm()/finite.norm().clamp_min(1e-20)))
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
            full_logits=30*torch.tanh(F.rms_norm(context['last_native_final'],(model.config.n_embd,),eps=eps)@torch.cat([U,Ucontrol]).T/30)
            oracle_checks.append(float((full_logits-selected).abs().max()))
            oracle_z=(context['last_native_final']-context['baseline_final'])@output_encoders[-1]
            oracle_ctrl=readout_prepared(context['control_fixed'],oracle_z,context['control_context'])
            oracle_modal=((cb-oracle_ctrl)[:,0::2]-(cb-oracle_ctrl)[:,1::2])
            native_effect=base-(selected[:,0]-selected[:,1])*direction
            source_z=delta.double()@Q
            predicted,final_z=run_dag(sparse_runtime,context['all_attention'],source_z,context['all_mlp_contexts'],case['final_context'],readpos)
            predicted_effect=base-(predicted[:,0]-predicted[:,1])*direction
            ctrl=readout_prepared(context['control_fixed'],final_z,context['control_context'])
            native_modal=((cb-selected[:,2:])[:,0::2]-(cb-selected[:,2:])[:,1::2])
            predicted_modal=((cb-ctrl)[:,0::2]-(cb-ctrl)[:,1::2])
            full_linear=(q0*delta.double()).sum((-1,-2)).T
            lost_linear=(q0*(delta.double()-source_z@programs[0]['input_basis'].T)).sum((-1,-2)).T
            for mode in ['linear','corrected','pruned']:
                teffect=-full_linear[:,0]*direction if mode=='linear' else predicted_effect-lost_linear[:,0]*direction
                tmodal=-full_linear[:,1:] if mode=='linear' else predicted_modal-lost_linear[:,1:]
                if mode=='pruned':
                    teffect=predicted_effect
                    tmodal=-(qp*delta.double()).sum((-1,-2)).T
                for i,e in enumerate(entries):
                    tc=tangent_reports[mode][source_name].setdefault(e[3],dict(target=[],prediction=[],native_modal=[],predicted_modal=[]))
                    tc['target'].append(float(native_effect[i]));tc['prediction'].append(float(teffect[i]))
                    tc['native_modal'].append(native_modal[i].tolist());tc['predicted_modal'].append(tmodal[i].tolist())
            initial_state=(context['baseline_after11'].double()+source_z@programs[0]['input_basis'].T).float()
            initial_native=native_selected(initial_state,x0,first,readpos,start_layer=12);initial_counts+=1
            initial_final_z=(context['last_native_final']-context['baseline_final'])@output_encoders[-1]
            initial_logits=readout_prepared(runtime['readout'],initial_final_z,case['final_context'])
            initial_ctrl=readout_prepared(context['control_fixed'],initial_final_z,context['control_context'])
            initial_effect=base-(initial_logits[:,0]-initial_logits[:,1])*direction
            initial_modal=((cb-initial_ctrl)[:,0::2]-(cb-initial_ctrl)[:,1::2])
            for i,e in enumerate(entries):
                ic=initial_reports[source_name].setdefault(e[3],dict(target=[],prediction=[],native_modal=[],predicted_modal=[],unprojected_prediction=[],unprojected_modal=[]))
                ic['target'].append(float(native_effect[i]));ic['prediction'].append(float(initial_effect[i]))
                ic['native_modal'].append(native_modal[i].tolist());ic['predicted_modal'].append(initial_modal[i].tolist())
                ic['unprojected_modal'].append((((cb-initial_native[:,2:])[:,0::2]-(cb-initial_native[:,2:])[:,1::2])[i]).tolist())
                ic['unprojected_prediction'].append(float(base[i]-(initial_native[i,0]-initial_native[i,1])*direction[i]))
            if source_name in ['A','B']:
                indices=[0,1] if source_name=='A' else [2,3,4]
                araw=raw.clone();araw[batch,apos]+=sum((aports[j]-pb[j])[batch,apos] for j in indices).float()
                block=model.transformer.h[11]
                aa,_=block.attn(F.rms_norm(araw,(araw.shape[-1],)),first)
                ast=araw+aa;ast=ast+block.mlp(F.rms_norm(ast,(ast.shape[-1],)));attractor_counts['block11']+=1
                ad=ast-context['baseline_after11']
                actual=native_selected(ast,x0,first,readpos,start_layer=12);attractor_counts['suffix']+=1
                amargin=(actual[:,0]-actual[:,1])*direction
                aeffect=base-amargin
                amodal=((cb-actual[:,2:])[:,0::2]-(cb-actual[:,2:])[:,1::2])
                al,_=run_dag(sparse_runtime,context['all_attention'],ad.double()@Q,context['all_mlp_contexts'],case['final_context'],readpos)
                peffect=base-(al[:,0]-al[:,1])*direction
                pmodal=-(qp*ad.double()).sum((-1,-2)).T
                for i,e in enumerate(entries):
                    ac=attractor_reports[source_name].setdefault(e[3],dict(target=[],prediction=[],native_modal=[],predicted_modal=[],native_margin=[],control_input_norm=[],subject_input_norm=[]))
                    ac['target'].append(float(aeffect[i]));ac['prediction'].append(float(peffect[i]))
                    ac['native_modal'].append(amodal[i].tolist());ac['predicted_modal'].append(pmodal[i].tolist());ac['native_margin'].append(float(amargin[i]))
                    ac['control_input_norm'].append(float(ad[i].norm()));ac['subject_input_norm'].append(float(delta[i].norm()))
            per_batch[source_name]=native_effect
            for i,e in enumerate(entries):
                cell=reuse_reports[source_name].setdefault(e[3],dict(target=[],prediction=[],native_modal=[],predicted_modal=[],oracle_modal=[]))
                cell['target'].append(float(native_effect[i]));cell['prediction'].append(float(predicted_effect[i]))
                cell['oracle_modal'].append(oracle_modal[i].tolist());cell['native_modal'].append(native_modal[i].tolist());cell['predicted_modal'].append(predicted_modal[i].tolist())
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
    for report in initial_reports.values():
        for cell in report.values():
            den=max(np.linalg.norm(cell['target']),1e-20)
            cell['unprojected_relative_error']=float(np.linalg.norm(np.asarray(cell['unprojected_prediction'])-cell['target'])/den)
            cell['unprojected_modal_error']=(np.linalg.norm(np.asarray(cell['unprojected_modal'])-cell['native_modal'],axis=0)/den).tolist()
            cell['relative_error']=float(np.linalg.norm(np.asarray(cell['prediction'])-cell['target'])/den)
            cell['modal_error']=(np.linalg.norm(np.asarray(cell['predicted_modal'])-cell['native_modal'],axis=0)/den).tolist()
    for report in reuse_reports.values():
        for cell in report.values():
            y=np.asarray(cell['target']);pred=np.asarray(cell['prediction']);den=max(np.linalg.norm(y),1e-20)
            cell['relative_error']=float(np.linalg.norm(pred-y)/den)
            cell['oracle_modal_error']=(np.linalg.norm(np.asarray(cell['oracle_modal'])-np.asarray(cell['native_modal']),axis=0)/den).tolist()
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
    for reports in tangent_reports.values():
        for report in reports.values():
            for cell in report.values():
                den=max(np.linalg.norm(cell['target']),1e-20)
                cell['relative_error']=float(np.linalg.norm(np.asarray(cell['prediction'])-cell['target'])/den)
                cell['modal_error']=(np.linalg.norm(np.asarray(cell['predicted_modal'])-cell['native_modal'],axis=0)/den).tolist()
    a=a and tangent_calls==8 and len(tangent_checks)==4 and max(tangent_checks)<=.01
    a=a and max(optimized_checks)<=1e-10
    b=a and all(cell['relative_error']<=.10 and max(cell['modal_error'])<=.05 for name in ['A','B'] for cell in tangent_reports['linear'][name].values())
    hybrid_reports={}
    for name in ['A','B','sum']:
        hybrid_reports[name]={}
        for family,cell in tangent_reports['pruned'][name].items():
            lc=tangent_reports['pruned'][name][family]
            native_ratio=(np.linalg.norm(cell['native_modal'],axis=0)/max(np.linalg.norm(cell['target']),1e-20)).tolist()
            hybrid_reports[name][family]=dict(target_error=cell['relative_error'],modal_error=lc['modal_error'],native_modal_ratio=native_ratio,
                prediction=cell['prediction'],predicted_modal=lc['predicted_modal'],target=cell['target'],native_modal=cell['native_modal'])
    c=a and all(cell['target_error']<=.10 and max(cell['modal_error'])<=.05 and max(cell['native_modal_ratio'])<=.10 for name in ['A','B'] for cell in hybrid_reports[name].values()) and all(cell['native_accuracy']>=.90 for cell in cells.values())
    attractor_specificity={}
    for name,report in attractor_reports.items():
        subject_values=[v for cell in reuse_reports[name].values() for v in cell['target']]
        control_values=[v for cell in report.values() for v in cell['target']]
        attractor_specificity[name]=float(np.linalg.norm(subject_values)/max(np.linalg.norm(control_values),1e-20))
        for family,cell in report.items():
            den=max(np.linalg.norm(reuse_reports[name][family]['target']),1e-20)
            cell['number_error_over_subject']=float(np.linalg.norm(np.array(cell['prediction'])-cell['target'])/den)
            cell['number_error_over_control']=float(np.linalg.norm(np.array(cell['prediction'])-cell['target'])/max(np.linalg.norm(cell['target']),1e-20))
            cell['modal_error_over_subject']=(np.linalg.norm(np.array(cell['predicted_modal'])-cell['native_modal'],axis=0)/den).tolist()
            cell['native_accuracy']=float(np.mean(np.array(cell['native_margin'])>0))
            cell['control_over_subject']=float(np.linalg.norm(cell['target'])/den)
    a=a and attractor_counts==dict(prefix=4,block11=8,suffix=8) and max(attractor_norm_checks)<=1e-5
    opposite=json.loads((OUT.parent/'subject_attention_freeze_v696_result.json').read_text())
    congruence_tests={}
    for name,report in attractor_reports.items():
        congruent=np.array([v for c in report.values() for v in c['target']])
        opposed=np.array([v for family in report for v in opposite['attractor_reports'][name][family]['target']])
        congruence_tests[name]=dict(damage_fraction=float(np.mean(congruent>1e-4)),paired_increase_fraction=float(np.mean(congruent-opposed>1e-4)),
                                   congruent_effect=congruent.tolist(),opposite_effect=opposed.tolist())
    b=a and all(t['damage_fraction']>=.75 and t['paired_increase_fraction']>=.90 for t in congruence_tests.values()) and all(c['native_accuracy']>=.90 for rs in attractor_reports.values() for c in rs.values())
    c=a and all(c['number_error_over_subject']<=.10 and max(c['modal_error_over_subject'])<=.05 for rs in attractor_reports.values() for c in rs.values())
    failed=[]

    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,cells=cells,
                predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),selectivity=selectivity,attractor_reports=attractor_reports,attractor_specificity=attractor_specificity,congruence_tests=congruence_tests,attractor_counts=attractor_counts,max_attractor_norm_error=max(attractor_norm_checks),max_optimized_reader_difference=max(optimized_checks),hybrid_reports=hybrid_reports,tangent_reports=tangent_reports,max_tangent_relative_check=max(tangent_checks),reader_values_per_batch=reader_prices,tangent_suffix_calls=tangent_calls,initial_reports=initial_reports,initial_interface_suffix_calls=initial_counts,max_full_state_readout_error=max(oracle_checks),failed_modal_oracle_ratios=[o/e for o,e in failed],reuse_reports=reuse_reports,cross_effects=cross,reuse_counts=reuse_counts,pre11_joint_vs_post11_sum_gap=boundary_gaps,
                max_baseline_replay_absolute_error=max(replay),max_exported_chain_replay_error=max(chain_checks),max_attention_projection_relative_error=max(attention_checks),additional_attention_calls=len(attention_checks),compiled_prices=compiled_prices,
                price=dict(mlp_runtime_values=tensor_values(sparse_runtime),mlp_producer_values=577536,rank=8,mlp_context_per_example=505,attention_context='native sequence background, exact compiler and first-value ports charged; T^2 r^2 score storage'),wall_seconds=time.perf_counter()-tic,
                scope='Congruent-number non-subject noun control on48new token-number combinations paired with v696; originallyfrozenframes/reader; no fitting. Opposite-number comparator isv696; sameembedding-number removal atpairedattractor, A/Bsourceports patchedonlyatthatpre11site. Native subject labels fixed. Not normmatched tosubject (inputnormsreported). All native context/reader/sourcegenerators charged; no fullmodel adoption.')
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'subject freeze')
    OUT.write_text(payload);print(json.dumps(result['predictions']));assert a


if __name__=='__main__':main()
