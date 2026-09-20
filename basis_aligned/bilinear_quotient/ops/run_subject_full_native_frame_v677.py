#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_joint_replay pred_b_conditional_fidelity pred_c_native_fidelity
"""Export full-native dynamic finite-observable response frame at every token position.

256 original-noun calibration rows, freeze SVD bases, evaluate48 opened position rows.
Fixed random8 matched baseline.16 prefix+20 suffix calls, zero model fits/backwards.
Full tensor contractions and exact RMS; native background ports charged.
"""
import os
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_joint_chain_v677_result.json'
PREDICTIONS=dict(pred_a_joint_replay='projected local response relative and absolute replay <=1e-4',
 pred_b_conditional_fidelity='calibration SVD8 conditional effect error <=.05 every evaluation cell',
 pred_c_native_fidelity='calibration SVD8 full native effect error <=.10 every evaluation cell')


def main():
    plan=dict(prefix_calls=16,suffix_calls=24,calibration_rows=256,evaluation_rows=48,width=8,
              additional_endpoint_chain_batches=8,basis_selection='snapshot-balanced response and four equal-weight finite-reader contrasts',observable_pairs=[['is','are'],['can','will'],['may','might'],['should','could']],controls=['dense','random_seed647'],
              analytic_reverse_passes=4,batched_readers=4,attention_pullbacks=24,reference_attention_forwards=48,all_token_positions=True,model_fits=0,backwards=0,execution_policy='managed_queue_only',predictions=PREDICTIONS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as original
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write,guard_torch_save
    sys.path.insert(0,str(POLY))
    from projected_bilinear_response import compile_response,evaluate,prepare_context,prepare_readout
    from shared_response_runtime import compile_runtime,execute as run_shared
    from conditional_mlp_chain import execute
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    decoder=json.loads((POLY/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64)
    unit=axis/axis.norm();threshold=float(decoder['threshold'])/float(axis.norm())
    eps=torch.finfo(torch.float32).eps
    U=model.lm_head.weight[[318,389]].double()
    counts=dict(prefix_calls=0,suffix_calls=0)
    def suffix(raw,x0,first,pos,frozen=None):
        counts['suffix_calls']+=1;x=raw;batch=torch.arange(len(x),device=x.device)
        result=dict(states=[],backgrounds=[],mlp_writes=[],attention={},full_states=[],raw_attention_inputs=[],full_mlp_inputs=[])
        for layer in range(11,18):
            b=model.transformer.h[layer]
            if layer>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
            if layer>=12:result['raw_attention_inputs'].append(x.double())
            if frozen is not None and layer>=12:a=frozen[layer]
            else:a,first=b.attn(F.rms_norm(x,(x.shape[-1],)),first)
            x=x+a
            if layer>=12:result['full_mlp_inputs'].append(x.double())
            m=b.mlp(F.rms_norm(x,(x.shape[-1],)))
            if layer>=12:
                result['backgrounds'].append(x[batch,pos].double())
                result['mlp_writes'].append((m[batch,pos]-b.mlp.Down_bias).double())
            x=x+m;result['states'].append(x[batch,pos].double());result['full_states'].append(x.double())
            result['attention'][layer]=a
        return result
    def margin(x,direction):
        logits=30*torch.tanh(F.rms_norm(x,(x.shape[-1],),eps=eps)@U.T/30)
        return (logits[:,0]-logits[:,1])*direction
    def capture(entries,with_truth):
        ids=torch.tensor([r['token_ids'] for r in entries],device='cuda')
        editpos=torch.tensor([r['subject_position'] for r in entries],device='cuda')
        readpos=torch.tensor([r['readout_position'] for r in entries],device='cuda')
        answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda')
        batch=torch.arange(len(entries),device='cuda');direction=torch.where(answers[:,0]==318,1.,-1.)
        initial=F.rms_norm(model.transformer.wte(ids),(model.config.n_embd,)).float()
        subject=initial[batch,editpos].double();orth=subject-(subject@unit)[:,None]*unit
        scale=((subject.square().sum(-1)-threshold**2)/orth.square().sum(-1)).sqrt()
        removed=initial.clone();removed[batch,editpos]=(threshold*unit+scale[:,None]*orth).float()
        raw,x0,first,pb,_=graph._capture(model,initial,torch,F)
        _,_,_,pr,_=graph._capture(model,removed,torch,F);counts['prefix_calls']+=2
        edited=raw.clone();edited[batch,editpos]+=sum((r-b)[batch,editpos] for b,r in zip(pb,pr)).float()
        base=suffix(raw,x0,first,readpos);frozen=suffix(edited,x0,first,readpos,base['attention']);dynamic=suffix(edited,x0,first,readpos)
        bm=margin(base['states'][-1],direction)
        conditional=bm-margin(frozen['states'][-1],direction)
        truth=None
        if with_truth:
            true=margin(dynamic['states'][-1],direction)
            truth=bm-true
        return dict(base=base,frozen=frozen,dynamic=dynamic,first=first,positions=readpos,direction=direction,conditional=conditional,truth=truth,
                    native_accuracy=float((bm>0).double().mean()),families=[r['family'] for r in entries],initial=x0[batch,readpos].double(),
                    attention_ports=[base['attention'][l][batch,readpos].double() for l in range(12,18)])
    calrows=[]
    for r in original.build_rows():
        ans=r['native_answer_id']
        calrows.append(dict(token_ids=r['token_ids'],subject_position=r['subject_position'],
             readout_position=len(r['token_ids'])-1,answer_ids=[ans,389 if ans==318 else 318],family='calibration'))
    import tiktoken
    encoding=tiktoken.get_encoding('gpt2')
    postrows=[]
    for r in original.build_rows():
        text='The '+r['subject']+' '+r['template_id']+' the '+r['attractor']
        ids=encoding.encode(text);subject_id=encoding.encode(' '+r['subject'])
        assert len(subject_id)==1 and ids.count(subject_id[0])==1
        ans=r['native_answer_id']
        postrows.append(dict(token_ids=ids,subject_position=ids.index(subject_id[0]),
            readout_position=len(ids)-1,answer_ids=[ans,389 if ans==318 else 318],family='calibration_post'))
    assert len(postrows)==128 and len({len(r['token_ids']) for r in postrows})==1
    calibration=[capture(calrows[:64],False),capture(calrows[64:],False),
                 capture(postrows[:64],False),capture(postrows[64:],False)]
    from full_suffix_readers import secant_readers
    readouts=[U]
    for pair in [('can','will'),('may','might'),('should','could')]:
        ids=[encoding.encode(' '+token) for token in pair]
        assert all(len(x)==1 for x in ids)
        readouts.append(model.lm_head.weight[[x[0] for x in ids]].double())
    readout_pairs=torch.stack(readouts)
    reader_blocks=[]
    for block in model.transformer.h[12:18]:
        a=block.attn
        reader_blocks.append(dict(left=block.mlp.Left.weight.double(),right=block.mlp.Right.weight.double(),down=block.mlp.Down.weight.double(),
            lambdas=block.lambdas.double(),attention={key:getattr(a,name).weight.double() for key,name in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]},
            mixture=a.lamb.double(),heads=a.n_head,head_eps=eps))
    reader_samples=[];reader_checks=[]
    for c in calibration:
        b,t,d=c['base']['full_states'][0].shape;blocks=[]
        for l,block in enumerate(reader_blocks):
            attention=model.transformer.h[l+12].attn
            cos,sin=attention.rotary(torch.zeros(b,t,attention.n_head,attention.head_dim,device='cuda'))
            blocks.append(dict(block,cos=cos[0,:,0].double(),sin=sin[0,:,0].double()))
        def full_trace(trace):
            return dict(states=trace['full_states'],raw_attention_inputs=trace['raw_attention_inputs'],mlp_inputs=trace['full_mlp_inputs'])
        base_trace=full_trace(c['base']);edited_trace=full_trace(c['dynamic'])
        readers=secant_readers(blocks,base_trace,edited_trace,c['first'].reshape(b,t,9,128).double(),readout_pairs,c['positions'],eps)
        def values(state):
            selected=state[torch.arange(b,device='cuda'),c['positions']]
            raw=torch.einsum('bd,rcd->rbc',F.rms_norm(selected,(d,),eps=eps),readout_pairs)
            logits=30*torch.tanh(raw/30)
            return logits[...,0]-logits[...,1]
        change=values(edited_trace['states'][-1])-values(base_trace['states'][-1])
        for l,q in enumerate(readers):
            got=(q*(edited_trace['states'][l]-base_trace['states'][l])).sum((-1,-2))
            reader_checks.append(float((got-change).abs().max()))
        reader_samples.append(readers)
    assert max(reader_checks)<=1e-4
    from balanced_response_frames import balanced_frames_qr as balanced_frames
    bases=[];encoders=[];singular=[];encoder_checks=[]
    for l in range(7):
        delta=torch.cat([(c['dynamic']['full_states'][l]-c['base']['full_states'][l]).reshape(-1,1152) for c in calibration]).cpu()
        reads=torch.cat([q[l].reshape(-1,1152) for q in reader_samples]).cpu()
        print('balancing boundary',l,'samples',len(delta),len(reads),flush=True)
        V,W,sv=balanced_frames(delta,reads,8)
        encoder_checks.append(float((W.T@V-torch.eye(8,dtype=torch.float64)).abs().max()))
        bases.append(V.cuda());encoders.append(W.cuda());singular.append(sv.tolist())
    generator=torch.Generator(device='cpu').manual_seed(647)
    random=[torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,generator=generator)).Q.cuda() for _ in range(7)]
    programs={};factors=[]
    for b in model.transformer.h[12:18]:
        factors.append((b.mlp.Left.weight.double(),b.mlp.Right.weight.double(),b.mlp.Down.weight.double()))
    for name,frames in [('calibration',bases),('random',random)]:
        outputs=encoders if name=='calibration' else frames
        programs[name]=[compile_response(*factor,frames[l],outputs[l+1]) for l,factor in enumerate(factors)]
    frames_by_name=dict(calibration=bases,random=random)
    encoders_by_name=dict(calibration=encoders,random=random)
    exported_cases=[];packed_checks=[];exported_runtime=None
    checks=[];absolute=[];reports={name:{} for name in programs}
    reports.update(input_projection_only={},final_projection_only={})
    blocks=[dict(left=b.mlp.Left.weight.double(),right=b.mlp.Right.weight.double(),
                 down=b.mlp.Down.weight.double(),bias=b.mlp.Down_bias.double(),lambdas=b.lambdas.double())
            for b in model.transformer.h[12:18]]
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        return value
    def score(c,panel):
        nonlocal exported_runtime
        z0=(c['frozen']['states'][0]-c['base']['states'][0])@encoders[0]
        contexts=[];scales=[]
        for l,p in enumerate(programs['calibration']):
            h=c['base']['backgrounds'][l];L,R,D=factors[l]
            m0=(((h@L.T)*(h@R.T))@D.T/(h.square().mean(-1,keepdim=True)+eps))@encoders[l+1]
            contexts.append(prepare_context(p,h,m0,eps))
            scales.append(model.transformer.h[l+12].lambdas[0].double())
        final_fixed,final_context=prepare_readout(c['base']['states'][-1],bases[-1],U,eps)
        runtime=compile_runtime(programs['calibration'],scales,final_fixed)
        exported_runtime=cpu(runtime)
        packed=run_shared(runtime,z0,contexts,final_context)
        exported_cases.append(cpu(dict(panel=panel,families=c['families'],initial=z0,contexts=contexts,
            final_context=final_context,reference_logits=packed,direction=c['direction'],
            conditional=c['conditional'],native=c['truth'])))
        xb=c['base']['states'][-1]
        initial=c['base']['states'][0]
        start_delta=c['frozen']['states'][0]-initial
        projected=(start_delta@encoders[0])@bases[0].T
        logits,_=execute(blocks,initial+projected,c['initial'],c['attention_ports'],U,eps)
        input_margin=(logits[:,0]-logits[:,1])*c['direction']
        final_delta=c['frozen']['states'][-1]-xb
        final_margin=margin(xb+(final_delta@encoders[-1])@bases[-1].T,c['direction'])
        for name,pred in [('input_projection_only',margin(xb,c['direction'])-input_margin),
                          ('final_projection_only',margin(xb,c['direction'])-final_margin)]:
            for i,family in enumerate(c['families']):
                key=panel+'|'+family
                r=reports[name].setdefault(key,dict(prediction=[],conditional=[],native=[]))
                r['prediction'].append(float(pred[i]));r['conditional'].append(float(c['conditional'][i]))
                if c['truth'] is not None:r['native'].append(float(c['truth'][i]))
        for name,program in programs.items():
            frames=frames_by_name[name];enc=encoders_by_name[name]
            z=(c['frozen']['states'][0]-c['base']['states'][0])@enc[0]
            for l,p in enumerate(program):
                z=z*model.transformer.h[l+12].lambdas[0].double()
                h=c['base']['backgrounds'][l];L,R,D=factors[l]
                m0=(((h@L.T)*(h@R.T))@D.T/(h.square().mean(-1,keepdim=True)+eps))@enc[l+1]
                got=evaluate(p,z,h,m0,eps)
                # Independent direct contraction at this proposed projected state.
                L,R,D=factors[l];d=z@frames[l].T
                def numerator(x):return ((x@L.T)*(x@R.T))@D.T
                dense=(d+numerator(h+d)/((h+d).square().mean(-1,keepdim=True)+eps)
                         -numerator(h)/(h.square().mean(-1,keepdim=True)+eps))@enc[l+1]
                checks.append(float((got-dense).norm()/dense.norm().clamp_min(1e-20)))
                absolute.append(float((got-dense).abs().max()));z=got
            xb=c['base']['states'][-1];xe=xb+z@frames[-1].T
            pred=margin(xb,c['direction'])-margin(xe,c['direction'])
            if name=='calibration':
                reference=30*torch.tanh(F.rms_norm(xe,(xe.shape[-1],),eps=eps)@U.T/30)
                packed_checks.append(float((packed-reference).abs().max()))
            for i,family in enumerate(c['families']):
                key=panel+'|'+family
                r=reports[name].setdefault(key,dict(prediction=[],conditional=[],native=[]))
                r['prediction'].append(float(pred[i]));r['conditional'].append(float(c['conditional'][i]))
                if c['truth'] is not None:r['native'].append(float(c['truth'][i]))
    for c in calibration:score(c,'calibration')
    rowfile=POLY/'SUBJECT_POSITION_STRESS_V644_ROWS.json'
    assert hashlib.sha256(rowfile.read_bytes()).hexdigest()=='03dd63a4b5aa3e4bf77f41280217d449084e2ad5121b0919f7ab0e07ce6c4db1'
    rows=json.loads(rowfile.read_text())
    for template in dict.fromkeys(r['template'] for r in rows):
        score(capture([r for r in rows if r['template']==template],True),'evaluation')
    for report in reports.values():
        for r in report.values():
            p=np.array(r['prediction'])
            for target in ['conditional','native']:
                if not r[target]:continue
                y=np.array(r[target]);yn=np.linalg.norm(y)
                r[target+'_metrics']=dict(relative_error=float(np.linalg.norm(p-y)/max(yn,1e-20)),
                    cosine=float(p@y/max(np.linalg.norm(p)*yn,1e-20)),target_rms=float(np.sqrt(np.mean(y*y))))
    a=max(checks)<=1e-4 and max(absolute)<=1e-4 and max(reader_checks)<=1e-4 and max(encoder_checks)<=1e-8 and max(packed_checks)<=1e-8 and counts==dict(prefix_calls=16,suffix_calls=24)
    evaluation=[r for k,r in reports['calibration'].items() if k.startswith('evaluation|')]
    b=a and all(r['conditional_metrics']['relative_error']<=.05 for r in evaluation)
    c=a and all(r['native_metrics']['relative_error']<=.10 for r in evaluation)
    values=sum(v.numel() for p in programs['calibration'] for v in p.values() if isinstance(v,torch.Tensor))+bases[-1].numel()+encoders[0].numel()+U.numel()+6
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,
        predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),reports=reports,
        max_packed_replay_absolute_error=max(packed_checks),max_encoder_duality_error=max(encoder_checks),max_reader_closure_absolute_error=max(reader_checks),max_local_relative_error=max(checks),max_local_absolute_error=max(absolute),fixed_values=values,
        context_output_encoder_values=sum(w.numel() for w in encoders[1:]),
        background_ports='six h vectors, six projected bias-free baseline MLP writes, starting delta, final baseline state',
        calibration_native_accuracy=[c['native_accuracy'] for c in calibration],calibration_singular_values=singular,wall_seconds=time.perf_counter()-tic,
        scope='Full-native dynamic response/readers at all token positions; four equal-weight contrasts, width8, unchanged256joint calibration prompts, exact QR balancing, 9216 extra initial-encoder values charged, same calibration/evaluation data; final projection is oracle. Data-informed residual basis, no native circuit adoption.')
    artifact=OUT.parent/'subject_response_v677_program.pt'
    package=dict(runtime=exported_runtime,cases=exported_cases,
        producer=cpu(dict(programs=programs['calibration'],initial_encoder=encoders[0],output_encoders=encoders[1:],
                          final_decoder=bases[-1],readout=U)),
        metadata=dict(scope='Conditional model only; native fidelity gate fails and context generation remains native',
                      fixed_values=values+sum(w.numel() for w in encoders[1:]),context_values_per_example=505,rank=8,
                      native_fidelity_gate=bool(c),predictions=result['predictions'],context_generation='Explicit output encoders included; native MLP factors and baseline generator remain external charged dependencies'))
    guard_torch_save(package,str(artifact),label='subject shared response export')
    result['artifact']=str(artifact.relative_to(ROOT))
    result['artifact_bytes']=artifact.stat().st_size
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'joint response chain')
    OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','fixed_values','max_local_relative_error','max_local_absolute_error','wall_seconds']},indent=2));assert a


if __name__=='__main__':main()
