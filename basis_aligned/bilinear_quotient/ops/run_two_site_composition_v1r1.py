#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_joint_state pred_c_increment_state pred_d_weak_axis_tripwire
"""Frozen two-site composition with full and separable quadratic baselines."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'two_site_composition_v1r1_result.json'
PREDICTIONS=dict(pred_a_instrument='counts,prior/native1e-4,weaknumerics1%,core1e-10,G/H1e-8',pred_b_joint_state='state joint number10%/modal5%all64cells',pred_c_increment_state='state conditional increments number10%/modal5%all128cells',pred_d_weak_axis_tripwire='ignored-attractor oracle fails at least one increment gate')
PLAN=dict(prefix=12,double_suffix=84,native_suffix=68,state_jvp_first=12,state_jvp_second=12,native_gradient=16,native_hessian=32,compiled_gradient=16,compiled_hessian=32,local_readout_replay=64,predictions=PREDICTIONS)


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
    from two_axis_state_readout import capture_state_coefficients,compile_core,execute,basis
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
    counts={k:0 for k in PLAN if k!='predictions'};groups=[];checks=[];native_replays=[];prior_replays=[];fold_replays=[];derivative_replays=[]
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
        zero=torch.zeros(len(entries),2,device='cuda',dtype=torch.float64)
        dummy_directions=torch.zeros(len(entries),1,raw.shape[-1],device='cuda',dtype=torch.float64);dummy_positions=torch.zeros(len(entries),device='cuda',dtype=torch.long);dummy_amplitudes=torch.zeros(len(entries),1,device='cuda',dtype=torch.float64)
        def evaluate(amplitudes,capture=None):
            counts['double_suffix']+=1
            state=raw.double()+torch.einsum('btdp,bp->btd',directions,amplitudes)
            return source_observables(model,state,x0,first,dummy_directions,dummy_positions,read,pairs,dummy_amplitudes,capture)
        def native(amplitudes):
            counts['native_suffix']+=1;x=(raw.double()+torch.einsum('btdp,bp->btd',directions,amplitudes)).float()
            for layer in range(11,18):
                block=model.transformer.h[layer]
                if layer>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
                attention,_=block.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+attention;x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
            logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30)
            values=logits.gather(1,pairs.reshape(len(x),8)).reshape(len(x),4,2)
            return (values[:,:,0]-values[:,:,1]).double()
        capture={};base=evaluate(zero,capture);baseline=native(zero)
        def state_function(amplitudes):
            local={};evaluate(amplitudes,local)
            return local['mlp_outputs'][17][batch,read]
        states,jet_checks=capture_state_coefficients(state_function,zero[:,0]);counts['state_jvp_first']+=3;counts['state_jvp_second']+=3
        jet_checks['state_baseline_replay']=float((states[:,0]-capture['mlp_outputs'][17][batch,read]).abs().max());checks.append(jet_checks)
        core=compile_core(states,model.lm_head.weight[pairs].double());assert sum(v.numel() for v in core.values())==69*len(entries)
        def derivatives(function,kind):
            with torch.enable_grad():
                z=zero.clone().requires_grad_();values=function(z);gradient=[];hessian=[]
                for output in range(4):
                    g=torch.autograd.grad(values[:,output].sum(),z,create_graph=True,retain_graph=True)[0];counts[kind+'_gradient']+=1
                    h=torch.stack([torch.autograd.grad(g[:,j].sum(),z,retain_graph=not(output==3 and j==1))[0] for j in range(2)],dim=1);counts[kind+'_hessian']+=2
                    gradient.append(g.detach());hessian.append(h.detach())
            return torch.stack(gradient,dim=1),torch.stack(hessian,dim=1)
        g,h=derivatives(evaluate,'native');cg,ch=derivatives(lambda z:execute(core,z),'compiled')
        derivative_replays.extend([float((cg-g).abs().max()),float((ch-h).abs().max())])
        zeros=torch.zeros_like(base).tolist();group=dict(panel=panel,template=template,families=[r['family'] for r in entries],coordinates=binding['coordinates'],target={'0,0':zeros},double={'0,0':zeros},predictions={mode:{'0,0':zeros} for mode in ['full_quadratic','separable_quadratic','state_program']},gradient=g.tolist(),hessian=h.tolist(),compiled_core={k:v.tolist() for k,v in core.items()})
        for coordinate in binding['coordinates']:
            amplitudes=torch.tensor(coordinate,device='cuda',dtype=torch.float64).expand_as(zero);label=key(coordinate)
            actual=baseline-native(amplitudes);reference=base-evaluate(amplitudes);native_replays.append(float((actual-reference).abs().max()))
            full=-torch.einsum('bop,bp->bo',g,amplitudes)-.5*torch.einsum('bp,bopq,bq->bo',amplitudes,h,amplitudes)
            separable=-torch.einsum('bop,bp->bo',g,amplitudes)-.5*torch.einsum('bop,bp->bo',h.diagonal(dim1=-2,dim2=-1),amplitudes.square())
            state_prediction=base-execute(core,amplitudes)
            x=torch.einsum('bk,bkd->bd',basis(amplitudes),states);logits=30*torch.tanh((norm64(x)[:,None,None,:]*model.lm_head.weight[pairs].double()).sum(-1)/30);dense=logits[:,:,0]-logits[:,:,1];counts['local_readout_replay']+=1
            fold_replays.append(float((execute(core,amplitudes)-dense).abs().max()))
            group['target'][label]=actual.tolist();group['double'][label]=reference.tolist()
            for mode,values in [('full_quadratic',full),('separable_quadratic',separable),('state_program',state_prediction)]:group['predictions'][mode][label]=values.tolist()
            if coordinate in [[1,0],[0,1]]:
                role='subject' if coordinate==[1,0] else 'attractor'
                for family in dict.fromkeys(group['families']):
                    ids=[i for i,f in enumerate(group['families']) if f==family];old=torch.tensor(prior_records[panel,role,family,'swap5']['target'],device='cuda',dtype=torch.float64)
                    prior_replays.append(float((actual[ids]-old).abs().max()))
        groups.append(group)
    records,numerical=score(groups,binding['budgets'])
    instrument=counts=={k:PLAN[k] for k in counts} and max(native_replays)<=1e-4 and max(prior_replays)<=1e-4 and max(fold_replays)<=1e-10 and max(derivative_replays)<=1e-8 and all(max(c.values())<=1e-8 for c in checks) and numerical<=.01
    joint=[r for r in records if r['mode']=='state_program' and r['kind']=='joint'];increment=[r for r in records if r['mode']=='state_program' and r['kind']!='joint'];assert len(joint)==64 and len(increment)==128
    passes=lambda rows:all(r['number_error']<=.1 and r['modal_error']<=.05 for r in rows)
    ignored=[r for r in records if r['mode']=='ignore_attractor_oracle' and r['kind']=='attractor_increment']
    verdicts=[instrument,instrument and passes(joint),instrument and passes(increment),instrument and not passes(ignored)]
    result=dict(plan=PLAN,counts=counts,predictions={k:bool(v) for k,v in zip(PREDICTIONS,verdicts)},binding_sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest(),max_native_reference_replay=max(native_replays),max_prior_replay=max(prior_replays),max_exact_readout_replay=max(fold_replays),max_derivative_replay=max(derivative_replays),max_budget_numerical_error=numerical,jet_checks=checks,score_control=control(),groups=groups,records=records,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps({k:result[k] for k in ['predictions','counts','max_native_reference_replay','max_prior_replay','max_exact_readout_replay','max_derivative_replay','max_budget_numerical_error','seconds']}))


if __name__=='__main__':main()
