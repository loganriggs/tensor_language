#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_full_fields pred_c_field_improvement pred_d_cubic_baseline
"""Quadratic upstream fields with explicit shared final RMS and token softcaps."""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'source_ood_v2_readout_fields_result.json'
PREDICTIONS=dict(pred_a_instrument='counts,closure,native/prior1e-4,readoutbase1e-10,projectedG/H1e-8',pred_b_full_fields='full-radius field program number10%/modal5%everycell',pred_c_field_improvement='full-radius field numbererror<=quadratic everycell',pred_d_cubic_baseline='worst fullfield numbererror<=worst priorcubic')
PLAN=dict(prefix=12,double_suffix=112,native_suffix=88,gradient=216,hessian=216,readout_gradient=96,readout_hessian=96,coefficients_per_ray=27,predictions=PREDICTIONS)

def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    from scipy.optimize import linprog
    import tiktoken
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    sys.path.insert(0,str(P));from native_source_observables import source_observables
    from six_source_complement import closed_ports
    from quadratic_budgeted_direction import choose as optimize
    from budgeted_modal_direction import choose as linear_choose
    from final_readout_field_program import pack_fields,evaluate_fields,evaluate_jet
    binding=json.loads((P/'SOURCE_OOD_V2_BINDING.json').read_text())
    for filename,digest in binding['sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest, filename
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for p in model.parameters():p.requires_grad_(False)
    assert all(not b.mlp.config.gated for b in model.transformer.h[11:])
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm())
    enc=tiktoken.get_encoding('gpt2');encoded=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(x)==1 for x in encoded);modal=torch.tensor([x[0] for x in encoded],device='cuda').reshape(3,2)
    counts=dict(prefix=0,double_suffix=0,native_suffix=0,gradient=0,hessian=0,readout_gradient=0,readout_hessian=0);records=[];replays=[];oldreplays=[];closures=[];deltas=[];derivative_replays=[];finite=[];contexts=[]
    frozen_path=A/'source_ood_v2_result.json';frozen=json.loads(frozen_path.read_text());assert frozen['predictions']['pred_a_instrument'];frozen_contexts={(c['panel'],c['role'],c['template']):c for c in frozen['contexts']};frozen_records={(c['panel'],c['role'],c['family'],c['arm']):c for c in frozen['records']}
    for dataset,pattern in [('prospective_v2','SOURCE_OOD_V2')]:
        for panel in ['opposite','congruent']:
            path=P/f'{pattern}_{panel.upper()}_ROWS.json';rows=json.loads(path.read_text())
            for template in dict.fromkeys(r['template'] for r in rows):
                entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
                initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1;pb=closed_ports(raw,pb);closures.append(float((sum(pb)-raw.double()).abs().max()))
                for role,key in [('subject','subject_position'),('attractor','control_position')]:
                    pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float();raw_removed,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1;pe=closed_ports(raw_removed,pe);closures.append(float((sum(pe)-raw_removed.double()).abs().max()))
                    old_ds=torch.stack([(pe[i]-pb[i])[batch,pos].float().double() for i in range(5)],dim=1)
                    exact_delta=(raw_removed.double()-raw.double())[batch,pos]
                    ds=torch.cat([old_ds,(exact_delta-old_ds.sum(1))[:,None,:]],dim=1)
                    def evaluate(a,capture=None):
                        counts['double_suffix']+=1
                        return source_observables(model,raw,x0,first,ds,pos,read,pairs,a,capture=capture)
                    def native(a):
                        counts['native_suffix']+=1;x=raw.clone();x[batch,pos]=(raw[batch,pos].double()+torch.einsum('bi,bid->bd',a.double(),ds)).float()
                        for l in range(11,18):
                            b=model.transformer.h[l]
                            if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
                            at,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+at;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
                        logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30);z=logits.gather(1,pairs.reshape(len(x),8)).reshape(len(x),4,2)
                        return (z[:,:,0]-z[:,:,1]).double()
                    difference=(raw_removed.double()-raw.double())[batch,pos];delta_error=ds.double().sum(1)-difference;deltas.append(dict(absolute=float(delta_error.abs().max()),relative=float(delta_error.norm()/difference.norm().clamp_min(1e-30))))
                    zero=torch.zeros(len(entries),6,device='cuda',dtype=torch.float64)
                    previous=frozen_contexts[(panel,role,template)];g=torch.tensor(previous['gradient'],device='cuda',dtype=torch.float64);h=torch.tensor(previous['hessian'],device='cuda',dtype=torch.float64);base=evaluate(zero);baseline=native(zero)
                    arms={'unitB':torch.tensor(previous['amplitudes']['unitB'],device='cuda',dtype=torch.float64)};metadata={'unitB':('unitB',1.)};jets={}
                    for base_arm in ['oldquad5','quad6','swap5']:
                        direction=torch.tensor(previous['amplitudes'][base_arm],device='cuda',dtype=torch.float64)
                        with torch.enable_grad():
                            t=torch.zeros(len(entries),device='cuda',dtype=torch.float64,requires_grad=True);capture={};value=evaluate(t[:,None]*direction,capture);state=capture['mlp_outputs'][17][batch,read];fields=pack_fields(state,model.lm_head.weight[pairs].double());finite.append(float((evaluate_fields(fields)-value).abs().max()));firsts=[];seconds=[]
                            for o in range(9):
                                d1=torch.autograd.grad(fields[:,o].sum(),t,create_graph=True,retain_graph=True)[0];counts['gradient']+=1
                                d2=torch.autograd.grad(d1.sum(),t,retain_graph=o!=8)[0];counts['hessian']+=1
                                firsts.append(d1.detach());seconds.append(.5*d2.detach())
                            jet=torch.stack([fields.detach(),torch.stack(firsts,dim=1),torch.stack(seconds,dim=1)],dim=1);jets[base_arm]=jet
                            u=torch.zeros(len(entries),device='cuda',dtype=torch.float64,requires_grad=True);reconstructed=evaluate_jet(jet,u[:,None])
                            for o in range(4):
                                d1=torch.autograd.grad(reconstructed[:,o].sum(),u,create_graph=True,retain_graph=True)[0];counts['readout_gradient']+=1
                                d2=torch.autograd.grad(d1.sum(),u,retain_graph=o!=3)[0];counts['readout_hessian']+=1
                                derivative_replays.extend([float((d1.detach()-torch.einsum('bi,bi->b',g[:,o],direction)).abs().max()),float((d2.detach()-torch.einsum('bi,bij,bj->b',direction,h[:,o],direction)).abs().max())])
                        for label,radius in [('full',1.),('half',.5),('quarter',.25)]:
                            arm=base_arm+'_'+label;arms[arm]=radius*direction;metadata[arm]=(base_arm,radius)
                    contexts.append(dict(panel=panel,role=role,template=template,field_jets={k:v.tolist() for k,v in jets.items()}))
                    for arm,av in arms.items():
                        actual=baseline-native(av);double=base-evaluate(av);replays.append(float((actual-double).abs().max()));prediction=-torch.einsum('boi,bi->bo',g,av)-.5*torch.einsum('bi,boij,bj->bo',av,h,av)
                        base_arm,radius=metadata[arm]
                        if base_arm!='unitB':
                            candidate_fields=jets[base_arm][:,0]+radius*jets[base_arm][:,1]+radius**2*jets[base_arm][:,2];positive=bool((candidate_fields[:,-1]>0).all());field_prediction=base-evaluate_fields(candidate_fields) if positive else None
                        else:positive=True;field_prediction=prediction
                        for family in dict.fromkeys(r['family'] for r in entries):
                            ids=[i for i,r in enumerate(entries) if r['family']==family];y=actual[ids];den=y[:,0].norm().clamp_min(1e-20)
                            if radius==1.:oldreplays.append(float((y-torch.tensor(frozen_records[(panel,role,family,base_arm)]['target'],device='cuda',dtype=torch.float64)).abs().max()))
                            records.append(dict(base_arm=base_arm,radius=radius,norm2_positive=positive,field_number_error=float((field_prediction[ids,0]-y[:,0]).norm()/den) if positive else None,field_modal_error=((field_prediction[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist() if positive else None,quadratic_absolute_error=float((prediction[ids,0]-y[:,0]).norm()),dataset=dataset,panel=panel,role=role,family=family,arm=arm,target=y.tolist(),number_error=float((prediction[ids,0]-y[:,0]).norm()/den),modal_error=((prediction[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),number_norm=float(den),native_capability=float((baseline[ids,0]>0).double().mean())))
    instrument=counts=={k:PLAN[k] for k in counts} and max(replays)<=1e-4 and max(oldreplays)<=1e-4 and max(derivative_replays)<=1e-8 and max(closures)<=1e-10 and all(d['absolute']<=1e-5 and d['relative']<=1e-6 for d in deltas) and max(finite)<=1e-10
    full=[c for c in records if c['radius']==1. and c['base_arm']!='unitB'];assert len(full)==48
    valid=all(c['norm2_positive'] for c in full)
    field_pass=instrument and valid and all(c['field_number_error']<=.1 and max(c['field_modal_error'])<=.05 for c in full)
    improved=instrument and valid and all(c['field_number_error']<=c['number_error'] for c in full)
    cubic_source=A/'source_ood_v2_cubic_radius_result.json';cubic=json.loads(cubic_source.read_text());assert cubic['predictions']['pred_a_instrument'];cubic_worst=max(c['cubic_number_error'] for c in cubic['records'] if c['radius']==1. and c['base_arm']!='unitB')
    matched=instrument and valid and max(c['field_number_error'] for c in full)<=cubic_worst
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(instrument),bool(field_pass),bool(improved),bool(matched)])),source_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),cubic_source_sha256=hashlib.sha256(cubic_source.read_bytes()).hexdigest(),max_native_reference_replay=max(replays),max_prior_replay=max(oldreplays),max_derivative_replay=max(derivative_replays),max_readout_replay=max(finite),max_state_closure=max(closures),delta_closures=deltas,contexts=contexts,records=records,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_native_reference_replay','max_prior_replay','max_derivative_replay','max_readout_replay','seconds']}))
if __name__=='__main__':main()
