#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_half_quadratic pred_c_full_cubic pred_d_cubic_improvement
"""Opened-panel radius and directional cubic diagnostic; no full third-order tensor."""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'source_ood_v2_cubic_radius_result.json'
PREDICTIONS=dict(pred_a_instrument='counts,closure,native/prior1e-4,projectedG/H1e-8,thirdFD15%or1e-7abs',pred_b_half_quadratic='half-radius quadratic number10%/modal5%everycell',pred_c_full_cubic='full-radius cubic number10%/modal5%everycell',pred_d_cubic_improvement='full-radius cubic numbererror<=quadratic everycell')
PLAN=dict(prefix=12,double_suffix=208,native_suffix=88,gradient=96,hessian=96,third=96,predictions=PREDICTIONS)

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
    binding=json.loads((P/'SOURCE_OOD_V2_BINDING.json').read_text())
    for filename,digest in binding['sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest, filename
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for p in model.parameters():p.requires_grad_(False)
    assert all(not b.mlp.config.gated for b in model.transformer.h[11:])
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm())
    enc=tiktoken.get_encoding('gpt2');encoded=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(x)==1 for x in encoded);modal=torch.tensor([x[0] for x in encoded],device='cuda').reshape(3,2)
    counts=dict(prefix=0,double_suffix=0,native_suffix=0,gradient=0,hessian=0,third=0);records=[];replays=[];oldreplays=[];closures=[];deltas=[];derivative_replays=[];finite=[];contexts=[]
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
                    def evaluate(a):
                        counts['double_suffix']+=1
                        return source_observables(model,raw,x0,first,ds,pos,read,pairs,a)
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
                    arms={'unitB':torch.tensor(previous['amplitudes']['unitB'],device='cuda',dtype=torch.float64)};metadata={'unitB':('unitB',1.)};cubics={}
                    for base_arm in ['oldquad5','quad6','swap5']:
                        direction=torch.tensor(previous['amplitudes'][base_arm],device='cuda',dtype=torch.float64)
                        with torch.enable_grad():
                            t=torch.zeros(len(entries),device='cuda',dtype=torch.float64,requires_grad=True);value=evaluate(t[:,None]*direction);third=[]
                            for o in range(4):
                                d1=torch.autograd.grad(value[:,o].sum(),t,create_graph=True,retain_graph=True)[0];counts['gradient']+=1
                                d2=torch.autograd.grad(d1.sum(),t,create_graph=True,retain_graph=True)[0];counts['hessian']+=1
                                d3=torch.autograd.grad(d2.sum(),t,retain_graph=o!=3)[0];counts['third']+=1
                                derivative_replays.extend([float((d1-torch.einsum('bi,bi->b',g[:,o],direction)).abs().max()),float((d2-torch.einsum('bi,bij,bj->b',direction,h[:,o],direction)).abs().max())]);third.append(d3.detach())
                        cube=torch.stack(third,dim=1);cubics[base_arm]=cube
                        step=.05;fd=(evaluate(2*step*direction)-2*evaluate(step*direction)+2*evaluate(-step*direction)-evaluate(-2*step*direction))/(2*step**3)
                        errors=(fd-cube).norm(dim=0);scales=cube.norm(dim=0)
                        finite.extend(dict(absolute=float(e),relative=float(e/s.clamp_min(1e-20))) for e,s in zip(errors,scales))
                        for label,radius in [('full',1.),('half',.5),('quarter',.25)]:
                            arm=base_arm+'_'+label;arms[arm]=radius*direction;metadata[arm]=(base_arm,radius)
                    contexts.append(dict(panel=panel,role=role,template=template,third={k:v.tolist() for k,v in cubics.items()}))
                    for arm,av in arms.items():
                        actual=baseline-native(av);double=base-evaluate(av);replays.append(float((actual-double).abs().max()));prediction=-torch.einsum('boi,bi->bo',g,av)-.5*torch.einsum('bi,boij,bj->bo',av,h,av)
                        base_arm,radius=metadata[arm];cubic=prediction-radius**3*cubics[base_arm]/6 if base_arm!='unitB' else prediction
                        for family in dict.fromkeys(r['family'] for r in entries):
                            ids=[i for i,r in enumerate(entries) if r['family']==family];y=actual[ids];den=y[:,0].norm().clamp_min(1e-20)
                            if radius==1.:oldreplays.append(float((y-torch.tensor(frozen_records[(panel,role,family,base_arm)]['target'],device='cuda',dtype=torch.float64)).abs().max()))
                            records.append(dict(base_arm=base_arm,radius=radius,cubic_number_error=float((cubic[ids,0]-y[:,0]).norm()/den),cubic_modal_error=((cubic[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),quadratic_absolute_error=float((prediction[ids,0]-y[:,0]).norm()),cubic_absolute_error=float((cubic[ids,0]-y[:,0]).norm()),dataset=dataset,panel=panel,role=role,family=family,arm=arm,target=y.tolist(),number_error=float((prediction[ids,0]-y[:,0]).norm()/den),modal_error=((prediction[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),number_norm=float(den),native_capability=float((baseline[ids,0]>0).double().mean())))
    instrument=counts=={k:PLAN[k] for k in counts} and max(replays)<=1e-4 and max(oldreplays)<=1e-4 and max(derivative_replays)<=1e-8 and max(closures)<=1e-10 and all(d['absolute']<=1e-5 and d['relative']<=1e-6 for d in deltas) and all(f['relative']<=.15 or f['absolute']<=1e-7 for f in finite)
    half=[c for c in records if c['radius']==.5];full=[c for c in records if c['radius']==1. and c['base_arm']!='unitB'];assert len(half)==len(full)==48
    quadratic=instrument and all(c['number_error']<=.1 and max(c['modal_error'])<=.05 for c in half)
    cubic=instrument and all(c['cubic_number_error']<=.1 and max(c['cubic_modal_error'])<=.05 for c in full)
    improved=instrument and all(c['cubic_number_error']<=c['number_error'] for c in full)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(instrument),bool(quadratic),bool(cubic),bool(improved)])),source_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),max_native_reference_replay=max(replays),max_prior_replay=max(oldreplays),max_derivative_replay=max(derivative_replays),max_state_closure=max(closures),delta_closures=deltas,finite_checks=finite,contexts=contexts,records=records,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_native_reference_replay','max_prior_replay','max_derivative_replay','seconds']}))
if __name__=='__main__':main()
