#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selectivity pred_c_prediction
"""Matched-strength native selective intervention control; frozen derivatives."""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'six_source_complement_v1_result.json'
PREDICTIONS=dict(pred_a_instrument='fiveport g/H1e-8,stateclosure1e-10,delta1e-5abs/1e-6rel,native/reference1e-4,FD1%/15%,counts',pred_b_selectivity='quad6 native80%alignedunitB retention/10%modal all48cells',pred_c_prediction='oldquad5andquad6 number10%/modal5%numberbudget everycell')
PLAN=dict(prefix=36,double_suffix=216,native_suffix=120,gradient=96,hessian=576,source_width=6,coefficients_per_context=108,predictions=PREDICTIONS)

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
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for p in model.parameters():p.requires_grad_(False)
    assert all(not b.mlp.config.gated for b in model.transformer.h[11:])
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm())
    enc=tiktoken.get_encoding('gpt2');encoded=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(x)==1 for x in encoded);modal=torch.tensor([x[0] for x in encoded],device='cuda').reshape(3,2)
    counts=dict(prefix=0,double_suffix=0,native_suffix=0,gradient=0,hessian=0);records=[];replays=[];oldreplays=[];closures=[];deltas=[];derivative_replays=[];finite=[];contexts=[]
    quad=json.loads((P/'QUADRATIC_BUDGETED_DIRECTIONS_V1.json').read_text());quad_lookup={(c['dataset'],c['panel'],c['role'],c['template']):c for c in quad['contexts']}
    for dataset,source,pattern in [('opened','five_source_modal_null_v1_result.json','SEMANTIC_PORT_FRESH'),('ood_opened','source_ood_v1_result.json','SOURCE_OOD_V1')]:
        oldpath=A/source;old=json.loads(oldpath.read_text());oldcontexts={(c['panel'],c['role'],c['template']):c for c in old['contexts']};oldrecords={(c['panel'],c['role'],c['family'],c['arm']):c for c in old['records']}
        assert all(c['source_sha256']==hashlib.sha256(oldpath.read_bytes()).hexdigest() for c in quad['contexts'] if c['dataset']==dataset)
        for panel in ['opposite','congruent']:
            path=P/f'{pattern}_{panel.upper()}_ROWS.json';rows=json.loads(path.read_text())
            for template in dict.fromkeys(r['template'] for r in rows):
                entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
                initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1;pb=closed_ports(raw,pb);closures.append(float((sum(pb)-raw.double()).abs().max()))
                for role,key in [('subject','subject_position'),('attractor','control_position')]:
                    pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float();raw_removed,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1;pe=closed_ports(raw_removed,pe);closures.append(float((sum(pe)-raw_removed.double()).abs().max()))
                    ds=torch.stack([(pe[i]-pb[i])[batch,pos].float() for i in range(6)],dim=1)
                    def evaluate(a):
                        counts['double_suffix']+=1
                        return source_observables(model,raw,x0,first,ds,pos,read,pairs,a)
                    def native(a):
                        counts['native_suffix']+=1;x=raw.clone();x[batch,pos]+=torch.einsum('bi,bid->bd',a.float(),ds)
                        for l in range(11,18):
                            b=model.transformer.h[l]
                            if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
                            at,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+at;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
                        logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30);z=logits.gather(1,pairs.reshape(len(x),8)).reshape(len(x),4,2)
                        return (z[:,:,0]-z[:,:,1]).double()
                    difference=(raw_removed.double()-raw.double())[batch,pos];delta_error=ds.double().sum(1)-difference;deltas.append(dict(absolute=float(delta_error.abs().max()),relative=float(delta_error.norm()/difference.norm().clamp_min(1e-30))))
                    previous=oldcontexts[(panel,role,template)];zero=torch.zeros(len(entries),6,device='cuda',dtype=torch.float64)
                    with torch.enable_grad():
                        a=zero.clone().requires_grad_();base=evaluate(a);gs=[];hs=[]
                        for o in range(4):
                            gi=torch.autograd.grad(base[:,o].sum(),a,create_graph=True,retain_graph=True)[0];counts['gradient']+=1
                            hi=torch.stack([torch.autograd.grad(gi[:,j].sum(),a,retain_graph=not(o==3 and j==5))[0] for j in range(6)],dim=1);counts['hessian']+=6;gs.append(gi.detach());hs.append(hi.detach())
                    base=base.detach();g=torch.stack(gs,dim=1);h=torch.stack(hs,dim=1);baseline=native(zero)
                    derivative_replays.extend([float((g[:,:,:5]-torch.tensor(previous['gradient'],device='cuda')).abs().max()),float((h[:,:,:5,:5]-torch.tensor(previous['hessian'],device='cuda')).abs().max())])
                    direction=torch.tensor([.3,-.2,.7,-.4,.5,.6],device='cuda',dtype=torch.float64).expand_as(zero);gd=torch.einsum('boi,bi->bo',g,direction);hd=torch.einsum('bi,boij,bj->bo',direction,h,direction)
                    for step in [.1,.2]:
                        plus=evaluate(step*direction);minus=evaluate(-step*direction);fd=(plus-minus)/(2*step);fdd=(plus+minus-2*base)/step**2
                        finite.append(dict(gradient=float(((fd-gd).norm(dim=0)/gd.norm(dim=0).clamp_min(1e-20)).max()),hessian=float(((fdd-hd).norm(dim=0)/hd.norm(dim=0).clamp_min(1e-20)).max())))
                    reference=np.array([0.,0.,1.,1.,1.,0.]);old_null=np.pad(np.array(previous['candidate_amplitudes']),((0,0),(0,1)))
                    choices=[optimize(gi,hi,ni,reference=reference) for gi,hi,ni in zip(g.cpu().numpy(),h.cpu().numpy(),old_null)];new=torch.tensor(np.array([v for v,_ in choices]),device='cuda',dtype=torch.float64)
                    old5=np.pad(np.array(quad_lookup[(dataset,panel,role,template)]['amplitudes']),((0,0),(0,1)))
                    contexts.append(dict(dataset=dataset,panel=panel,role=role,template=template,gradient=g.tolist(),hessian=h.tolist(),amplitudes=new.tolist(),optimizer_checks=[v for _,v in choices]))
                    arms=dict(unitB=torch.tensor(reference,device='cuda').expand_as(zero),oldquad5=torch.tensor(old5,device='cuda'),quad6=new,all6=torch.ones_like(zero))
                    for arm,av in arms.items():
                        actual=baseline-native(av);double=base-evaluate(av);replays.append(float((actual-double).abs().max()));prediction=-torch.einsum('boi,bi->bo',g,av)-.5*torch.einsum('bi,boij,bj->bo',av,h,av)
                        for family in dict.fromkeys(r['family'] for r in entries):
                            ids=[i for i,r in enumerate(entries) if r['family']==family];y=actual[ids];den=y[:,0].norm().clamp_min(1e-20)
                            if arm=='unitB':oldreplays.append(float((y-torch.tensor(oldrecords[(panel,role,family,'unitB')]['target'],device='cuda')).abs().max()))
                            records.append(dict(dataset=dataset,panel=panel,role=role,family=family,arm=arm,target=y.tolist(),number_error=float((prediction[ids,0]-y[:,0]).norm()/den),modal_error=((prediction[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),number_norm=float(den)))
    instrument=counts=={k:PLAN[k] for k in counts} and max(replays)<=1e-4 and max(oldreplays)<=1e-4 and max(derivative_replays)<=1e-8 and max(closures)<=1e-10 and all(d['absolute']<=1e-5 and d['relative']<=1e-6 for d in deltas) and all(f['gradient']<=.01 and f['hessian']<=.15 for f in finite)
    lookup={(r['dataset'],r['panel'],r['role'],r['family'],r['arm']):r for r in records};comparisons=[]
    for row in records:
        if row['arm'] not in ['oldquad5','quad6']:continue
        y=np.array(row['target']);b=np.array(lookup[(row['dataset'],row['panel'],row['role'],row['family'],'unitB')]['target']);den=max(np.linalg.norm(y[:,0]),1e-30)
        comparisons.append(dict(dataset=row['dataset'],panel=row['panel'],role=row['role'],family=row['family'],arm=row['arm'],retention=float(y[:,0]@b[:,0]/max(b[:,0]@b[:,0],1e-30)),modal_ratio=float(max(np.linalg.norm(y[:,1:],axis=0))/den),number_error=row['number_error'],modal_error=max(row['modal_error'])))
    primary=[c for c in comparisons if c['arm']=='quad6'];assert len(primary)==48
    selective=instrument and all(c['retention']>=.8 and c['modal_ratio']<=.1 for c in primary);predictive=instrument and all(c['number_error']<=.1 and c['modal_error']<=.05 for c in comparisons)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(instrument),bool(selective),bool(predictive)])),max_native_reference_replay=max(replays),max_prior_replay=max(oldreplays),max_old_derivative_replay=max(derivative_replays),max_state_closure=max(closures),delta_closures=deltas,finite_checks=finite,contexts=contexts,comparisons=comparisons,records=records,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_native_reference_replay','max_old_derivative_replay','max_state_closure','seconds']}))
if __name__=='__main__':main()
