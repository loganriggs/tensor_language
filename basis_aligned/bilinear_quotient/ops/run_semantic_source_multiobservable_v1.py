#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_prediction pred_c_selectivity
"""Four-output quadratic source circuit screen. No fits, opened96rows/2sites.
24prefix,224double+160native suffix forwards,64gradient+192Hessian-row reverse.
All native generators charged. Null: number prediction hides modal collateral.
"""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'semantic_source_multiobservable_v1_result.json'
ARMS=dict(single2=[1,0,0],single3=[0,1,0],single4=[0,0,1],pair23=[1,1,0],pair24=[1,0,1],pair34=[0,1,1],unit=[1,1,1],negative=[-1,-1,-1],mixed=[1,-1,1])
PREDICTIONS=dict(pred_a_instrument='native effect replay1e-4; FDgradient1%/Hessian15%; symmetry1e-4; exactcounts',pred_b_prediction='number own-effect error10%; modal error5%number-effect everycell/arm',pred_c_selectivity='native modal effects<=10%number-effect everycell/arm')
PLAN=dict(prefix=24,double_suffix=224,native_suffix=160,gradient=64,hessian=192,fits=0,amplitudes=ARMS,coefficients_per_context=36,predictions=PREDICTIONS)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    sys.path.insert(0,str(P));from native_source_observables import source_observables
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for p in model.parameters():p.requires_grad_(False)
    assert all(not b.mlp.config.gated for b in model.transformer.h[11:])
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm())
    enc=tiktoken.get_encoding('gpt2');encoded=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(x)==1 for x in encoded);modal=torch.tensor([x[0] for x in encoded],device='cuda').reshape(3,2)
    counts=dict(prefix=0,double_suffix=0,native_suffix=0,gradient=0,hessian=0);checks=[];records=[];replays=[];contexts=[]
    for panel,digest in [('opposite','1869db8a18d4c89c28fa81a6bb57a19497e6f8692831eb4391f5d23c84556d73'),('congruent','bfba23ab066beb14f5cd469edce137db7ec1e8ccc26bfc36b66f896426b17511')]:
        path=P/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;rows=json.loads(path.read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float();_,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                ds=torch.stack([(pe[i]-pb[i])[batch,pos].float() for i in [2,3,4]],dim=1)
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
                with torch.enable_grad():
                    a=torch.zeros(len(entries),3,device='cuda',dtype=torch.float64,requires_grad=True);base=evaluate(a);gs=[];hs=[]
                    for o in range(4):
                        g=torch.autograd.grad(base[:,o].sum(),a,create_graph=True,retain_graph=True)[0];counts['gradient']+=1
                        h=torch.stack([torch.autograd.grad(g[:,j].sum(),a,retain_graph=not(o==3 and j==2))[0] for j in range(3)],dim=1);counts['hessian']+=3;gs.append(g.detach());hs.append(h.detach())
                base=base.detach();g=torch.stack(gs,dim=1);h=torch.stack(hs,dim=1);baseline=native(torch.zeros_like(a));direction=torch.tensor([.7,-.4,.5],device='cuda',dtype=torch.float64).expand(len(a),3)
                check=dict(panel=panel,role=role,template=template,symmetry=float((h-h.transpose(-1,-2)).norm()/h.norm()),finite=[])
                gd=torch.einsum('boi,bi->bo',g,direction);hd=torch.einsum('bi,boij,bj->bo',direction,h,direction)
                for step in [.1,.2]:
                    plus=evaluate(step*direction);minus=evaluate(-step*direction);fd=(plus-minus)/(2*step);fdd=(plus+minus-2*base)/(step**2)
                    check['finite'].append(dict(step=step,gradient_relative=((fd-gd).norm(dim=0)/gd.norm(dim=0).clamp_min(1e-20)).tolist(),hessian_relative=((fdd-hd).norm(dim=0)/hd.norm(dim=0).clamp_min(1e-20)).tolist()))
                checks.append(check);contexts.append(dict(panel=panel,role=role,template=template,gradient=g.tolist(),hessian=h.tolist()))
                for arm,amplitude in ARMS.items():
                    av=torch.tensor(amplitude,device='cuda',dtype=torch.float64).expand(len(a),3);actual=baseline-native(av);reference=base-evaluate(av);replays.append(float((actual-reference).abs().max()));linear=-torch.einsum('boi,bi->bo',g,av);quadratic=linear-.5*torch.einsum('bi,boij,bj->bo',av,h,av)
                    for family in dict.fromkeys(r['family'] for r in entries):
                        ids=[i for i,r in enumerate(entries) if r['family']==family];y=actual[ids];den=y[:,0].norm().clamp_min(1e-20)
                        records.append(dict(panel=panel,role=role,family=family,arm=arm,target=y.tolist(),quadratic=quadratic[ids].tolist(),linear=linear[ids].tolist(),number_error=float((quadratic[ids,0]-y[:,0]).norm()/den),modal_error=((quadratic[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),linear_modal_error=((linear[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),native_modal_ratio=(y[:,1:].norm(dim=0)/den).tolist(),number_norm=float(den),native_capability=float((baseline[ids,0]>0).double().mean())))
    a=max(replays)<=1e-4 and counts=={k:PLAN[k] for k in counts} and all(c['symmetry']<=1e-4 and all(max(f['gradient_relative'])<=.01 and max(f['hessian_relative'])<=.15 for f in c['finite']) for c in checks)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and all(c['number_error']<=.1 and max(c['modal_error'])<=.05 for c in records)),bool(a and all(max(c['native_modal_ratio'])<=.1 for c in records))])),max_native_effect_replay=max(replays),checks=checks,records=records,contexts=contexts,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_native_effect_replay','seconds']}))
if __name__=='__main__':main()
