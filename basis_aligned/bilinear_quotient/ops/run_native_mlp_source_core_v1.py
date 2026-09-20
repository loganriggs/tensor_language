#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_ports pred_b_local_derivatives pred_c_local_curvature_prediction
"""Native ML P11 exact rational source core, not a whole-suffix replacement.
24prefix,16fullsuffix,64reader reverse,80attention JVP;
80localMLP forward,64localgradient+320Hessian-row reverse.0fits.
Null: local MLP11 curvature alone misses other native computation paths.
"""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'native_mlp_source_core_v1_result.json';EXPORT=A/'native_mlp_source_core_v1.pt'
PREDICTIONS=dict(pred_a_native_ports='old fullgradient replay1e-8 andlocal rational output replay1e-8; counts',pred_b_local_derivatives='analytic local gradient/Hessian versusdenseautograd1e-8',pred_c_local_curvature_prediction='fullgradient plusonlyMLP11curvature predicts number10%/modal5% everycell/arm')
PLAN=dict(prefix=24,fullsuffix=16,reader_reverse=64,attention_jvp=80,local_forward=80,local_gradient=64,local_hessian=320,fits=0,predictions=PREDICTIONS)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
    import torch
    import numpy as np
    import torch.nn.functional as F
    import tiktoken
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write,guard_torch_save
    sys.path.insert(0,str(P));from native_source_observables import source_observables,first_source_mlp_input,mlp_write64
    from normalized_mlp_source_core import compile_core,evaluate,derivatives_at_zero
    assert not OUT.exists() and not EXPORT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm());eps=torch.finfo(torch.float32).eps
    enc=tiktoken.get_encoding('gpt2');words=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(w)==1 for w in words);modal=torch.tensor([w[0] for w in words],device='cuda').reshape(3,2)
    old=json.loads((A/'five_source_modal_null_v1_result.json').read_text());oldctx={(c['panel'],c['role'],c['template']):c for c in old['contexts']}
    counts={k:0 for k in ['prefix','fullsuffix','reader_reverse','attention_jvp','local_forward','local_gradient','local_hessian']};checks=[];records=[];prices=[];exported=False
    for panel,digest in [('opposite','1869db8a18d4c89c28fa81a6bb57a19497e6f8692831eb4391f5d23c84556d73'),('congruent','bfba23ab066beb14f5cd469edce137db7ec1e8ccc26bfc36b66f896426b17511')]:
        path=P/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;rows=json.loads(path.read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float();_,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                ds=torch.stack([(pe[i]-pb[i])[batch,pos].float() for i in range(5)],dim=1);previous=oldctx[(panel,role,template)];capture={}
                with torch.enable_grad():
                    a=torch.zeros(len(entries),5,device='cuda',dtype=torch.float64,requires_grad=True);full=source_observables(model,raw,x0,first,ds,pos,read,pairs,a,capture);counts['fullsuffix']+=1
                    qs=[];gs=[]
                    for o in range(4):
                        q,g=torch.autograd.grad(full[:,o].sum(),(capture['mlp_outputs'][11],a),retain_graph=o<3);qs.append(q.detach());gs.append(g.detach());counts['reader_reverse']+=1
                h0=capture['mlp_inputs'][11].detach();readers=torch.stack(qs,dim=2);fullg=torch.stack(gs,dim=1);oldg=torch.tensor(previous['gradient'],device='cuda',dtype=torch.float64);oldh=torch.tensor(previous['hessian'],device='cuda',dtype=torch.float64)
                K=[];zero=torch.zeros_like(a);hcheck=[]
                for j in range(5):
                    direction=torch.zeros_like(a);direction[:,j]=1
                    hbase,k=torch.autograd.functional.jvp(lambda z:first_source_mlp_input(model,raw,first,ds,pos,z),zero,direction,create_graph=False);counts['attention_jvp']+=1;K.append(k);hcheck.append(float((hbase-h0).abs().max()))
                K=torch.stack(K,dim=-1);b=model.transformer.h[11];m=b.mlp
                core=compile_core(m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double(),m.Down_bias.double(),h0,K,readers,eps);localg,localh=derivatives_at_zero(core);localg=localg.sum(1);localh=localh.sum(1)
                def dense(z):
                    counts['local_forward']+=1;state=h0+torch.einsum('btdp,bp->btd',K,z);return torch.einsum('btod,btd->bo',readers,mlp_write64(b,state))
                candidate=torch.tensor(previous['candidate_amplitudes'],device='cuda',dtype=torch.float64);unitB=torch.tensor([0,0,1,1,1],device='cuda',dtype=torch.float64).expand_as(zero)
                amplitudes=[zero,unitB,candidate,torch.ones_like(zero)];refs=[];localchecks=[]
                for z in amplitudes:
                    true=dense(z);pred=evaluate(core,z[:,None,:]).sum(1);localchecks.append(float((pred-true).abs().max()));refs.append(true)
                with torch.enable_grad():
                    az=zero.clone().requires_grad_();output=dense(az);ag=[];ah=[]
                    for o in range(4):
                        g=torch.autograd.grad(output[:,o].sum(),az,create_graph=True,retain_graph=True)[0];ag.append(g.detach());counts['local_gradient']+=1
                        ah.append(torch.stack([torch.autograd.grad(g[:,j].sum(),az,retain_graph=not(o==3 and j==4))[0] for j in range(5)],dim=1));counts['local_hessian']+=5
                check=dict(panel=panel,role=role,template=template,full_gradient_replay=float((fullg-oldg).abs().max()),source_input_replay=max(hcheck),local_output_replay=max(localchecks),local_gradient_replay=float((localg-torch.stack(ag,dim=1)).abs().max()),local_hessian_replay=float((localh-torch.stack(ah,dim=1)).abs().max()),local_vs_full_hessian_relative=float((localh-oldh).norm()/oldh.norm().clamp_min(1e-20)))
                checks.append(check);prices.append(dict(panel=panel,role=role,template=template,runtime_values=sum(v.numel() for v in core.values()),source_sensitivity_values=K.numel(),reader_values=readers.numel(),background_values=h0.numel()))
                if not exported:
                    package=dict(core={k:v[0].cpu() for k,v in core.items()},amplitudes=torch.stack([z[0].cpu() for z in amplitudes]),references=torch.stack([v[0].cpu() for v in refs]),scope='Fixed localMLP11 affine source interface and fixed downstream readers, not fullsuffix.',text=entries[0]['text'],role=role)
                    guard_torch_save(package,EXPORT);exported=True
                for arm,z in [('unitB',unitB),('null_full',candidate),('null_half',candidate*.5)]:
                    prediction=-torch.einsum('bop,bp->bo',fullg,z)-.5*torch.einsum('bp,bopq,bq->bo',z,localh,z)
                    for family in dict.fromkeys(r['family'] for r in entries):
                        ids=[i for i,r in enumerate(entries) if r['family']==family];original=next(c for c in old['records'] if c['panel']==panel and c['role']==role and c['family']==family and c['arm']==arm);target=torch.tensor(original['target'],device='cuda',dtype=torch.float64);den=target[:,0].norm().clamp_min(1e-20);pred=prediction[ids]
                        records.append(dict(panel=panel,role=role,family=family,arm=arm,target=target.tolist(),prediction=pred.tolist(),number_error=float((pred[:,0]-target[:,0]).norm()/den),modal_error=((pred[:,1:]-target[:,1:]).norm(dim=0)/den).tolist()))
    a=counts=={k:PLAN[k] for k in counts} and all(max(c[k] for k in ['full_gradient_replay','source_input_replay','local_output_replay'])<=1e-8 for c in checks)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and all(max(c['local_gradient_replay'],c['local_hessian_replay'])<=1e-8 for c in checks)),bool(a and all(c['number_error']<=.1 and max(c['modal_error'])<=.05 for c in records))])),checks=checks,records=records,prices=prices,export_file=str(EXPORT),export_bytes=EXPORT.stat().st_size,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','export_bytes','seconds']}))
if __name__=='__main__':main()
