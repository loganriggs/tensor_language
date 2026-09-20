#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_chain_closure pred_b_mlp_only pred_c_mlp_readout
"""Exact source-Hessian chain decomposition, then test grouped curvature omissions.
All original native generators charged. Opened96rows/2sites,0fits.
Null: attention and readout curvature cannot be discarded despite exact MLP cores.
"""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'native_source_curvature_sum_v1_result.json';EXPORT=A/'native_source_curvature_sum_v1.pt'
PREDICTIONS=dict(pred_a_chain_closure='fullg/Hsum abs/relative1e-8; localMLPreplay1e-8; counts',pred_b_mlp_only='MLPcurvature+fullgradient predictsnumber10/modal5 everycell',pred_c_mlp_readout='MLP+readoutcurvature+fullgradient predictsnumber10/modal5 everycell')
PLAN=dict(prefix=24,full_forward=16,full_source_jvp=80,reader_reverse=64,mlp_compile=112,mlp_replay=224,local_attention_forward=112,local_readout_forward=16,local_gradient=512,local_hessian=2560,fits=0,predictions=PREDICTIONS)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
    import torch
    import numpy as np
    import torch.nn.functional as F
    import tiktoken
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write,guard_torch_save
    sys.path.insert(0,str(P));from native_source_observables import attention_write64,mlp_write64,norm64
    from source_differential_ports import capture_ports
    from normalized_mlp_source_core import compile_core,evaluate,derivatives_at_zero
    assert not OUT.exists() and not EXPORT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm());eps=torch.finfo(torch.float32).eps
    enc=tiktoken.get_encoding('gpt2');words=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(w)==1 for w in words);modal=torch.tensor([w[0] for w in words],device='cuda').reshape(3,2)
    old=json.loads((A/'five_source_modal_null_v1_result.json').read_text());oldctx={(c['panel'],c['role'],c['template']):c for c in old['contexts']}
    counts={k:0 for k in ['prefix','full_forward','full_source_jvp','reader_reverse','mlp_compile','mlp_replay','local_attention_forward','local_readout_forward','local_gradient','local_hessian']};checks=[];records=[];prices=[];export_cases=[]
    for panel,digest in [('opposite','1869db8a18d4c89c28fa81a6bb57a19497e6f8692831eb4391f5d23c84556d73'),('congruent','bfba23ab066beb14f5cd469edce137db7ec1e8ccc26bfc36b66f896426b17511')]:
        path=P/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;rows=json.loads(path.read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float();_,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                ds=torch.stack([(pe[i]-pb[i])[batch,pos].float() for i in range(5)],dim=1);previous=oldctx[(panel,role,template)];
                ports=capture_ports(model,raw,x0,first,ds,pos,read,pairs)
                for k,v in ports['calls'].items():counts[k]+=v
                zero=torch.zeros(len(entries),5,device='cuda',dtype=torch.float64);terms={};local_replay=[]
                def local_hessian(fn,kind):
                    with torch.enable_grad():
                        z=zero.clone().requires_grad_();y=fn(z);counts[kind]+=1;hs=[]
                        for o in range(4):
                            g=torch.autograd.grad(y[:,o].sum(),z,create_graph=True,retain_graph=True)[0];counts['local_gradient']+=1
                            hs.append(torch.stack([torch.autograd.grad(g[:,j].sum(),z,retain_graph=not(o==3 and j==4))[0] for j in range(5)],dim=1));counts['local_hessian']+=5
                    return torch.stack(hs,dim=1).detach()
                for index,l in enumerate(ports['layers']):
                    b=model.transformer.h[l];m=b.mlp;h0=ports['mlp_inputs'][index];K=ports['mlp_input_directions'][index];q=ports['mlp_readers'][index]
                    core=compile_core(m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double(),m.Down_bias.double(),h0,K,q,eps);counts['mlp_compile']+=1
                    terms[f'mlp{l}']=derivatives_at_zero(core)[1].sum(1)
                    for amplitude in [zero,torch.ones_like(zero)]:
                        y=mlp_write64(b,h0+torch.einsum('btdp,bp->btd',K,amplitude));dense=torch.einsum('btod,btd->bo',q,y);counts['mlp_replay']+=1
                        local_replay.append(float((evaluate(core,amplitude[:,None,:]).sum(1)-dense).abs().max()))
                    if l==11:
                        before=raw.double();beforeK=torch.zeros(*raw.shape,5,device='cuda',dtype=torch.float64);beforeK[batch,pos]=ds.double().transpose(1,2)
                    else:
                        before=b.lambdas[0].double()*ports['mlp_outputs'][index-1]+b.lambdas[1].double()*x0.double();beforeK=b.lambdas[0].double()*ports['mlp_output_directions'][index-1]
                    aq=ports['attention_readers'][index]
                    def attention_local(amplitude):
                        x=before+torch.einsum('btdp,bp->btd',beforeK,amplitude)
                        return torch.einsum('btod,btd->bo',aq,attention_write64(b,x,first))
                    terms[f'attention{l}']=local_hessian(attention_local,'local_attention_forward')
                final=ports['mlp_outputs'][-1][batch,read];finalK=ports['mlp_output_directions'][-1][batch,read]
                def readout_local(amplitude):
                    x=final+torch.einsum('bdp,bp->bd',finalK,amplitude);logits=30*torch.tanh((norm64(x)[:,None,None,:]*model.lm_head.weight[pairs].double()).sum(-1)/30)
                    return logits[:,:,0]-logits[:,:,1]
                terms['readout']=local_hessian(readout_local,'local_readout_forward')
                total=sum(terms.values());oldH=torch.tensor(previous['hessian'],device='cuda',dtype=torch.float64);oldG=torch.tensor(previous['gradient'],device='cuda',dtype=torch.float64);G=ports['full_gradient']
                checks.append(dict(panel=panel,role=role,template=template,gradient_replay=float((G-oldG).abs().max()),source_replay=ports['max_source_replay'],hessian_closure_absolute=float((total-oldH).abs().max()),hessian_closure_relative=float((total-oldH).norm()/oldH.norm().clamp_min(1e-20)),local_core_replay=max(local_replay)))
                mlp=sum(v for k,v in terms.items() if k.startswith('mlp'));attn=sum(v for k,v in terms.items() if k.startswith('attention'))
                grouped=dict(full=total,mlp_only=mlp,mlp_readout=mlp+terms['readout'],attention_readout=attn+terms['readout'])
                candidate=torch.tensor(previous['candidate_amplitudes'],device='cuda',dtype=torch.float64);unitB=torch.tensor([0,0,1,1,1],device='cuda',dtype=torch.float64).expand_as(zero)
                export_cases.append(dict(panel=panel,role=role,template=template,gradient=G.cpu(),hessian_reference=oldH.cpu(),terms={k:v.cpu() for k,v in terms.items()},amplitudes=dict(unitB=unitB.cpu(),null_full=candidate.cpu(),null_half=(candidate*.5).cpu()),families=[r['family'] for r in entries]))
                prices.append(dict(panel=panel,role=role,template=template,transport_values=sum(x.numel() for k in ['mlp_input_directions','mlp_output_directions','mlp_readers','attention_readers'] for x in ports[k]),component_hessian_values=sum(v.numel() for v in terms.values())))
                for mode,H in grouped.items():
                    for arm,z in [('unitB',unitB),('null_full',candidate),('null_half',candidate*.5)]:
                        prediction=-torch.einsum('bop,bp->bo',G,z)-.5*torch.einsum('bp,bopq,bq->bo',z,H,z)
                        for family in dict.fromkeys(r['family'] for r in entries):
                            ids=[i for i,r in enumerate(entries) if r['family']==family];original=next(c for c in old['records'] if c['panel']==panel and c['role']==role and c['family']==family and c['arm']==arm);target=torch.tensor(original['target'],device='cuda',dtype=torch.float64);den=target[:,0].norm().clamp_min(1e-20);pred=prediction[ids]
                            records.append(dict(panel=panel,role=role,family=family,arm=arm,mode=mode,target=target.tolist(),prediction=pred.tolist(),number_error=float((pred[:,0]-target[:,0]).norm()/den),modal_error=((pred[:,1:]-target[:,1:]).norm(dim=0)/den).tolist()))
    a=counts=={k:PLAN[k] for k in counts} and all(max(c[k] for k in ['gradient_replay','source_replay','hessian_closure_absolute','hessian_closure_relative','local_core_replay'])<=1e-8 for c in checks)
    def passes(mode):return all(c['number_error']<=.1 and max(c['modal_error'])<=.05 for c in records if c['mode']==mode)
    guard_torch_save(dict(cases=export_cases,scope='Exact local curvature contributions at baseline, transported to five source amplitudes. Native generators required.'),EXPORT)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and passes('mlp_only')),bool(a and passes('mlp_readout'))])),checks=checks,records=records,prices=prices,export_file=str(EXPORT),export_bytes=EXPORT.stat().st_size,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','export_bytes','seconds']}))
if __name__=='__main__':main()
