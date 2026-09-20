#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_output pred_b_exact_hessian pred_c_one_shot_speed
"""Native conditional source-attention compiler and paired producer benchmark.
Full native contexts/readers/Jacobians remain charged. No fits or approximation.
24prefix+16fullforward+80JVP+64readerreverse;112compiles;
224native+224compiled replays,112Hessians each native/compiled.
"""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'native_attention_source_core_v1_result.json'
PREDICTIONS=dict(pred_a_exact_output='native output maxabs1e-8',pred_b_exact_hessian='compiled/native/saved Hessians absandrelative1e-8',pred_c_one_shot_speed='sum(compile+foldedHessian time)<sum(native Hessian time)')
PLAN=dict(prefix=24,full_forward=16,full_source_jvp=80,reader_reverse=64,compiles=112,native_replay=224,compiled_replay=224,native_hessian=112,compiled_hessian=112,predictions=PREDICTIONS)
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
    from attention_source_core import compile_core,execute
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm());eps=torch.finfo(torch.float32).eps
    enc=tiktoken.get_encoding('gpt2');words=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(w)==1 for w in words);modal=torch.tensor([w[0] for w in words],device='cuda').reshape(3,2)
    old=json.loads((A/'five_source_modal_null_v1_result.json').read_text());oldctx={(c['panel'],c['role'],c['template']):c for c in old['contexts']}
    counts=dict(prefix=0,full_forward=0,full_source_jvp=0,reader_reverse=0,compiles=0,native_replay=0,compiled_replay=0,native_hessian=0,compiled_hessian=0);checks=[];prices=[]
    saved=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)
    saved_cases={(c['panel'],c['role'],c['template']):c for c in saved['cases']}

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
                zero=torch.zeros(len(entries),5,device='cuda',dtype=torch.float64)
                def hessian(fn):
                    with torch.enable_grad():
                        z=zero.clone().requires_grad_();y=fn(z);hs=[]
                        for o in range(4):
                            g=torch.autograd.grad(y[:,o].sum(),z,create_graph=True,retain_graph=True)[0]
                            hs.append(torch.stack([torch.autograd.grad(g[:,j].sum(),z,retain_graph=not(o==3 and j==4))[0] for j in range(5)],dim=1))
                    return torch.stack(hs,dim=1).detach()
                def timed(fn):
                    torch.cuda.synchronize();t0=time.perf_counter();result=fn();torch.cuda.synchronize();return result,time.perf_counter()-t0
                def values(v):return v.numel() if isinstance(v,torch.Tensor) else sum(values(x) for x in v.values()) if isinstance(v,dict) else 0
                for index,l in enumerate(ports['layers']):
                    block=model.transformer.h[l];att=block.attn;q=ports['attention_readers'][index]
                    if l==11:
                        before=raw.double();K=torch.zeros(*raw.shape,5,device='cuda',dtype=torch.float64);K[batch,pos]=ds.double().transpose(1,2)
                    else:
                        before=block.lambdas[0].double()*ports['mlp_outputs'][index-1]+block.lambdas[1].double()*x0.double();K=block.lambdas[0].double()*ports['mlp_output_directions'][index-1]
                    # Weight materialization is part of compilation, not a free cache.
                    def compiler():
                        weights={key:getattr(att,name).weight.double() for key,name in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]}
                        cos,sin=att.rotary(torch.zeros(len(raw),raw.shape[1],att.n_head,att.head_dim,device='cuda',dtype=torch.float32))
                        return compile_core(weights,before,K,q,first,att.lamb.double(),att.n_head,cos[0,:,0].double(),sin[0,:,0].double(),eps,eps)
                    core,compile_seconds=timed(compiler);counts['compiles']+=1
                    def dense(a):return torch.einsum('btod,btd->bo',q,attention_write64(block,before+torch.einsum('btdp,bp->btd',K,a),first))
                    def folded(a):return execute(core,a)
                    replay=[]
                    for amplitude in [zero,torch.ones_like(zero)]:
                        native=dense(amplitude);compiled=folded(amplitude);counts['native_replay']+=1;counts['compiled_replay']+=1;replay.append(float((native-compiled).abs().max()))
                    if index%2==0:
                        hn,native_seconds=timed(lambda:hessian(dense));hc,folded_seconds=timed(lambda:hessian(folded))
                    else:
                        hc,folded_seconds=timed(lambda:hessian(folded));hn,native_seconds=timed(lambda:hessian(dense))
                    counts['native_hessian']+=1;counts['compiled_hessian']+=1
                    savedH=saved_cases[(panel,role,template)]['terms'][f'attention{l}'].to('cuda');den=hn.norm().clamp_min(1e-20)
                    checks.append(dict(panel=panel,role=role,template=template,layer=l,output_replay=max(replay),hessian_absolute=float((hc-hn).abs().max()),hessian_relative=float((hc-hn).norm()/den),saved_hessian_replay=float((hn-savedH).abs().max()),compile_seconds=compile_seconds,native_hessian_seconds=native_seconds,folded_hessian_seconds=folded_seconds))
                    prices.append(dict(panel=panel,role=role,template=template,layer=l,compiled_values=values(core),native_attention_weight_values=sum(getattr(att,n).weight.numel() for n in ['c_q','c_k','c_q2','c_k2','c_v','c_proj']),background_values=before.numel(),source_sensitivity_values=K.numel(),reader_values=q.numel(),cached_first_values=first.numel()))
    a=counts=={k:PLAN[k] for k in counts} and all(c['output_replay']<=1e-8 for c in checks)
    b=a and all(max(c[k] for k in ['hessian_absolute','hessian_relative','saved_hessian_replay'])<=1e-8 for c in checks)
    compile_time=sum(c['compile_seconds'] for c in checks);native_time=sum(c['native_hessian_seconds'] for c in checks);folded_time=sum(c['folded_hessian_seconds'] for c in checks)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(b and compile_time+folded_time<native_time)])),checks=checks,prices=prices,compile_seconds=compile_time,native_hessian_seconds=native_time,folded_hessian_seconds=folded_time,one_shot_ratio=(compile_time+folded_time)/native_time,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','one_shot_ratio','seconds']}))
if __name__=='__main__':main()
