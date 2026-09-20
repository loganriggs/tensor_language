#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_polynomial_replay pred_b_full_ht pred_c_rank8
"""Native homogeneous two-MLP numerator fold and bounded pair-tree HT baseline."""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'native_two_mlp_quartic_ht_v1r1_result.json'
PREDICTIONS=dict(pred_a_polynomial_replay='same pure native path versus folded quartic relative1e-8',pred_b_full_ht='full pair-tree coefficient replay relative1e-10',pred_c_rank8='rank8 pair-tree symmetrized coefficient relative error<=10%')
PLAN=dict(prefix=24,full_forward=16,full_source_jvp=80,reader_reverse=64,folds=16,pure_path_replays=64,predictions=PREDICTIONS)
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
    from quartic_pair_tree import factor,reconstruct,symmetrize
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm());eps=torch.finfo(torch.float32).eps
    enc=tiktoken.get_encoding('gpt2');words=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(w)==1 for w in words);modal=torch.tensor([w[0] for w in words],device='cuda').reshape(3,2)
    old=json.loads((A/'five_source_modal_null_v1_result.json').read_text());oldctx={(c['panel'],c['role'],c['template']):c for c in old['contexts']}
    counts={k:0 for k in PLAN if k!='predictions'};checks=[];exports=[]

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
                K=ports['mlp_input_directions'][0][batch,pos];q=ports['mlp_readers'][1][batch,pos]
                m1=model.transformer.h[11].mlp;m2=model.transformer.h[12].mlp;lam=model.transformer.h[12].lambdas[0].double()
                L1,R1,D1=[x.weight.double() for x in [m1.Left,m1.Right,m1.Down]]
                L2,R2,D2=[x.weight.double() for x in [m2.Left,m2.Right,m2.Down]]
                LK=torch.einsum('hd,bdi->bhi',L1,K);RK=torch.einsum('hd,bdi->bhi',R1,K)
                S=torch.einsum('dh,bhi,bhj->bdij',D1,LK,RK);S=lam*(S+S.transpose(-1,-2))/2
                left=torch.einsum('hd,bdij->bhij',L2,S);right=torch.einsum('hd,bdij->bhij',R2,S);C=torch.einsum('bod,dh->boh',q,D2)
                H=torch.einsum('boh,bhij,bhkl->boijkl',C,left,right);counts['folds']+=1
                replay=[]
                gen=torch.Generator(device='cuda');gen.manual_seed(842)
                for a in [torch.zeros(len(K),5,device='cuda',dtype=torch.float64),torch.ones(len(K),5,device='cuda',dtype=torch.float64),torch.randn(len(K),5,device='cuda',dtype=torch.float64,generator=gen),-torch.ones(len(K),5,device='cuda',dtype=torch.float64)]:
                    x=torch.einsum('bdi,bi->bd',K,a);h=lam*((x@L1.T)*(x@R1.T))@D1.T
                    direct=torch.einsum('bod,bd->bo',q,((h@L2.T)*(h@R2.T))@D2.T)
                    folded=torch.einsum('boijkl,bi,bj,bk,bl->bo',H,a,a,a,a)
                    replay.append(float((direct-folded).norm()/direct.norm().clamp_min(1e-30)));counts['pure_path_replays']+=1
                tensor=H.cpu();exports.append(dict(panel=panel,role=role,template=template,hessian_not_applicable_quartic=tensor))
                for row,hn in enumerate(tensor.numpy()):
                    sym=symmetrize(hn);den=max(np.linalg.norm(sym),1e-30)
                    for representative,target in [('native_pair',hn),('symmetric',sym)]:
                        full=reconstruct(factor(target,25));reduced=reconstruct(factor(target,8))
                        checks.append(dict(panel=panel,role=role,template=template,row=row,representative=representative,polynomial_replay=max(replay),full_coefficient_error=float(np.linalg.norm(full-target)/max(np.linalg.norm(target),1e-30)),rank8_polynomial_coefficient_error=float(np.linalg.norm(symmetrize(reduced)-sym)/den)))
    result=dict(plan=PLAN,counts=counts,checks=checks,predictions=dict(pred_a_polynomial_replay=counts=={k:PLAN[k] for k in counts} and all(c['polynomial_replay']<1e-8 for c in checks),pred_b_full_ht=all(c['full_coefficient_error']<1e-10 for c in checks),pred_c_rank8=all(c['rank8_polynomial_coefficient_error']<=.1 for c in checks)),scope='Single nominated-position homogeneous MLP11->direct lambda12->MLP12 numerator path under native derivative source frame and downstream readers. Biases, backgrounds, RMS denominators and attention12 path excluded explicitly; not full native finite edit or causal extraction.',seconds=time.perf_counter()-tic)
    export=A/'native_two_mlp_quartic_ht_v1r1.pt';guard_torch_save(exports,export)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps(result['predictions']))
if __name__=='__main__':main()
