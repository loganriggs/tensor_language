#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_full_quadratic pred_c_no_cross_group
"""Full15-direction five-source finite test, no new derivatives or fits.
24prefix/256native suffix batches; opened96rows/2sites. Native generators charged.
Null: source-group cross terms or higher-order effects invalidate simplification.
"""
import os,json,time,sys,hashlib,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'five_source_full_span_v1_result.json'
SETS=[(i,) for i in range(5)]+list(itertools.combinations(range(5),2))
PREDICTIONS=dict(pred_a_instrument='symmetricdesignrank15; knownnativecorners1e-4; exactcounts',pred_b_full_quadratic='fullquad number10/modal5 everycell',pred_c_no_cross_group='zero A-B curvature crosses number10/modal5 everycell')
PLAN=dict(prefix=24,suffix=256,fits=0,derivatives=0,source_sets=SETS,predictions=PREDICTIONS)
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
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm());eps=torch.finfo(torch.float32).eps
    enc=tiktoken.get_encoding('gpt2');words=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(w)==1 for w in words);modal=torch.tensor([w[0] for w in words],device='cuda').reshape(3,2)
    old=json.loads((A/'five_source_modal_null_v1_result.json').read_text());oldctx={(c['panel'],c['role'],c['template']):c for c in old['contexts']}
    counts=dict(prefix=0,suffix=0);records=[];replays=[]
    census=json.loads((A/'semantic_port_pairs_fresh_v1_result.json').read_text())

    for panel,digest in [('opposite','1869db8a18d4c89c28fa81a6bb57a19497e6f8692831eb4391f5d23c84556d73'),('congruent','bfba23ab066beb14f5cd469edce137db7ec1e8ccc26bfc36b66f896426b17511')]:
        path=P/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;rows=json.loads(path.read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float();_,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                ds=torch.stack([(pe[i]-pb[i])[batch,pos].float() for i in range(5)],dim=1);previous=oldctx[(panel,role,template)];
                G=torch.tensor(previous['gradient'],device='cuda',dtype=torch.float64);H=torch.tensor(previous['hessian'],device='cuda',dtype=torch.float64)
                blocked=H.clone();blocked[:,:,:2,2:]=0;blocked[:,:,2:,:2]=0
                def native(a):
                    counts['suffix']+=1;x=raw.clone();x[batch,pos]+=torch.einsum('bi,bid->bd',a.float(),ds)
                    for l in range(11,18):
                        block=model.transformer.h[l]
                        if l>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
                        at,_=block.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+at;x=x+block.mlp(F.rms_norm(x,(x.shape[-1],)))
                    logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30);z=logits.gather(1,pairs.reshape(len(x),8)).reshape(len(x),4,2)
                    return (z[:,:,0]-z[:,:,1]).double()
                zero=torch.zeros(len(entries),5,device='cuda');base=native(zero)
                for selected in SETS:
                    amp=zero.clone();amp[:,list(selected)]=1;av=amp.double();target=base-native(amp)
                    known=graph.PORTS[selected[0]] if len(selected)==1 else {(0,1):'A',(2,3):'pair23',(2,4):'pair24',(3,4):'pair34'}.get(selected)
                    for mode,curvature in [('full',H),('no_AB',blocked),('linear',torch.zeros_like(H))]:
                        pred=-torch.einsum('bop,bp->bo',G,av)-.5*torch.einsum('bp,bopq,bq->bo',av,curvature,av)
                        for family in dict.fromkeys(r['family'] for r in entries):
                            ids=[i for i,r in enumerate(entries) if r['family']==family];y=target[ids];den=y[:,0].norm().clamp_min(1e-20)
                            if mode=='full' and known:replays.append(float((y[:,0]-torch.tensor(census['data'][panel][role][known][family]['effects'],device='cuda',dtype=torch.float64)).abs().max()))
                            records.append(dict(panel=panel,role=role,family=family,source_set=list(selected),mode=mode,target=y.tolist(),prediction=pred[ids].tolist(),number_error=float((pred[ids,0]-y[:,0]).norm()/den),modal_error=((pred[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),number_norm=float(den)))
    design=[]
    for selected in SETS:
        av=np.zeros(5);av[list(selected)]=1;design.append([av[i]*av[j]*(1 if i==j else 2**.5) for i in range(5) for j in range(i,5)])
    rank=int(np.linalg.matrix_rank(design));a=rank==15 and counts==dict(prefix=24,suffix=256) and max(replays)<=1e-4
    def passes(mode):return all(c['number_error']<=.1 and max(c['modal_error'])<=.05 for c in records if c['mode']==mode)
    result=dict(plan=PLAN,counts=counts,design_rank=rank,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and passes('full')),bool(a and passes('no_AB'))])),max_native_replay=max(replays),records=records,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','design_rank','max_native_replay','seconds']}))
if __name__=='__main__':main()
