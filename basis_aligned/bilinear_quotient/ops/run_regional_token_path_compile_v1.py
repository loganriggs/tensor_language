#!/usr/bin/env python3
# BQGATE:6 body forwards;32 sequences lengths7-11;2 suffix arms;no fit;180sec.
"""pred_a producer/native replay<=1e-5 and compiled actual write<=1e-10;
pred_b independent synthetic write replay<=1e-10;
pred_c compiled native suffix margins versus saved first-swap<=1e-5 relative.
Null: missing a gate, normalization or source-position edge breaks path compilation.
Context coefficient K is not independently generated; all external ports are charged.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary
from shared_cubic_source_projection_v1 import cross_factors
from common_quadratic_ports_v1 import source_ports,private_writers
from factorial_source_ports_v1 import write
from jacclust.tt_model import apply_rotary_emb
from conditional_token_path_v1 import compile_path,execute_path
STEM='REGIONAL_TOKEN_PATH_COMPILE_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_BEHAVIOR_CONTROLS_V1_ROWS.json').read_text())['rows'];buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==32 and len(buckets)==5 and max(buckets)==11
    batches=[(length,ids[off:off+8]) for length,ids in sorted(buckets.items()) for off in range(0,len(ids),8)];assert len(batches)==6
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    for name in ('FACTORIAL_SOURCE_PORTS_V1_CONTROL.json','COMMON_QUADRATIC_PORTS_V1_CONTROL.json'):
        c=json.loads((P/name).read_text());assert max(v for v in c.values() if isinstance(v,(int,float)))<=1e-10
    assert json.loads((P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json').read_text())['relative_replay_error']<=1e-12
    assert json.loads((P/'RESIDUAL_PAYLOAD_UNROLL_V1_CONTROL.json').read_text())['relative_replay_error']<=1e-12
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('6 body forwards32sequences7-11tokens;2suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    pre=torch.empty(32,1152,device='cuda');writes=torch.zeros(32,2,1152,dtype=torch.float64,device='cuda');replays=[];count=0
    folds=torch.load(P/'REGIONAL_PRODUCER_OV_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    C=torch.load(P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['current_readers'].cuda()
    producer_errors=[]
    operators=torch.zeros(32,3,9,2,1152,dtype=torch.float64,device='cuda');source_deltas=torch.zeros(32,3,9,2,dtype=torch.float64,device='cuda')
    torch.manual_seed(61207);synthetic=torch.randn_like(source_deltas)*.1;synthetic_direct=torch.zeros(32,1152,dtype=torch.float64,device='cuda')
    handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
    def producer_hook(layer,module,args,out):
        inp=args[0];n,length,_=inp.shape
        q,k,q2,k2=[getattr(module,name)(inp).reshape(n,length,9,128) for name in ('c_q','c_k','c_q2','c_k2')]
        cos,sin=module.rotary(q)
        q,k,q2,k2=[apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin).double() for v in (q,k,q2,k2)]
        routing=(torch.einsum('nthd,nshd->nhts',q,k)/128)*(torch.einsum('nthd,nshd->nhts',q2,k2)/128)
        routing=routing*torch.ones(length,length,dtype=torch.bool,device='cuda').tril()[None,None]
        fold=folds[str(layer)];E=fold['current_value_readers'].cuda();H=fold['first_value_readers'].cuda()
        current_values=torch.einsum('nsd,ahd->nsha',inp.double(),E)
        first_values=torch.einsum('nsd,ahd->nsha',captured['token_first'].double(),H)
        donor=torch.arange(n,device='cuda')^1
        variants=torch.stack([torch.einsum('nhts,nsha->nta',routing,current_values+(first_values[donor] if mask else first_values)) for mask in range(2)],1)
        reference=float(fold['residual_scale'])*(out[0].double()@C.T)
        producer_errors.append(float((variants[:,0]-reference).norm()/reference.norm()))
        captured['producer_variants'].append(variants)
        captured['routes'].append(routing[:,:,:,1]);captured['source_deltas'].append(first_values[donor,1]-first_values[:,1])
    for layer in (8,9,13):
        handles.append(model.transformer.h[layer].attn.register_forward_hook(lambda module,args,out,layer=layer:producer_hook(layer,module,args,out)))
    try:
        for length,ids in batches:
            assert len(ids)%2==0 and all(ids[j]^1==ids[j+1] for j in range(0,len(ids),2));donor_indices=torch.arange(len(ids),device='cuda')^1
            captured['producer_variants']=[];captured['routes']=[];captured['source_deltas']=[]
            assert all(rows[i]['ids'][1]!=rows[i^1]['ids'][1] and rows[i]['ids'][:1]+rows[i]['ids'][2:]==rows[i^1]['ids'][:1]+rows[i^1]['ids'][2:] for i in ids)
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            initial=model.transformer.h[0].lambdas;captured['token_first']=F.rms_norm(initial[0]*x0+initial[1]*x0,(1152,))
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            rho=(r.double().square().mean(-1,keepdim=True)+eps).sqrt();producer_variants=sum(captured['producer_variants']);routes=torch.stack(captured['routes'],1);source_deltas[ids]=torch.stack(captured['source_deltas'],1);downstream=[]
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                parent,children=source_ports(source,atoms);writers=private_writers(query,atoms,cross_factors(atoms,*weights))
                down=torch.einsum('nh,n,nhjo->njo',gate/g0[None],parent,writers)/rho[:,pos,None];downstream.append(down)
                synth_child=torch.einsum('njh,njha->na',routes[:,:,:,pos],synthetic[ids])/rho[:,pos]
                synthetic_direct[ids]+=write(gate/g0[None],parent,synth_child,writers)
                variants=[children+(producer_variants[:,mask,pos]-producer_variants[:,0,pos])/rho[:,pos] for mask in range(2)]
                writes[ids]+=torch.stack([write(gate/g0[None],parent,u,writers) for u in variants],1)
            operators[ids]=compile_path(routes,torch.stack(downstream,1))
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:
        for handle in handles:handle.remove()
    assert count==6
    def margins(delta):
        z=pre+delta.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    direct=writes[:,1]-writes[:,0];compiled=execute_path(operators,source_deltas);synth=execute_path(operators,synthetic)
    write_error=float((compiled-direct).norm()/direct.norm());synthetic_error=float((synth-synthetic_direct).norm()/synthetic_direct.norm())
    prior=torch.load(P/'REGIONAL_BEHAVIOR_CONTROLS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');compiled_margins=margins(compiled)
    margin_error=float((compiled_margins-prior['arm_margins'][1].cuda()).norm()/prior['arm_margins'][1].norm())
    pred_a=max(producer_errors+replays)<=1e-5 and write_error<=1e-10
    result={'pred_a':pred_a,'pred_b':pred_a and synthetic_error<=1e-10,'pred_c':pred_a and synthetic_error<=1e-10 and margin_error<=1e-5}
    torch.save(dict(representative_operators=operators[:2].cpu(),source_deltas=source_deltas.cpu(),compiled_writes=compiled.cpu(),synthetic_writes=synth.cpu(),compiled_margins=compiled_margins.cpu(),pre=pre.cpu()),art)
    result.update(compiled_write_relative_error=write_error,synthetic_write_relative_error=synthetic_error,compiled_suffix_margin_relative_error=margin_error,producer_replay_errors=producer_errors,native_attention_replay_errors=replays,operator_shape=list(operators.shape),operator_floats_per_context=operators[0].numel(),stored_contexts=[0,1],body_forwards=count,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='Exact54-port linear map for one changed cue position, recipient routes/normalization/parent/query writers fixed. Synthetic independent value-port edits are not necessarily valid token changes. Two representativecontext operatorssaved; K generation stillrequiresnativecontext. No independentextraction or compressionadoption.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
