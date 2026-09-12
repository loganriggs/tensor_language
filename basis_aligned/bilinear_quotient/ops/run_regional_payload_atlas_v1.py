#!/usr/bin/env python3
# BQGATE:7 body forwards;32 sequences lengths4-12;39 batched suffix arms;no fit;180sec.
"""pred_a projected residual/child/full-donor replay<=1e-5;
pred_b all-payload transplant>=10%native and>=5/8positive EACHfamily;
pred_c embedding-only transfer>=50%all-payload, unrelated<=.5regional EACHfamily.
Null: the large embedding recurrence coefficient need not dominate causal cue transfer.
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
from residual_payload_unroll_v1 import coefficients
STEM='REGIONAL_PAYLOAD_ATLAS_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_V2_ROWS.json').read_text())['rows'];buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==32 and len(buckets)==7 and max(buckets)<=12 and all(len(v)<=8 for v in buckets.values())
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    for name in ('FACTORIAL_SOURCE_PORTS_V1_CONTROL.json','COMMON_QUADRATIC_PORTS_V1_CONTROL.json'):
        c=json.loads((P/name).read_text());assert max(v for v in c.values() if isinstance(v,(int,float)))<=1e-10
    assert json.loads((P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json').read_text())['relative_replay_error']<=1e-12
    assert json.loads((P/'RESIDUAL_PAYLOAD_UNROLL_V1_CONTROL.json').read_text())['relative_replay_error']<=1e-12
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('7 body forwards32sequences4-12tokens;39suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    pre=torch.empty(32,1152,device='cuda');writes=torch.zeros(32,39,1152,dtype=torch.float64,device='cuda');replays=[];count=0
    fold=torch.load(P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');current_readers=fold['current_readers'].cuda();first_readers=fold['first_readers'].cuda();lambdas=torch.stack([block.lambdas.double() for block in model.transformer.h]);embedding_scale,update_scales=coefficients(lambdas);producer_errors=[];child_errors=[]
    handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
    for layer in range(17):
        handles.append(model.transformer.h[layer].attn.register_forward_hook(lambda module,args,out,layer=layer:captured['attention'].update({layer:out[0].double()@current_readers.T})))
        handles.append(model.transformer.h[layer].mlp.register_forward_hook(lambda module,args,out,layer=layer:captured['mlp'].update({layer:out.double()@current_readers.T})))
    try:
        for length,ids in sorted(buckets.items()):
            assert len(ids)%2==0 and all(ids[j]^1==ids[j+1] for j in range(0,len(ids),2));donor_indices=torch.arange(len(ids),device='cuda')^1
            captured['attention']={};captured['mlp']={}
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            rho=(r.double().square().mean(-1,keepdim=True)+eps).sqrt();numerators=[embedding_scale*(x0.double()@current_readers.T)]+[update_scales[j]*captured['attention'][j] for j in range(17)]+[update_scales[j]*captured['mlp'][j] for j in range(17)];reference=r.double()@current_readers.T;producer_errors.append(float((sum(numerators)-reference).norm()/reference.norm()))
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                parent,children=source_ports(source,atoms);writers=private_writers(query,atoms,cross_factors(atoms,*weights))
                parts=[n[:,pos] for n in numerators]+[rho[:,pos],first[:,pos]@first_readers.T]
                def child(parts):return sum(parts[:35])/parts[35]+parts[36]
                child_errors.append(float((child(parts)-children).norm()/children.norm()))
                variants=[children]+[child([v[donor_indices] if j==changed else v for j,v in enumerate(parts)]) for changed in range(37)]+[child([v[donor_indices] for v in parts])]
                writes[ids]+=torch.stack([write(gate/g0[None],parent,u,writers) for u in variants],1)
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:
        for handle in handles:handle.remove()
    assert count==7
    def margins(delta):
        z=pre+delta.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    prior=torch.load(P/'REGIONAL_FACTOR_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['write_vertices'].cuda()
    recipient_error=float((writes[:,0]-prior[:,0]).norm()/prior[:,0].norm());donor_error=float((writes[:,38]-prior[:,4]).norm()/prior[:,4].norm())
    arms=torch.stack([margins(writes[:,j]-writes[:,0]) for j in range(39)]);cells=[]
    for family in (0,1):
        uk=[i for i,r in enumerate(rows) if r['family']==family and r['cue']=='British'];us=[i+1 for i in uk];native=arms[0,uk]-arms[0,us];transfer=(native[None]-(arms[:,uk]-arms[:,us]))/2
        cells.append(dict(family=family,native_regional_gap=float(native[:,0].mean()),mean_transfer_by_arm=transfer.mean(1).tolist(),positive_regional_by_arm=(transfer[:,:,0]>0).sum(1).tolist(),embedding_regional_meanabs=float(transfer[1,:,0].abs().mean()),embedding_unrelated_meanabs=float(transfer[1,:,1].abs().mean())))
    instrument=max(producer_errors+child_errors+[donor_error,recipient_error])<=1e-5;full=instrument and all(c['mean_transfer_by_arm'][38][0]>=.1*c['native_regional_gap'] and c['positive_regional_by_arm'][38]>=5 for c in cells)
    result={'pred_a':instrument,'pred_b':full,'pred_c':full and all(c['mean_transfer_by_arm'][1][0]>=.5*c['mean_transfer_by_arm'][38][0] and c['embedding_unrelated_meanabs']<=.5*c['embedding_regional_meanabs'] for c in cells)}
    torch.save(dict(pre=pre.cpu(),write_vertices=writes.cpu(),arm_margins=arms.cpu()),art)
    result.update(recipient_replay_error=recipient_error,all_payload_donor_replay_error=donor_error,residual_unroll_errors=producer_errors,child_decomposition_errors=child_errors,native_attention_replay_errors=replays,cells=cells,arm_names=['native','embedding_numerator']+['attention_'+str(j) for j in range(17)]+['MLP_'+str(j) for j in range(17)]+['RMS_divisor','first_stream','all_payload_ports'],body_forwards=count,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='Exact conditional residual-write atlas for frozen regional payload. Isolated producer swaps retain recipient RMS and all downstream gates. Module labels are attribution ports, not identified semantic circuits.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','child_decomposition_errors')}),flush=True)
if __name__=='__main__':main()
