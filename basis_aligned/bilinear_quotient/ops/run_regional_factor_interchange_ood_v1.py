#!/usr/bin/env python3
# BQGATE:10 body forwards;48 sequences lengths6-14;16 batched suffix arms;no fit;180sec.
"""pred_a frozen recipient/donor replay<=1e-5 and interaction reconstruction<=1e-10;
pred_b full-port transplant regional transfer>=10%native and >=4/6positive EACHfamily;
pred_c child-bundle transfer>=90%full transfer and unrelated<=.5regional EACHfamily.
Null: parent need not be the regional cue carrier even if the whole block contributes.
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
from factorial_source_ports_v1 import vertices,mobius,reconstruct
STEM='REGIONAL_FACTOR_INTERCHANGE_OOD_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json').read_text())['rows'];buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==48 and len(buckets)==8 and max(buckets)==14
    batches=[(length,ids[off:off+8]) for length,ids in sorted(buckets.items()) for off in range(0,len(ids),8)];assert len(batches)==10
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    for name in ('FACTORIAL_SOURCE_PORTS_V1_CONTROL.json','COMMON_QUADRATIC_PORTS_V1_CONTROL.json'):
        c=json.loads((P/name).read_text());assert max(v for v in c.values() if isinstance(v,(int,float)))<=1e-10
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('10 body forwards48sequences6-14tokens;16suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    pre=torch.empty(48,1152,device='cuda');writes=torch.zeros(48,16,1152,dtype=torch.float64,device='cuda');replays=[];count=0
    handle=model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))
    try:
        for length,ids in batches:
            assert len(ids)%2==0 and all(ids[j]^1==ids[j+1] for j in range(0,len(ids),2));donor_indices=torch.arange(len(ids),device='cuda')^1
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                parent,children=source_ports(source,atoms);writers=private_writers(query,atoms,cross_factors(atoms,*weights))
                base_ports=[gate/g0[None],parent,children,writers];donor_ports=[v[donor_indices] for v in base_ports]
                writes[ids]+=vertices(base_ports,donor_ports).permute(1,0,2)
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:handle.remove()
    assert count==10
    def margins(delta):
        z=pre+delta.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    prior=torch.load(P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writes']['collapsed_all'].cuda()
    recipient_error=float((writes[:,0]-prior).norm()/prior.norm());donor=prior[torch.arange(48,device='cuda')^1];donor_error=float((writes[:,15]-donor).norm()/donor.norm())
    arms=torch.stack([margins(writes[:,mask]-writes[:,0]) for mask in range(16)]);interactions=mobius(arms);grid_error=float((reconstruct(interactions)-arms).norm()/arms.norm());cells=[]
    for family in (0,1,2,3):
        uk=[i for i,r in enumerate(rows) if r['family']==family and r['cue']=='British'];us=[i+1 for i in uk]
        native=arms[0,uk]-arms[0,us];transfer=(native[None]-(arms[:,uk]-arms[:,us]))/2
        cells.append(dict(family=family,native_regional_gap=float(native[:,0].mean()),mean_transfer_by_mask=transfer.mean(1).tolist(),positive_regional_by_mask=(transfer[:,:,0]>0).sum(1).tolist(),child_regional_meanabs=float(transfer[4,:,0].abs().mean()),child_unrelated_meanabs=float(transfer[4,:,1].abs().mean()),interaction_contrast_by_mask=((interactions[:,uk]-interactions[:,us])/2).mean(1).tolist()))
    instrument=recipient_error<=1e-5 and donor_error<=1e-5 and grid_error<=1e-10
    full=instrument and all(c['mean_transfer_by_mask'][15][0]>=.1*c['native_regional_gap'] and c['positive_regional_by_mask'][15]>=4 for c in cells)
    result={'pred_a':instrument,'pred_b':full,'pred_c':full and all(c['mean_transfer_by_mask'][4][0]>=.9*c['mean_transfer_by_mask'][15][0] and c['child_unrelated_meanabs']<=.5*c['child_regional_meanabs'] for c in cells)}
    torch.save(dict(pre=pre.cpu(),write_vertices=writes.cpu(),arm_margins=arms.cpu(),interaction_margins=interactions.cpu()),art)
    result.update(recipient_replay_error=recipient_error,donor_replay_error=donor_error,grid_reconstruction_error=grid_error,native_attention_replay_errors=replays,cells=cells,port_bits=dict(gate=1,parent=2,children=4,query_writers=8),body_forwards=count,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='Paired-context factor interventions with recipient native remainder fixed. Conditional finite-grid accounting, not a universal semantic abstraction or four-property promotion.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
