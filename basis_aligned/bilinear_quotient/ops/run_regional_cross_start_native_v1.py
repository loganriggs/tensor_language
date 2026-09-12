#!/usr/bin/env python3
# BQGATE:10 body forwards;48 sequences lengths6-14;3 batched suffix arms;no fit;180sec.
"""pred_a native replay<=1e-5, independent/target write error<=.1;
pred_b independent removal>=10%native cue and>=4/6positive EACHfamily;
pred_c signed removal error<=.1relative EACHfamily.
Null: source graph similarity does not preserve native effects across independent fits.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary
from shared_cubic_source_projection_v1 import cross_factors
from cubic_source_mixture_execute_v1 import execute
STEM='REGIONAL_CROSS_START_NATIVE_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json').read_text())['rows'];buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==48 and len(buckets)==8 and max(buckets)==14
    batches=[(length,ids[off:off+8]) for length,ids in sorted(buckets.items()) for off in range(0,len(ids),8)];assert len(batches)==10
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('10 body forwards48sequences6-14tokens;3suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    other=torch.load(P/'SHARED_CUBIC_SOURCE_CONTINUE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'][0]
    permutation=[i for i in range(16) if i not in (11,2)]+[11,2];other=other[permutation].cuda()
    simple=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    programs=dict(secant=(other,torch.eye(16,dtype=torch.float64,device='cuda')),collapsed=(simple,torch.eye(16,dtype=torch.float64,device='cuda')))
    pre=torch.empty(48,1152,device='cuda');writes={name:torch.zeros(48,1152,dtype=torch.float64,device='cuda') for name in ('secant','collapsed','secant_all','collapsed_all')};replays=[];count=0
    handle=model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))
    try:
        for length,ids in batches:
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                for name,(components,matrix) in programs.items():
                    terms=execute(query,source,components,matrix,cross_factors(components,*weights));change=(terms[:,:,-2:]*(gate/g0[None])[:,:,None,None]).sum((1,2));writes[name+'_all'][ids]+=change
                    if pos in (0,length-2):writes[name][ids]+=change
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:handle.remove()
    assert count==10
    def margins(removal):
        z=pre-removal.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    candidate=writes['collapsed_all'];independent=writes['secant_all']
    arms=torch.stack([margins(torch.zeros_like(pre)),margins(candidate),margins(independent)],1);cells=[]
    for family in (0,1,2,3):
        uk=[i for i,r in enumerate(rows) if r['family']==family and r['cue']=='British'];us=[i+1 for i in uk];ids=sorted(uk+us)
        native=arms[uk,0]-arms[us,0];target_reduction=native-(arms[uk,1]-arms[us,1]);other_reduction=native-(arms[uk,2]-arms[us,2]);target_effect=arms[ids,1]-arms[ids,0];other_effect=arms[ids,2]-arms[ids,0]
        error=float((other_effect[:,0]-target_effect[:,0]).norm()/target_effect[:,0].norm().clamp_min(1e-30));paired_error=float((other_reduction[:,0]-target_reduction[:,0]).norm()/target_reduction[:,0].norm().clamp_min(1e-30))
        cells.append(dict(family=family,native_mean=float(native[:,0].mean()),independent_mean_reduction=float(other_reduction[:,0].mean()),target_mean_reduction=float(target_reduction[:,0].mean()),independent_positive=int((other_reduction[:,0]>0).sum()),signed_effect_relative_error=error,paired_cue_reduction_relative_error=paired_error,independent_unrelated_meanabs_reduction=float(other_reduction[:,1].abs().mean())))
    write_error=float((independent-candidate).norm()/candidate.norm());instrument=max(replays)<=1e-5 and write_error<=.1;behavior=instrument and all(c['independent_mean_reduction']>=.1*c['native_mean'] and c['independent_positive']>=4 for c in cells)
    result={'pred_a':instrument,'pred_b':behavior,'pred_c':behavior and all(c['signed_effect_relative_error']<=.1 and c['paired_cue_reduction_relative_error']<=.1 for c in cells)}
    torch.save(dict(pre=pre.cpu(),target_write=candidate.cpu(),independent_write=independent.cpu(),arm_margins=arms.cpu()),art)
    result.update(cells=cells,native_replay_errors=replays,relative_write_error=write_error,arm_order=['native','delete_target','delete_independent_fit'],arm_margins=arms.cpu().tolist(),body_forwards=count,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='Independent-fit block selected from weights only, each retains its own eliminated query writers. Native held-out correspondence, not full four-property promotion.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','arm_margins')}),flush=True)
if __name__=='__main__':main()
