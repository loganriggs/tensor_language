#!/usr/bin/env python3
# BQGATE:9 body forwards;48 sequences lengths13-19;7 suffix arms;no fit;180sec.
"""pred_a native/priorcomponent and128rank write replay<=1e-5.
pred_b both32 write/removal-effect<=.1 EACHfamily; pred_c both64 same.
Query32/key32 separate diagnostics; isotropic orthogonal remainder, no textfit.
Exact symmetric eigensolve.9bodybatches48rows13–19tokens7suffixarms180sec.
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
from regional_cue_row_check_v1 import validate
from spectral_norm_quadratic_v1 import compile_norm,norm_mean
STEM='REGIONAL_SPECTRAL_NORMALIZERS_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_VALUE_FRESH_GEO_V1_ROWS.json').read_text())['rows'];validate(rows);buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==48 and len(buckets)==7 and max(buckets)==19
    batches=[(length,ids[off:off+8]) for length,ids in sorted(buckets.items()) for off in range(0,len(ids),8)];assert len(batches)==9
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    for name in ('FACTORIAL_SOURCE_PORTS_V1_CONTROL.json','COMMON_QUADRATIC_PORTS_V1_CONTROL.json'):
        c=json.loads((P/name).read_text());assert max(v for v in c.values() if isinstance(v,(int,float)))<=1e-10
    assert json.loads((P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json').read_text())['relative_replay_error']<=1e-12
    assert json.loads((P/'RESIDUAL_PAYLOAD_UNROLL_V1_CONTROL.json').read_text())['relative_replay_error']<=1e-12
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('9 body forwards48sequences13-19tokens;7suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    pre=torch.empty(48,1152,device='cuda');writes=torch.zeros(48,6,1152,dtype=torch.float64,device='cuda');replays=[];count=0
    norm_programs=[compile_norm(t) for t in (q1,q2,k1,k2)]
    handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
    try:
        for length,ids in batches:
            assert len(ids)%2==0 and all(ids[j]^1==ids[j+1] for j in range(0,len(ids),2));donor_indices=torch.arange(len(ids),device='cuda')^1
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            rho=(r.double().square().mean(-1,keepdim=True)+eps).sqrt()
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                parent,children=source_ports(source,atoms);writers=private_writers(query,atoms,cross_factors(atoms,*weights))
                qnorm=(qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)
                knorm=(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)
                approx={rank:[norm_mean(query,norm_programs[0],rank)*norm_mean(query,norm_programs[1],rank),
                              norm_mean(current[:,pos],norm_programs[2],rank)*norm_mean(current[:,pos],norm_programs[3],rank)] for rank in (32,64,128)}
                gates=[gate,1/(128**2*(approx[32][0]*knorm).sqrt()),1/(128**2*(qnorm*approx[32][1]).sqrt())]+[1/(128**2*(approx[rank][0]*approx[rank][1]).sqrt()) for rank in (32,64,128)]
                writes[ids]+=torch.stack([write(g/g0[None],parent,children,writers) for g in gates],1)
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:
        for handle in handles:handle.remove()
    assert count==9
    def margins(delta):
        z=pre+delta.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    baseline=margins(torch.zeros_like(pre));arms=torch.stack([margins(-writes[:,j]) for j in range(6)])
    effects=baseline[None]-arms;cells=[]
    previous=torch.load(P/'REGIONAL_VALUE_FRESH_GEO_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['write_vertices'][:,0].cuda()
    replay=float((writes[:,0]-previous).norm()/previous.norm())
    for arm,name in enumerate(('query32','key32','both32','both64','both128'),1):
        for family in range(4):
            ids=[i for i,row in enumerate(rows) if row['family']==family]
            cells.append(dict(arm=name,family=family,write_relative_error=float((writes[ids,arm]-writes[ids,0]).norm()/writes[ids,0].norm()),
                effect_relative_error=float((effects[arm,ids,0]-effects[0,ids,0]).norm()/effects[0,ids,0].norm()),
                reference_effect_norm=float(effects[0,ids,0].norm()),predicted_effect_norm=float(effects[arm,ids,0].norm())))
    instrument=max(replays+[replay]+[c['write_relative_error'] for c in cells if c['arm']=='both128'])<=1e-5
    def passed(name):return instrument and all(c['write_relative_error']<=.1 and c['effect_relative_error']<=.1 for c in cells if c['arm']==name)
    result={'pred_a':instrument,'pred_b':passed('both32'),'pred_c':passed('both64')}
    torch.save(dict(pre=pre.cpu(),write_vertices=writes.cpu(),baseline_margins=baseline.cpu(),removal_margins=arms.cpu()),art)
    result.update(cells=cells,prior_component_replay=replay,native_attention_replays=replays,norm_reader_scalars={str(k):4*9*k*1152+4*9*k+4*9 for k in (32,64,128)},
                  seconds=time.perf_counter()-tic,body_forwards=count,artifact_sha=digest(art),source_shas=binding,
                  scope='Frozen shared component with top32/64 weight eigenreaders plus isotropic complementary norm;128 exact endpoint. No fit. Native input states retained; conditional norm extraction, not whole-model closure or arbitrary-input guarantee.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
