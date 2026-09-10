#!/usr/bin/env python3
# BQGATE: 13forwards104seq; fixed two stable products; local readout variants.
"""pred_a live instrument; pred_b both pairs <=.20 conditional response error;
pred_c removal mean absolute CE >=.05 class and <=.01 nonclass, both pairs.
"""
import os, sys, json, time, signal
from pathlib import Path
RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parents[3]; POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(POLY)]
import torch
import torch.nn.functional as F
from circuit_fast_screen_producer import Bilin18TorchBackend
from circuit_fast_screen_managed_runner import atomic_create_json
from induction_context_transport_v2 import digest
OUT=POLY/'STABLE_JOINT32_REFLECTION_V1_RESULT.json'; BIND=POLY/'STABLE_JOINT32_REFLECTION_V1_BINDING.json'

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==s for p,s in binding.items())
    program=torch.load(POLY/'STABLE_JOINT32_REFLECTION_V1_PROGRAM.pt',map_location='cpu',weights_only=True)
    rows=torch.load(POLY/'STABLE_JOINT32_REFLECTION_V1_ROWS.pt',map_location='cpu',weights_only=True)
    assert rows.shape==(96,257) and len(program['pronoun_ids'])==6
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,forwards=13,sequences=104,rows_shape=list(rows.shape))));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    backend=Bilin18TorchBackend.load('cuda');m=backend.model;v=program['v'].cuda();vf=v.float()
    pairs={n:{k:t.cuda() for k,t in p.items()} for n,p in program['programs'].items()}
    counts=[0,0];checks={};records=[];cache={};mode={'reflect':False}
    def count(_mod,args):counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=13
    def capture(_mod,args,out):cache['x']=args[0].detach().clone();cache['y']=out.detach().clone()
    def reflect(_mod,args):
        if mode['reflect']:
            x=args[0];return (x-2*(x@vf)[...,None]*vf,)
    h1=m.transformer.h[0].attn.register_forward_pre_hook(count)
    h2=m.transformer.h[17].mlp.register_forward_hook(capture)
    h3=m.transformer.h[17].mlp.register_forward_pre_hook(reflect)
    def forward(idx):
        x=F.rms_norm(m.transformer.wte(idx),(1152,));x0=x;v1=None
        for block in m.transformer.h:x,v1=block(x,v1,x0)
        return x
    def logits(h):return 30*torch.tanh(m.lm_head(F.rms_norm(h,(1152,)))/30)
    def ce(z,t):return F.cross_entropy(z.flatten(0,1),t.flatten(),reduction='none').reshape(t.shape)
    def sums(t,mask):return (t*mask).sum(1).double().cpu().tolist()
    try:
        for start in range(0,96,8):
            row=rows[start:start+8].cuda();tg=row[:,1:];h=forward(row[:,:-1]);x=cache['x'];y=cache['y']
            rx=x-2*(x@vf)[...,None]*vf
            # Removing capture prevents overwriting the original native input/output.
            h2.remove();full_y=m.transformer.h[17].mlp(rx);h2=m.transformer.h[17].mlp.register_forward_hook(capture)
            full_h=h-y+full_y;base=logits(h);full=logits(full_h)
            if start==0:
                mode['reflect']=True;ref=logits(forward(row[:,:-1]));mode['reflect']=False
                checks['native_replay_max_abs']=float((ref-full).abs().max())
                checks['native_replay_relative_l2']=float((ref-full).norm()/ref.norm())
            mask=torch.arange(256,device='cuda')[None,:]>=64
            cm=torch.isin(tg,torch.tensor(program['pronoun_ids'],device='cuda')) & mask;nm=(~cm)&mask
            bce=ce(base,tg);fce=ce(full,tg)-bce
            response=full[...,:50257]-base[...,:50257];response-=response.mean(-1,keepdim=True)
            record=dict(row_start=start,class_count=cm.sum(1).cpu().tolist(),nonclass_count=nm.sum(1).cpu().tolist(),
                full_ce_abs=sums(fce.abs(),cm),full_ce_squared=sums(fce.square(),cm),
                full_response_squared=sums(response.square().sum(-1),cm),pairs={})
            for name,p in pairs.items():
                xd=x.double();rd=rx.double()
                products=(xd@p['a'].T)*(xd@p['b'].T);refproducts=(rd@p['a'].T)*(rd@p['b'].T)
                change=(refproducts-products)@p['w'].T
                # Check exact zero-reflection algebra separately from rounded native rx.
                exactrx=xd-2*(xd@v)[...,None]*v
                exact=((exactrx@p['a'].T)*(exactrx@p['b'].T)-products)@p['w'].T
                av,bv=p['a']@v,p['b']@v
                readers=.5*(bv[:,None]*p['a']+av[:,None]*p['b'])-(av*bv)[:,None]*v
                formula=-4*(xd@v)[...,None]*(xd@readers.T@p['w'].T)
                err=float((formula-exact).norm()/exact.norm().clamp_min(1e-30))
                checks['product_delta_max_relative']=max(checks.get('product_delta_max_relative',0.),err)
                z=logits(h+change.float());dz=z[...,:50257]-base[...,:50257];dz-=dz.mean(-1,keepdim=True)
                dc=ce(z,tg)-bce
                removal_ce=ce(logits(h-(products@p['w'].T).float()),tg)-bce
                record['pairs'][name]=dict(response_error_squared=sums((dz-response).square().sum(-1),cm),
                    ce_error_squared=sums((dc-fce).square(),cm),reflection_ce_signed=sums(dc,cm),
                    removal_class_abs=sums(removal_ce.abs(),cm),removal_nonclass_abs=sums(removal_ce.abs(),nm))
            records.append(record)
    finally:h1.remove();h2.remove();h3.remove()
    total=lambda key:sum(sum(r[key]) for r in records)
    nc,nn=total('class_count'),total('nonclass_count');reports={}
    for name in pairs:
        ps=lambda key:sum(sum(r['pairs'][name][key]) for r in records)
        reports[name]=dict(reflection_response_error=(ps('response_error_squared')/max(total('full_response_squared'),1e-30))**.5,
            reflection_ce_error=(ps('ce_error_squared')/max(total('full_ce_squared'),1e-30))**.5,
            removal_class_mean_absolute_ce=ps('removal_class_abs')/max(nc,1),
            removal_nonclass_mean_absolute_ce=ps('removal_nonclass_abs')/max(nn,1))
    live=total('full_ce_abs')/max(nc,1)
    valid=counts==[13,104] and nc>=20 and live>=.05 and checks['native_replay_max_abs']<=1e-3 and checks['native_replay_relative_l2']<=1e-5 and checks['product_delta_max_relative']<=1e-10
    finite=all(__import__('math').isfinite(x) for r in reports.values() for x in r.values());valid=bool(valid and finite)
    out=dict(schema='stable.joint32.reflection.v1',predictions={'pred_a_instrument':valid,
        'pred_b_conditional_sufficiency':bool(valid and all(r['reflection_response_error']<=.20 and r['reflection_ce_error']<=.20 for r in reports.values())),
        'pred_c_selective_removal':bool(valid and all(r['removal_class_mean_absolute_ce']>=.05 and r['removal_nonclass_mean_absolute_ce']<=.01 for r in reports.values()))},
        reports=reports,checks=checks,class_positions=nc,nonclass_positions=nn,full_reflection_class_mean_absolute_ce=live,rows=records,
        wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=counts[0],sequences=counts[1],native_weight_saving=0),
        runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),scope='Conditional two-product reflection/removal on opened corpus rows. Native background retained; no independent extraction, antecedent binding, or OOD claim.')
    atomic_create_json(OUT,out);print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__=='__main__':main()
