#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_unit_quadratic pred_c_signed_quadratic
"""Float64 reference quadratic in3 B-source amplitudes, native epsilon/RoPE.
Opened96rows,2roles.24prefix,144suffix forward,16gradient+48Hessian-row reverse.
No fitting. All native weights/ports and derivative generation charged.
Null: small Boolean triple interaction does not imply a useful quadratic jet.
"""
import os,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'semantic_source_jet_v2r1_result.json'
PREDICTIONS=dict(pred_a_instrument='prior unit replay1e-4; Hessian symmetry relative1e-4; central gradient1% and Hessian15% at h=.1,.2',pred_b_unit_quadratic='unit-amplitude quadratic effect error<=10% every cell',pred_c_signed_quadratic='negative/mixed/doubled quadratic effect error<=10% every cell')
PLAN=dict(prefix=24,suffix=144,gradient_reverse=16,hessian_reverse=48,fits=0,coefficients_per_context=9,predictions=PREDICTIONS,amplitudes={'unit':[1,1,1],'negative':[-1,-1,-1],'mixed':[1,-1,1],'double':[2,2,2]})
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    assert not OUT.exists();torch.set_num_threads(4);torch.set_grad_enabled(False);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm())
    assert all(not b.mlp.config.gated for b in model.transformer.h[11:])
    eps=torch.finfo(torch.float32).eps
    prior=json.loads((A/'semantic_port_pairs_fresh_v1_result.json').read_text());counts=dict(prefix=0,suffix=0,gradient_reverse=0,hessian_reverse=0);records=[];checks=[];replay=[]
    for panel,digest in [('opposite','1869db8a18d4c89c28fa81a6bb57a19497e6f8692831eb4391f5d23c84556d73'),('congruent','bfba23ab066beb14f5cd469edce137db7ec1e8ccc26bfc36b66f896426b17511')]:
        path=P/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;rows=json.loads(path.read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda')
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float()
                _,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                directions=torch.stack([(pe[i]-pb[i])[batch,pos].float() for i in [2,3,4]],dim=1)
                raw64=raw.double();x064=x0.double();first64=first.double();directions64=directions.double()
                def norm(x):return F.rms_norm(x,(x.shape[-1],),eps=eps)
                def evaluate(amplitudes):
                    counts['suffix']+=1
                    edit=torch.zeros_like(raw64);edit[batch,pos]=torch.einsum('bi,bid->bd',amplitudes.double(),directions64)
                    x=raw64+edit
                    for l in range(11,18):
                        b=model.transformer.h[l]
                        if l>11:x=b.lambdas[0].double()*x+b.lambdas[1].double()*x064
                        att=b.attn;z=norm(x);bs,t,d=z.shape;heads=att.n_head;hd=att.head_dim
                        projections=[F.linear(z,getattr(att,name).weight.double()).reshape(bs,t,heads,hd) for name in ['c_q','c_k','c_q2','c_k2','c_v']]
                        q,k,q2,k2,value=projections
                        cos,sin=att.rotary(torch.zeros(bs,t,heads,hd,device=x.device,dtype=torch.float32));cos=cos.double();sin=sin.double()
                        def rotate(y):
                            y=norm(y);y1,y2=y[...,:hd//2],y[...,hd//2:]
                            return torch.cat([y1*cos+y2*sin,-y1*sin+y2*cos],dim=-1)
                        q,k,q2,k2=[rotate(y) for y in [q,k,q2,k2]]
                        scores=torch.einsum('bthd,bshd->bhts',q,k)/hd;scores2=torch.einsum('bthd,bshd->bhts',q2,k2)/hd
                        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril();pattern=(scores*scores2).masked_fill(~mask,0.)
                        value=(1-att.lamb.double())*value+att.lamb.double()*first64.reshape_as(value)
                        at=torch.einsum('bhts,bshd->bthd',pattern,value).reshape(bs,t,d)
                        x=x+F.linear(at,att.c_proj.weight.double());z=norm(x)
                        x=x+F.linear(F.linear(z,b.mlp.Left.weight.double())*F.linear(z,b.mlp.Right.weight.double()),b.mlp.Down.weight.double(),b.mlp.Down_bias.double())
                    logits=30*torch.tanh(F.linear(norm(x[batch,read]),model.lm_head.weight.double())/30);z=logits.gather(1,answers)
                    return z[:,0]-z[:,1]
                with torch.enable_grad():
                    a=torch.zeros(len(entries),3,device='cuda',dtype=torch.float64,requires_grad=True);base=evaluate(a)
                    g=torch.autograd.grad(base.sum(),a,create_graph=True)[0];counts['gradient_reverse']+=1
                    h=torch.stack([torch.autograd.grad(g[:,j].sum(),a,retain_graph=j<2)[0] for j in range(3)],dim=1);counts['hessian_reverse']+=3
                base=base.detach().double();g=g.detach().double();h=h.detach().double()
                sym=float((h-h.transpose(-1,-2)).norm()/h.norm().clamp_min(1e-20));direction=torch.tensor([.7,-.4,.5],device='cuda').expand(len(entries),3)
                gd=(g*direction).sum(-1);hd=torch.einsum('bi,bij,bj->b',direction.double(),h,direction.double())
                finite=[]
                for step in [.1,.2]:
                    plus=evaluate(step*direction).double();minus=evaluate(-step*direction).double()
                    fd=(plus-minus)/(2*step);fdd=(plus+minus-2*base)/(step**2)
                    finite.append(dict(step=step,gradient_relative=float((fd-gd).norm()/gd.norm().clamp_min(1e-20)),hessian_relative=float((fdd-hd).norm()/hd.norm().clamp_min(1e-20))))
                checks.append(dict(panel=panel,role=role,template=template,hessian_symmetry=sym,finite=finite))
                for arm,amp in PLAN['amplitudes'].items():
                    av=torch.tensor(amp,device='cuda',dtype=torch.float32).expand(len(entries),3);actual=base-evaluate(av).double();linear=-(g*av).sum(-1);quadratic=linear-.5*torch.einsum('bi,bij,bj->b',av.double(),h,av.double())
                    for family in dict.fromkeys(r['family'] for r in entries):
                        indices=[i for i,r in enumerate(entries) if r['family']==family];y=actual[indices];den=y.norm().clamp_min(1e-20)
                        if arm=='unit':replay.append(float((y-torch.tensor(prior['data'][panel][role]['B'][family]['effects'],device='cuda')).abs().max()))
                        records.append(dict(panel=panel,role=role,family=family,arm=arm,target=y.tolist(),linear=linear[indices].tolist(),quadratic=quadratic[indices].tolist(),gradient=g[indices].tolist(),hessian=h[indices].tolist(),quadratic_error=float((quadratic[indices]-y).norm()/den),linear_error=float((linear[indices]-y).norm()/den),target_norm=float(den)))
    a=max(replay)<=1e-4 and counts==dict(prefix=24,suffix=144,gradient_reverse=16,hessian_reverse=48) and all(c['hessian_symmetry']<=1e-4 and all(f['gradient_relative']<=.01 and f['hessian_relative']<=.15 for f in c['finite']) for c in checks)
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and all(c['quadratic_error']<=.1 for c in records if c['arm']=='unit')),bool(a and all(c['quadratic_error']<=.1 for c in records if c['arm']!='unit'))])),max_replay_absolute=max(replay),checks=checks,records=records,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_replay_absolute','seconds']}))
if __name__=='__main__':main()
