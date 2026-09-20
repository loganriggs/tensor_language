#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selective_retention pred_c_prediction
"""Prospective five-source test: new48texts/96sites,19source arms.
12prefix,192double+160native suffix forwards,32gradient+160Hessian-row reverse;96 bounded LPs.
All native generators charged. Null: number prediction hides modal collateral.
"""
import os,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'source_ood_v1_result.json'
SOURCE_SETS=[[i] for i in range(5)]+[[i,j] for i in range(5) for j in range(i+1,5)]
ARMS={'+'.join(map(str,selected)):[int(i in selected) for i in range(5)] for selected in SOURCE_SETS}
ARMS.update(unitB=[0,0,1,1,1],all5=[1,1,1,1,1],null_full=None,null_half=None)
PREDICTIONS=dict(pred_a_instrument='nativefloat64 effectreplay1e-4, FDgradient1%/Hessian15%,symmetry1e-4,LP residual1e-8,designrank15,counts',pred_b_selective_retention='null_full native alignedretention>=80%unitB andmodal<=10% everycell',pred_c_prediction='number10%modal5%number-effect every19arm/cell',pred_d_native_capability='native correctnumber margin positive>=90%everycell')
PLAN=dict(prefix=12,double_suffix=192,native_suffix=160,gradient=32,hessian=160,fits=0,linear_programs=96,amplitudes=ARMS,coefficients_per_context=80,predictions=PREDICTIONS)

def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    from scipy.optimize import linprog
    import tiktoken
    import circuit_fast_screen_producer as producer
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    sys.path.insert(0,str(P));from native_source_observables import source_observables
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for p in model.parameters():p.requires_grad_(False)
    assert all(not b.mlp.config.gated for b in model.transformer.h[11:])
    dec=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder'];axis=torch.tensor(dec['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(dec['threshold'])/float(axis.norm())
    enc=tiktoken.get_encoding('gpt2');encoded=[enc.encode(' '+w) for w in ['can','will','may','might','should','could']];assert all(len(x)==1 for x in encoded);modal=torch.tensor([x[0] for x in encoded],device='cuda').reshape(3,2)
    counts=dict(prefix=0,double_suffix=0,native_suffix=0,gradient=0,hessian=0);checks=[];records=[];replays=[];contexts=[];lp_checks=[];derivative_replay=[]
    for panel,digest in [('opposite','47039b1f0f2d42320ce0e63ae13933de647313ae20a61115fd4ec24bd7526aae'),('congruent','69ee6cf513f7ea2222ef563375ce12aab95b4d910faad2141e619309dabdbca2')]:
        path=P/f'SOURCE_OOD_V1_{panel.upper()}_ROWS.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;rows=json.loads(path.read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            entries=[r for r in rows if r['template']==template];tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda');answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),3,2)],dim=1)
            initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=graph._capture(model,initial,torch,F);counts['prefix']+=1
            for role,key in [('subject','subject_position'),('attractor','control_position')]:
                pos=torch.tensor([r[key] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit;removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float();_,_,_,pe,_=graph._capture(model,removed,torch,F);counts['prefix']+=1
                ds=torch.stack([(pe[i]-pb[i])[batch,pos].float() for i in range(5)],dim=1)
                def evaluate(a):
                    counts['double_suffix']+=1
                    return source_observables(model,raw,x0,first,ds,pos,read,pairs,a)
                def native(a):
                    counts['native_suffix']+=1;x=raw.clone();x[batch,pos]+=torch.einsum('bi,bid->bd',a.float(),ds)
                    for l in range(11,18):
                        b=model.transformer.h[l]
                        if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
                        at,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+at;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
                    logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30);z=logits.gather(1,pairs.reshape(len(x),8)).reshape(len(x),4,2)
                    return (z[:,:,0]-z[:,:,1]).double()
                with torch.enable_grad():
                    a=torch.zeros(len(entries),5,device='cuda',dtype=torch.float64,requires_grad=True);base=evaluate(a);gs=[];hs=[]
                    for o in range(4):
                        g=torch.autograd.grad(base[:,o].sum(),a,create_graph=True,retain_graph=True)[0];counts['gradient']+=1
                        h=torch.stack([torch.autograd.grad(g[:,j].sum(),a,retain_graph=not(o==3 and j==4))[0] for j in range(5)],dim=1);counts['hessian']+=5;gs.append(g.detach());hs.append(h.detach())
                base=base.detach();g=torch.stack(gs,dim=1);h=torch.stack(hs,dim=1);baseline=native(torch.zeros_like(a));direction=torch.tensor([.3,-.2,.7,-.4,.5],device='cuda',dtype=torch.float64).expand(len(a),5)
                check=dict(panel=panel,role=role,template=template,symmetry=float((h-h.transpose(-1,-2)).norm()/h.norm()),finite=[])
                gd=torch.einsum('boi,bi->bo',g,direction);hd=torch.einsum('bi,boij,bj->bo',direction,h,direction)
                for step in [.1,.2]:
                    plus=evaluate(step*direction);minus=evaluate(-step*direction);fd=(plus-minus)/(2*step);fdd=(plus+minus-2*base)/(step**2)
                    check['finite'].append(dict(step=step,gradient_relative=((fd-gd).norm(dim=0)/gd.norm(dim=0).clamp_min(1e-20)).tolist(),hessian_relative=((fdd-hd).norm(dim=0)/hd.norm(dim=0).clamp_min(1e-20)).tolist()))
                checks.append(check)
                candidates=[]
                for gi in g.cpu().numpy():
                    n=-gi[0];M=-gi[1:];scale=np.linalg.norm(M,axis=1);assert min(scale)>1e-20
                    signed=np.sign(n[2:].sum()) or 1.
                    solution=linprog(-signed*n/max(np.linalg.norm(n),1e-20),A_eq=M/scale[:,None],b_eq=np.zeros(3),bounds=[(-1,1)]*5,method='highs')
                    assert solution.success
                    candidates.append(solution.x);lp_checks.append(float(np.max(np.abs(M@solution.x/scale))))
                candidate=torch.tensor(np.array(candidates),device='cuda',dtype=torch.float64)
                contexts.append(dict(panel=panel,role=role,template=template,gradient=g.tolist(),hessian=h.tolist(),candidate_amplitudes=candidate.tolist()))
                for arm,amplitude in ARMS.items():
                    av=torch.tensor(amplitude,device='cuda',dtype=torch.float64).expand(len(a),5) if amplitude is not None else candidate*(.5 if arm=='null_half' else 1.)
                    actual=baseline-native(av);reference=base-evaluate(av);replays.append(float((actual-reference).abs().max()));linear=-torch.einsum('boi,bi->bo',g,av);quadratic=linear-.5*torch.einsum('bi,boij,bj->bo',av,h,av)
                    for family in dict.fromkeys(r['family'] for r in entries):
                        ids=[i for i,r in enumerate(entries) if r['family']==family];y=actual[ids];den=y[:,0].norm().clamp_min(1e-20)
                        records.append(dict(panel=panel,role=role,family=family,arm=arm,target=y.tolist(),quadratic=quadratic[ids].tolist(),linear=linear[ids].tolist(),number_error=float((quadratic[ids,0]-y[:,0]).norm()/den),modal_error=((quadratic[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),linear_modal_error=((linear[ids,1:]-y[:,1:]).norm(dim=0)/den).tolist(),native_modal_ratio=(y[:,1:].norm(dim=0)/den).tolist(),number_norm=float(den),native_capability=float((baseline[ids,0]>0).double().mean())))
    designs=np.array([ARMS['+'.join(map(str,selected))] for selected in SOURCE_SETS]);rank=int(np.linalg.matrix_rank(np.array([[v[i]*v[j] for i in range(5) for j in range(i,5)] for v in designs])));assert rank==15
    a=max(replays)<=1e-4 and max(lp_checks)<=1e-8 and counts=={k:PLAN[k] for k in counts} and all(c['symmetry']<=1e-4 and all(max(f['gradient_relative'])<=.01 and max(f['hessian_relative'])<=.15 for f in c['finite']) for c in checks)
    selected=[]
    for c in records:
        if c['arm']!='null_full':continue
        base=next(x for x in records if x['panel']==c['panel'] and x['role']==c['role'] and x['family']==c['family'] and x['arm']=='unitB')
        y=np.array(c['target'])[:,0];yb=np.array(base['target'])[:,0]
        selected.append(dict(panel=c['panel'],role=c['role'],family=c['family'],aligned_retention=float(y@yb/max(yb@yb,1e-30)),modal_ratio=max(c['native_modal_ratio'])))
    result=dict(plan=PLAN,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(a and all(c['aligned_retention']>=.8 and c['modal_ratio']<=.1 for c in selected)),bool(a and all(c['number_error']<=.1 and max(c['modal_error'])<=.05 for c in records)),bool(all(c['native_capability']>=.9 for c in records))])),selection=selected,max_lp_residual=max(lp_checks),quadratic_design_rank=rank,max_native_effect_replay=max(replays),checks=checks,records=records,contexts=contexts,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_native_effect_replay','seconds']}))
if __name__=='__main__':main()
