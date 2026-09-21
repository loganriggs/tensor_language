#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_values pred_c_native_gain pred_d_covariance
"""Two graph widths, three input metrics, two starts; exact output solves.
Widths367/560mixed plus32private squares:399/592products,896198/1342028floats.
Native isotropic alpha0/0.5/1 mixture (alpha0 is covariance-only).
1500cosine Adam steps .01, parent-extended and random initializations.
Primary width560 alpha.5, winner selected by fitting objective.
pred_a_instrument implicit/dense and exportrelative<1e-8, physicalprice exact.
pred_b_values all3opened<=15% and<=1.10separate768baseline.
pred_c_native_gain primarynativeequalpairerror<=.8samewidthalpha0control.
pred_d_covariance primarycovarianceerror<=1.10samewidthalpha0control.
Null: covariance-isotropic fidelity conflict persists or graph capacity insufficient.
Nativez/h remain supplied, no nativeforwards; no fair-cost adoption claim vs512.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'DUAL_GEOMETRY_SOURCE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from dual_geometry_source_metric import DualGeometrySourceMetric
 from global_mixed_source_graph import export,score,source_reads
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 path=P/'DUAL_GEOMETRY_SOURCE_V1.json';assert not path.exists()
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').read_text());parent=torch.load(P/'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True)[meta['winners']['1']]
 S=torch.linalg.inv(d['inverse_root']).cuda();Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);T=(Q/d['scales'][:,None,None]).cuda()
 assert float((torch.einsum('ai,oij,jb->oab',S,T,S)-d['teacher'].cuda()).norm()/d['teacher'].norm())<1e-10
 records=[];programs={}
 for width in plan['widths']:
  for alpha in plan['alphas']:
   metric=DualGeometrySourceMetric(T,S,alpha)
   for seed in plan['seeds']:
    g=torch.Generator().manual_seed(10100+seed);params=[]
    for name,n in [('left_reader',width),('right_reader',width),('square_reader',32)]:
     a=torch.randn((1152,n),dtype=T.dtype,generator=g)
     if seed==0:a[:,:parent[name].shape[1]]=parent[name]
     a=a.cuda();a/=a.norm(dim=0);a.requires_grad_();params.append(a)
    L,R,V=params;optimizer=torch.optim.Adam(params,lr=plan['rate']);best=float('inf');history=[]
    for step in range(plan['steps']+1):
     optimizer.zero_grad();loss,W,v=metric.loss(L,R,V);value=float(loss.detach());assert math.isfinite(value)
     if value<best:best=value;state=[a.detach().clone() for a in [L,R,V,W,v]];beststep=step
     if step%500==0:history.append(dict(step=step,objective=value));print(width,alpha,seed,step,value,flush=True)
     if step==plan['steps']:break
     optimizer.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();optimizer.step()
     with torch.no_grad():
      for a in params:a.div_(a.norm(dim=0))
    with torch.no_grad():
     l,r,s,w,v=state;replay=abs(float(metric.explicit(l,r,s,w,v))-best)
     raw=torch.einsum('ir,or,jr->oij',l,w,r);hat=(raw+raw.transpose(-1,-2))/2;hat[5]+=(s*v)@s.T
     E=hat-T;native=((E.square().sum((-1,-2)).reshape(3,2).sum(1)/metric.energies[0]).mean()).sqrt();cov=((torch.einsum('ai,oij,jb->oab',S,E,S).square().sum((-1,-2)).reshape(3,2).sum(1)/metric.energies[1]).mean()).sqrt()
     private=torch.zeros((6,32),dtype=T.dtype,device='cuda');private[5]=v
     big=export((S@torch.cat([l,s],1)).cpu(),(S@torch.cat([r,s],1)).cpu(),torch.cat([w,private],1).cpu(),d)
     p={k:a.clone() for k,a in big.items()}
     for name in ('left_reader','right_reader'):p[name]=big[name][:,:width].clone()
     p['product_weights']=big['product_weights'][:width].clone();p['square_reader']=big['left_reader'][:,width:].clone();p['square_weights']=big['product_weights'][width:,5].clone();p['square_output']=torch.tensor(5)
     actual=source_reads(d['z'][:32],p);actual[:,5]+=(d['z'][:32]@p['square_reader']).square()@p['square_weights'];reference=source_reads(d['z'][:32],big);execution=float((actual-reference).norm()/reference.norm())
     price=sum(a.numel() for a in p.values() if a.is_floating_point());assert price==2310*width+48428
     key=f'{width}_{alpha}_seed{seed}';programs[key]=p;row=dict(key=key,width=width,alpha=alpha,seed=seed,objective=best,best_step=beststep,dense_replay=replay,execution_replay=execution,native_isotropic_equal_pair_error=float(native),calibration_shaped_error=float(cov),source_products=width+32,stored_floats=price,history=history,**score(big,d));records.append(row);print('RECORD',json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
 winners={f'{width}_{alpha}':min((r for r in records if r['width']==width and r['alpha']==alpha),key=lambda r:r['objective'])['key'] for width in plan['widths'] for alpha in plan['alphas']};selected={k:next(r for r in records if r['key']==v) for k,v in winners.items()};primary=selected['560_0.5'];control=selected['560_0']
 pred=dict(pred_a_instrument=max(max(r['dense_replay'],r['execution_replay']) for r in records)<1e-8,pred_b_values=all(x<=.15 and x<=1.1*y for x,y in zip(primary['per_mode_errors'],plan['scalar_baseline'])),pred_c_native_gain=primary['native_isotropic_equal_pair_error']<=.8*control['native_isotropic_equal_pair_error'],pred_d_covariance=primary['calibration_shaped_error']<=1.1*control['calibration_shaped_error'])
 torch.save(programs,P/'DUAL_GEOMETRY_SOURCE_PROGRAMS_V1.pt');path.write_text(json.dumps(dict(plan=plan,records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
