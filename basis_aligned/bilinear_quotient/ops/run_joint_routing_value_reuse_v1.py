#!/usr/bin/env python3
# BQGATE:0bodyforwards;8192coefficient+4096diagonalprobes;batch64;240sec.
"""pred_a replay<=1e-10/whitening<=1e-8/runtime<=180;
pred_b >=4 modes train>=.95/testcos>=.9 bothpositions;
pred_c >=4same modes native-private-gate relativeerror<=.1 bothdirections/positions.
Null: no substantial shared pair at this frozen weight-only screen.
"""
import os,sys,json,time,signal,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary,EPS
from joint_routing_value_polynomial_v1 import contract
from coefficient_canonical_reuse_v1 import fit_pair,cosines
STEM='JOINT_ROUTING_VALUE_REUSE_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'COEFFICIENT_CANONICAL_REUSE_V1_CONTROL.json').read_text());assert control['heldout_relative_error']<=1e-10
    poly=json.loads((P/'JOINT_ROUTING_VALUE_POLYNOMIAL_V1_CONTROL.json').read_text());assert max([v for k,v in poly.items() if k!='gradient_errors']+poly['gradient_errors'])<=1e-10
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('0 body forwards; pooled weight CCA and independent coefficient/native-gate tests');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(240);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def w(k):return sd[k].double().cuda()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mix=float(sd['transformer.h.17.attn.lamb']);v=torch.cat([(1-mix)*w('transformer.h.17.attn.c_v.weight'),mix*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    identity=torch.eye(1152,device='cuda',dtype=torch.float64).reshape(1152,9,128);qr=rotary(8,128).cuda();maps={}
    for pos in (0,7):
        rotation=qr.T@rotary(pos,128).cuda()
        maps[pos]=[torch.cat([torch.einsum('ab,hbd->had',rotation,k),torch.zeros_like(k)],-1) for k in (k1,k2)]
    def coefficient_features(seed):
        gen=torch.Generator(device='cuda').manual_seed(seed);parts={}
        for pos in (0,7):
            rows=[];ka,kb=maps[pos]
            for _ in range(32):
                q=torch.randint(0,2,(64,2,1152),device='cuda',generator=gen).double()*2-1
                s=torch.randint(0,2,(64,3,2304),device='cuda',generator=gen).double()*2-1
                rows.append(contract(q,s,q1,ka,q2,kb,v,identity))
            parts[pos]=torch.cat(rows)
        return parts
    train=coefficient_features(73300);x=torch.cat(list(train.values()));cov=x.T@x/len(x)
    fits=[]
    for h,j in itertools.combinations(range(9),2):
        hs=slice(128*h,128*(h+1));js=slice(128*j,128*(j+1))
        a,b,s=fit_pair(cov[hs,hs],cov[js,js],cov[hs,js]);fits.append((float(s.mean()),h,j,a,b,s))
    best=max(fits,key=lambda z:(z[0],-z[1],-z[2]));score,h,j,a,b,sv=best
    hs=slice(128*h,128*(h+1));js=slice(128*j,128*(j+1));eye=torch.eye(8,device='cuda')
    whitening=max(float((a.T@cov[hs,hs]@a-eye).norm()),float((b.T@cov[js,js]@b-eye).norm()))
    tests=coefficient_features(73301);projections={};reports=[];eligible=sv>=.95
    for pos,z in tests.items():
        aa=z[:,hs]@a;bb=z[:,js]@b;c=cosines(aa,bb);eligible=eligible&(c>=.9)
        projections[str(pos)]={'a':aa.cpu(),'b':bb.cpu()};reports.append(dict(position=pos,coefficient_cosines=c.tolist()))
    coefficient_eligible=eligible.clone();gen=torch.Generator(device='cuda').manual_seed(73302);checks=[];native={}
    for report in reports:
        pos=report['position'];ka,kb=maps[pos];sr=rotary(pos,128).cuda();rows=[]
        for _ in range(32):
            q=F.rms_norm(torch.randn(64,1152,device='cuda',dtype=torch.float64,generator=gen),(1152,),eps=EPS)
            s=torch.cat([F.rms_norm(torch.randn(q.shape,device='cuda',dtype=q.dtype,generator=gen),(1152,),eps=EPS) for _ in range(2)],-1)
            scores=[];factors=[]
            for qm,km in ((q1,k1),(q2,k2)):
                qa=torch.einsum('ni,hki->nhk',q,qm);ks=torch.einsum('ni,hki->nhk',s[:,:1152],km)
                factors.append(((qa.square().mean(-1)+EPS)*(ks.square().mean(-1)+EPS)).sqrt())
                scores.append(((F.rms_norm(qa,(128,),eps=EPS)@qr.T)*(F.rms_norm(ks,(128,),eps=EPS)@sr.T)).sum(-1)/128)
            gates=1/(128**2*factors[0]*factors[1]);value=torch.einsum('ni,hki->nhk',s,v);direct=scores[0][...,None]*scores[1][...,None]*value
            raw=contract(q[:,None].expand(-1,2,-1),s[:,None].expand(-1,3,-1),q1,ka,q2,kb,v,identity).reshape(64,9,128)
            checks.append(float((raw*gates[...,None]-direct).norm()/direct.norm()))
            aa=raw[:,h]@a;bb=raw[:,j]@b
            rows.append(torch.stack([aa*gates[:,h,None],bb*gates[:,h,None],bb*gates[:,j,None],aa*gates[:,j,None]],1).cpu())
        z=torch.cat(rows);native[str(pos)]=z
        error_a=((z[:,1]-z[:,0]).square().sum(0)/z[:,0].square().sum(0)).sqrt()
        error_b=((z[:,3]-z[:,2]).square().sum(0)/z[:,2].square().sum(0)).sqrt()
        eligible=eligible&((error_a<=.1)&(error_b<=.1)).cuda();report.update(native_error_a=error_a.tolist(),native_error_b=error_b.tolist())
    seconds=time.perf_counter()-tic
    result={'pred_a':max(checks)<=1e-10 and whitening<=1e-8 and seconds<=180,'pred_b':int(coefficient_eligible.sum())>=4,'pred_c':int(eligible.sum())>=4}
    torch.save(dict(pair=[h,j],readers=[a.cpu(),b.cpu()],covariance=cov.cpu(),coefficient_projections=projections,native_projections=native),artifact)
    result.update(pair=[h,j],train_correlations=sv.tolist(),all_pair_mean_correlations=[dict(pair=[z[1],z[2]],mean=z[0]) for z in fits],reports=reports,coefficient_passing_modes=int(coefficient_eligible.sum()),joint_passing_modes=int(eligible.sum()),whitening_error=whitening,native_replay_error=max(checks),execution_seconds=seconds,artifact_sha=digest(artifact),source_shas=binding)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','all_pair_mean_correlations')}),flush=True)
if __name__=='__main__':main()
