"""Native weight-derived correction screen; no tensor expansion or label fitting."""
import json,time
import torch
from output_specific_correction import project,controls
from noncentral_gaussian_cp import project_shifted
from audit_root_matched_reader import CK
from audit_conditional_residual_accounting import P,SCALE,load
from audit_balanced_shared_followup import scores

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls()
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double();loc=torch.linalg.solve(S,mu)
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=cache['writer'].double();vocab=state['lm_head.weight'].double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].double()
 teacher=[readers[:,4:].T@w(17,'Down')/SCALE,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left')@S,w(16,'Right')@S]
 zero=tuple(t[4:].double() for t in cache['projections']['covariance']['zero_projection']);native=project_shifted(teacher,loc,zero);print('Native shifted projections ready',time.monotonic()-start,flush=True)
 oldx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();oldy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/SCALE
 oldpairs=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat']).T
 fresh=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);newx=fresh['rows'].double();newy=fresh['target'].double()/SCALE;newpairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T
 panels=[('original',oldx,oldy,oldpairs),('opened256',newx,newy,newpairs)];rows=[]
 for seed in [1001,1002]:
  p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');C=p['coefficients']/SCALE;fs=p['factors'];parent=project(C[4:], [a@S for a in fs],[a@mu for a in fs]);m,l,Q=[a-b for a,b in zip(native,parent)]
  ev,V=torch.linalg.eigh(Q);ix=ev.abs().argsort(dim=1,descending=True);ev=ev.gather(1,ix);V=V.gather(2,ix[:,None,:].expand(-1,1152,-1));print('Residual eigensystem',seed,time.monotonic()-start,flush=True)
  for name,x,y,pairs in panels:
   z=torch.linalg.solve(S,(x-mu).T).T;pv=torch.cat([torch.stack([xx@a.T for a in fs]).prod(0)@C.T for xx in x.split(2048)]);base=scores(pv,y,*pairs)
   affine=m+z@l.T
   # Compute all projections once; reuse their prefix sums across ranks.
   corrections={rank:affine.clone() for rank in [0,8,16,32,1152]}
   for g in range(12):
    zz=z@V[g];terms=(zz.square()-1)*ev[g]
    for rank in [8,16,32,1152]:corrections[rank][:,g]+=terms[:,:rank].sum(1)
   for rank,correction in corrections.items():
    pred=pv.clone();pred[:,4:]+=correction;assert torch.equal(pred[:,:4],pv[:,:4]);s=scores(pred,y,*pairs)
    # Original-coordinate compiled replay, no dense whitened runtime needed.
    if name=='original' and rank==16:
     readers=torch.linalg.solve(S.T,V.reshape(-1,1152,1152)[:,:,:rank]).transpose(1,2)
     directions=readers.reshape(-1,1152);bias=-(directions@mu);linear=torch.linalg.solve(S.T,l.T).T;const=m-linear@mu-ev[:,:rank].sum(1)
     cp=((x[:64]@directions.T+bias).reshape(64,12,rank).square()*ev[:,:rank]).sum(-1)+x[:64]@linear.T+const
     replay=float((cp-correction[:64]).norm()/correction[:64].norm());assert replay<1e-8
    else:replay=None
    row=dict(seed=seed,panel=name,rank=rank,baseline=base,corrected=s,small_value_ratio=s['small_value_rms']/base['small_value_rms'],small_response_ratio=s['small_response_rms']/base['small_response_rms'],added_products=12*rank,added_stored_coefficients=12*((rank+1)*1152+2*rank+1),compiled_replay=replay,parent_sha256=h);rows.append(row);print(seed,name,rank,row['small_value_ratio'],row['small_response_ratio'],flush=True)
 primary=[r for r in rows if r['panel']=='original' and r['rank']==16]
 result=dict(controls=checks,rows=rows,pred_primary=all(r['small_value_ratio']<=.85 and r['small_response_ratio']<=.85 for r in primary),seconds=time.monotonic()-start,scope='Exact weight-derived Gaussian Hermite residual correction on outputs4–15 only; frozenCPparents, no textlabel fitting. All evaluation panels opened. Rank16primary; rank0affine andfull1152diagnostic. Nonhomogeneous local approximation, no fullmodel/OOD/semantic claim.')
 (P/'OUTPUT_SPECIFIC_CORRECTION_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
