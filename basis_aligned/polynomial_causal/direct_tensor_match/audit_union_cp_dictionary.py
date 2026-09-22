"""Exact Gaussian native refits of single/union frozen CP dictionaries."""
import json,time
import torch
from check_union_cp_dictionary import check
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import cross,project_shifted
from quartic_cp_profile import profile
from audit_root_matched_reader import CK
from audit_conditional_residual_accounting import P,SCALE,load
from audit_balanced_shared_followup import scores

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();controls=check()
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double();loc=torch.linalg.solve(S,mu)
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');vocab=state['lm_head.weight'].double();uw=vocab@cache['writer'].double();readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].double()
 teacher=[readers[:,4:].T@w(17,'Down')/SCALE,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left')@S,w(16,'Right')@S]
 projection=project_shifted(teacher,loc,tuple(t[4:].double() for t in cache['projections']['covariance']['zero_projection']))
 programs=[load(f'MIXED_CP_FEATURES_SEED{s}_V1.pt') for s in [1001,1002]];fs=[torch.cat([p['factors'][i] for p,h in programs]) for i in range(4)];fw=[f@S for f in fs];bs=[f@mu for f in fs]
 G=gram_dynamic(fw,bs,fw,bs);print('Union Gram ready',time.monotonic()-start,flush=True)
 parts=[]
 for left in range(0,1024,128):
  parts.append(cross(teacher,loc,projection,[f[left:left+128] for f in fw],[b[left:left+128] for b in bs]));print('Native cross',left,time.monotonic()-start,flush=True)
 X=torch.cat(parts,1);oldx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();oldy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/SCALE;oldpairs=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat']).T
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);newpairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T
 fits=[]
 for name,sl in [('single1001',slice(0,512)),('single1002',slice(512,1024)),('union',slice(0,1024))]:
  g=G[sl,sl];cc=X[:,sl];loss,C=profile(g,cc,ridge=1e-6);normal=float((C@(g+1e-6*torch.eye(len(g),dtype=g.dtype))-cc).norm()/cc.norm());assert normal<1e-8;fits.append((name,sl,C,normal,float(loss)))
 rows=[]
 for panel,x,y,pairs in [('original',oldx,oldy,oldpairs),('opened256',data['rows'].double(),data['target'].double()/SCALE,newpairs)]:
  phi=torch.cat([torch.stack([xx@f.T for f in fs]).prod(0) for xx in x.split(1024)])
  base=phi[:,:512]@(programs[0][0]['coefficients']/SCALE).T
  for name,sl,C,normal,loss in fits:
   pred=base.clone();pred[:,4:]=phi[:,sl]@C.T;assert torch.equal(pred[:,:4],base[:,:4]);assert torch.isfinite(pred).all();s=scores(pred,y,*pairs)
   rows.append(dict(panel=panel,name=name,normal_residual=normal,objective_without_teacher_constant=loss,atom_count=C.shape[1],**s));print(panel,name,s['small_value_rms'],s['small_response_rms'],flush=True)
  for idx,(p,h) in enumerate(programs):
   pred=phi[:,idx*512:(idx+1)*512]@(p['coefficients']/SCALE).T;rows.append(dict(panel=panel,name=f'original_mixed{1001+idx}',sha256=h,**scores(pred,y,*pairs)))
 old=[r for r in rows if r['panel']=='original'];u=next(r for r in old if r['name']=='union');singles=[r for r in old if r['name'].startswith('single')];ratios={k:u[k]/min(r[k] for r in singles) for k in ['small_value_rms','small_response_rms']}
 out=dict(controls=controls,rows=rows,primary_ratios=ratios,pred_primary=all(v<=.85 for v in ratios.values()),seconds=time.monotonic()-start,scope='NativeGaussianweightrefits, fixedbankunion1024vs512, outputs4–15only; no factorlearning/textlabelsfitted. Primary originalopenedpanel; secondaryopened256documents. More computation, not compression/adoption. Same ridge1e-6, numericalnormal checks; not certified unregularized optimum.')
 (P/'UNION_CP_DICTIONARY_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
