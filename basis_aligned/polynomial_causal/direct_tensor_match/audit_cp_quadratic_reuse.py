import json,time
import torch
from quadratic_reuse_moments import gram,controls
from noncentral_gaussian_cp import affine_moment
from cp_linear_reuse import compile_program,evaluate
from quartic_product_reassociation import compile_products
from audit_conditional_residual_accounting import P,SCALE,load
from audit_balanced_shared_followup import scores

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double();oldx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();oldy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/SCALE;oldpair=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat']).T;data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);pairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T;rows=[];proposals=[]
 for seed in [1001,1002]:
  p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');fs=p['factors'];C=p['coefficients']/SCALE;F=[a@S for a in fs];b=[a@mu for a in fs];a=torch.cat([F[0],F[2]]);bb=torch.cat([F[1],F[3]]);alpha=torch.cat([b[0],b[2]]);beta=torch.cat([b[1],b[3]]);G=gram(a,bb,alpha,beta);mean=(a*bb).sum(1)+alpha*beta;K=G-mean[:,None]*mean[None,:];norm=K.diag().clamp_min(1e-30).sqrt();cos=(K/(norm[:,None]*norm[None,:])).abs();cos.fill_diagonal_(-1);target=cos.argmax(1);ix=torch.arange(1024);atoms=ix%512;half=ix//512;old=[F[j][atoms] for j in range(4)];ob=[b[j][atoms] for j in range(4)];new=[];nb=[]
  for j in range(4):
   replacement=a[target] if j%2==0 else bb[target];offset=alpha[target] if j%2==0 else beta[target];new.append(torch.where((half==j//2)[:,None],replacement,old[j]));nb.append(torch.where(half==j//2,offset,ob[j]))
  oo=affine_moment(old+old,ob+ob);on=affine_moment(old+new,ob+nb);nn=affine_moment(new+new,nb+nb);scales=on/nn.clamp_min(1e-30);errors=(oo-on.square()/nn.clamp_min(1e-30)).clamp_min(0)*C.square().sum(0)[atoms];selected=[];used=set();removed=set();retained=set()
  for i in errors.argsort().tolist():
   j=int(target[i]);atom=i%512
   if atom in used or i in retained or j in removed:continue
   selected.append((i,j,float(scales[i])));used.add(atom);removed.add(i);retained.add(j)
   if len(selected)==128:break
  assert len(selected)==128;proposals.append(dict(seed=seed,parent_sha256=h,selected=selected));base=compile_program(fs,C)
  for count in [32,64,128]:
   nf=[f.clone() for f in fs];nc=C.clone()
   for i,j,scale in selected[:count]:
    for slot in [0,1]:nf[2*(i//512)+slot][i%512]=fs[2*(j//512)+slot][j%512]
    nc[:,i%512]*=scale
   q=compile_program(nf,nc);dag,out,meta=compile_products(q['indices'],q['coefficients']);assert len(q['bank'])==2048-2*count
   xx=oldx[:64];raw=torch.cat([(xx@fs[0].T)*(xx@fs[1].T),(xx@fs[2].T)*(xx@fs[3].T)],dim=1);edited=raw.clone()
   for i,j,scale in selected[:count]:edited[:,i]=raw[:,j]
   direct=(edited[:,:512]*edited[:,512:])@nc.T;replay=float((evaluate(q,xx)-direct).norm()/direct.norm());assert replay<1e-10
   for panel,x,y,pair in [('original',oldx,oldy,oldpair),('opened256',data['rows'].double(),data['target'].double()/SCALE,pairs)]:
    pred=torch.cat([evaluate(q,xx) for xx in x.split(1024)]);parent=torch.cat([evaluate(base,xx) for xx in x.split(1024)]);e=pred-parent;delta=parent[pair[1]]-parent[pair[0]];de=e[pair[1]]-e[pair[0]];fv=(e.square().sum(0)/parent.square().sum(0)).sqrt();fr=(de.square().sum(0)/delta.square().sum(0)).sqrt();row=dict(compiled_quadratic_replay=replay,seed=seed,count=count,panel=panel,unique_readers=len(q['bank']),stored_floats=q['bank'].numel()+nc.numel()+1152*16,products=meta['cost']['products'],parent_value_error=float(e.norm()/parent.norm()),parent_response_error=float(de.norm()/delta.norm()),feature_parent_value_errors=fv.tolist(),feature_parent_response_errors=fr.tolist(),native=scores(pred,y,*pair));rows.append(row);print(seed,count,panel,row['parent_value_error'],row['parent_response_error'],max(fr.tolist()),flush=True)
 primary=[r for r in rows if r['count']==128 and r['panel']=='original'];passed=all(r['parent_value_error']<=.01 and r['parent_response_error']<=.01 and max(r['feature_parent_value_errors']+r['feature_parent_response_errors'])<=.05 for r in primary)
 (P/'CP_QUADRATIC_REUSE_V1.json').write_text(json.dumps(dict(controls=checks,rows=rows,proposals=proposals,pred_primary=passed,seconds=time.monotonic()-start,scope='Quadratic replacements selected by calibration-Gaussian localquartic editenergy, no label fitting. Combined errors measured on opened panels, not native adoption. Same output readouts up to peratom scalar; price actualshared reader/product graph.'),indent=2)+'\n')
if __name__=='__main__':main()
