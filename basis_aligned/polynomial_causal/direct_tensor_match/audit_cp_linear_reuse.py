"""Fixed-bank reader reuse with downstream Gaussian pricing; opened diagnostics."""
import json,time
import torch
from cp_linear_reuse import compile_program,evaluate
from check_cp_linear_reuse import controls
from noncentral_gaussian_cp import affine_moment
from audit_conditional_residual_accounting import P,SCALE,load
from audit_balanced_shared_followup import scores

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double()
 oldx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();oldy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/SCALE;oldpair=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat']).T
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);newpair=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T;rows=[];stats=[]
 for seed in [1001,1002]:
  p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');fs=p['factors'];C=p['coefficients']/SCALE;flat=torch.cat(fs);F=flat@S;b=flat@mu;M=F@F.T+b[:,None]*b[None,:];norm=M.diag().clamp_min(1e-30).sqrt();cos=M/(norm[:,None]*norm[None,:]);cos.fill_diagonal_(0);target=cos.abs().argmax(1);near=cos.abs().max(1).values
  sources=torch.arange(2048);atoms=sources%512;slots=sources//512
  old=[F[j*512+atoms] for j in range(4)];ob=[b[j*512+atoms] for j in range(4)];new=[torch.where((slots==j)[:,None],F[target],old[j]) for j in range(4)];nb=[torch.where(slots==j,b[target],ob[j]) for j in range(4)]
  oo=affine_moment(old+old,ob+ob);on=affine_moment(old+new,ob+nb);nn=affine_moment(new+new,nb+nb);beta=on/nn.clamp_min(1e-30);err=(oo-2*beta*on+beta.square()*nn).clamp_min(0);energy=err*C.square().sum(0)[atoms]
  order=energy.argsort();selected=[];usedatoms=set();removed=set();retained=set()
  for i in order.tolist():
   j=int(target[i]);a=i%512
   if a in usedatoms or i in retained or j in removed:continue
   selected.append((i,j,float(beta[i])));usedatoms.add(a);removed.add(i);retained.add(j)
   if len(selected)==128:break
  assert len(selected)==128
  baseprog=compile_program(fs,C);stats.append(dict(seed=seed,original_unique_readers=len(baseprog['bank']),nearest_cosine_quantiles=torch.quantile(near,torch.tensor([0,.5,.9,1.],dtype=near.dtype)).tolist(),selected=selected))
  for count in [32,64,128]:
   nf=[f.clone() for f in fs];nc=C.clone()
   for i,j,scale in selected[:count]:nf[i//512][i%512]=flat[j];nc[:,i%512]*=scale
   prog=compile_program(nf,nc);assert len(prog['bank'])<=2048-count
   for panel,x,y,pairs in [('original',oldx,oldy,oldpair),('opened256',data['rows'].double(),data['target'].double()/SCALE,newpair)]:
    def predict(p):return torch.cat([evaluate(p,xx) for xx in x.split(2048)])
    pred=predict(prog);parent=predict(baseprog);direct=torch.stack([x[:64]@f.T for f in nf]).prod(0)@nc.T;replay=float((evaluate(prog,x[:64])-direct).norm()/direct.norm());assert replay<1e-10
    refdelta=parent[pairs[1]]-parent[pairs[0]];delta=pred[pairs[1]]-pred[pairs[0]];row=dict(seed=seed,count=count,panel=panel,unique_readers=len(prog['bank']),stored_floats=prog['bank'].numel()+nc.numel()+1152*16,stored_indices=prog['indices'].numel(),variable_products=1536,parent_value_error=float((pred-parent).norm()/parent.norm()),parent_response_error=float((delta-refdelta).norm()/refdelta.norm()),native=scores(pred,y,*pairs),compiled_replay=replay);rows.append(row);print(seed,count,panel,row['parent_value_error'],row['parent_response_error'],flush=True)
 primary=[r for r in rows if r['count']==128 and r['panel']=='original'];result=dict(controls=checks,rows=rows,proposal_statistics=stats,pred_primary=all(r['parent_value_error']<=.01 and r['parent_response_error']<=.01 for r in primary),seconds=time.monotonic()-start,scope='Exact signedreader compilation; approximate readermerges selectedwithoutnativeevaluationlabels using downstreamGaussianeditenergy. No refit, openedpanels, no semantic/OOD/adoption. Local editenergy ignores cross terms between simultaneous edits; full errors evaluated explicitly.')
 (P/'CP_LINEAR_REUSE_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
