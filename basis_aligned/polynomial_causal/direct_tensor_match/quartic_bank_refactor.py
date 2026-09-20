"""Joint shared-product bank refactor; frozen root and analytic Gaussian mean."""
import itertools,json,time
from pathlib import Path
import torch
from fit_projected_quadratic import fit
from gaussian_quartic_mean import quadratic_moments
from arithmetic_dag import DAG
P=Path(__file__).resolve().parent

class Core:
 def __init__(self,T):self.T=T;self.L=torch.eye(T.shape[1],dtype=T.dtype)
 def cross(self,a,b):return torch.einsum('ki,vij,kj->vk',a,self.T,b)

def evaluate(s,x):
 q=((x@s['A'].T)*(x@s['B'].T))@s['bank_writer'].T;i,j=torch.triu_indices(4,4)
 return ((q[:,i]*q[:,j])@s['Z'].T)@s['W'].T+s['constant']

def build(s):
 d=DAG(degree_limit=4);inputs=[d.input(i) for i in range(s['A'].shape[1])];left=[d.linear(zip(inputs,r.tolist())) for r in s['A']];right=[d.linear(zip(inputs,r.tolist())) for r in s['B']];products=[d.product(a,b) for a,b in zip(left,right)];bank=[d.linear(zip(products,r.tolist())) for r in s['bank_writer']];roots=[d.product(bank[i],bank[j]) for i in range(4) for j in range(i,4)];shared=[d.linear(zip(roots,r.tolist())) for r in s['Z']];one=d.constant();out=[d.linear(list(zip(shared,r.tolist()))+[(one,float(c))]) for r,c in zip(s['W'],s['constant'])];return d,out

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.perf_counter();core=torch.load(P/'QUARTIC_BANK_CORE_V1.pt',weights_only=True);source=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);s={k:v.double() for k,v in source['programs'][8].items()};op=Core(core['weighted_core']);norm=op.T.square().sum();records=[];best={};allfits={}
 for width,optimizer,lr,seed in itertools.product([8,12],['adam','muon'],[.005,.03],[0,1]):
  torch.manual_seed(seed);a=torch.randn(width,32)/32**.5;b=torch.randn_like(a)/32**.5;info,(a,b,c)=fit(op,a,b,optimizer,lr,500,.001);features=.5*(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:]);reconstructed=torch.einsum('vk,kij->vij',c,features);error=float((reconstructed-op.T).square().sum()/norm);cancel=float(c.square().sum()/reconstructed.square().sum());row=dict(width=width,optimizer=optimizer,lr=lr,seed=seed,retained_weighted_energy=1-error,cancellation_ratio=cancel,**info);records.append(row);key=(width,optimizer,lr,seed);allfits[key]=dict(a=a,b=b,c=c)
  if width not in best or row['best_objective']<best[width][0]['best_objective']:best[width]=(row,a,b,c)
  print({k:v for k,v in row.items() if k!='history'},flush=True)
 panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];fresh=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];mu=panels[0]['mean'].double();M=panels[0]['covariance'].double();exports={};winners=[];i,j=torch.triu_indices(4,4)
 for width,(row,a,b,c) in best.items():
  A=a@core['input_mapback'];B=b@core['input_mapback'];C=torch.linalg.solve(core['sqrt_root_sensitivity'],c);m,G=quadratic_moments(A,B,mu,M);m=C@m;G=C@G@C.T;rootmean=m[i]*m[j]+G[i,j];constant=source['mean_teacher_gaussian'].double()-s['W']@s['Z']@rootmean;model=dict(A=A,B=B,bank_writer=C,W=s['W'],Z=s['Z'],constant=constant);archive={k:v.float() for k,v in model.items()};loaded={k:v.double() for k,v in archive.items()};d,out=build(loaded);x=panels[0]['rows'][:16].double();replay=float((d.evaluate(out,x)-evaluate(loaded,x)).norm()/evaluate(loaded,x).norm());cost=d.cost(out);assert cost['stored_coefficients']==sum(v.numel() for v in archive.values()) and cost['products']==width+10 and replay<1e-10
  diagnostics=[float((evaluate(loaded,p['rows'].double())-y).norm()/y.norm()) for p,y in zip(panels,targets)];fresherrors=[float((evaluate(loaded,p['rows'].double())-p['targets'].double()).norm()/p['targets'].double().norm()) for p in fresh];winners.append(dict(row,diagnostic_errors=diagnostics,fresh_errors=fresherrors,cost=cost,graph_replay=replay));exports[width]=archive;print('WINNER',width,diagnostics,fresherrors,cost,flush=True)
 eight=next(w for w in winners if w['width']==8);pred=dict(pred_a_energy=eight['retained_weighted_energy']>=.99,pred_b_composed=eight['fresh_errors'][0]<=.18440681,pred_c_cost=eight['cost']['stored_coefficients']<47312 and eight['cost']['products']<26);torch.save(dict(programs=exports,allfits=allfits,teacher_scale=source['teacher_scale']),P/'QUARTIC_BANK_REFACTOR_V1.pt');(P/'QUARTIC_BANK_REFACTOR_V1.json').write_text(json.dumps(dict(records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start,scope='Weight-selected local root-sensitivity bank refactor; full composed quartic and fresh panels evaluated afterward. No empirical selection, no global optimum or stable semantic feature claim.'),indent=2)+'\n');print(pred)
if __name__=='__main__':main()
