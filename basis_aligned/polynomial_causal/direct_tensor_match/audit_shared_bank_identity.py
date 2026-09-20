"""Restart feature identity with noncentral Gaussian mean explicitly separated."""
import itertools,json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from audit_bank_function_metric import embedding
from noncentral_bank_refit import features
P=Path(__file__).resolve().parent

def compare(f,g,m):
 E=embedding(features(f['a'],f['b']),m);F=embedding(features(g['a'],g['b']),m);rows={}
 for name,a,b in [('full',E,F),('centered',E[:,:-1],F[:,:-1])]:
  cos=(a/a.norm(dim=1,keepdim=True))@(b/b.norm(dim=1,keepdim=True)).T;i,j=linear_sum_assignment(-cos.abs().numpy());matched=cos[i,j].abs();rows[name+'_feature_mean_cosine']=float(matched.mean());rows[name+'_feature_minimum_cosine']=float(matched.min());x=f['c']@a;y=g['c']@b;rows[name+'_bank_function_cosine']=float((x*y).sum()/(x.norm()*y.norm()))
 return rows

def main():
 torch.set_num_threads(1);fits=torch.load(P/'NONCENTRAL_BANK_REFIT_V1.pt',weights_only=True)['allfits'];core=torch.load(P/'QUARTIC_BANK_CORE_V1.pt',weights_only=True);mu=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['mean'].double();m=core['input_mapback']@mu;chosen={k:f for k,f in fits.items() if k[0]==8};rows=[]
 for (ka,a),(kb,b) in itertools.combinations(chosen.items(),2):rows.append(dict(first=list(ka),second=list(kb),**compare(a,b,m)))
 f=next(iter(chosen.values()));perm=torch.arange(7,-1,-1);scale=torch.linspace(-2.,-1.,8,dtype=torch.float64);g=dict(a=f['a'][perm]*scale[:,None],b=f['b'][perm],c=f['c'][:,perm]/scale);control=compare(f,g,m);assert min(control.values())>1-1e-10
 out=dict(pairs=rows,gauge_control=control,minimum_centered_feature_cosine=min(r['centered_feature_minimum_cosine'] for r in rows),minimum_centered_bank_function_cosine=min(r['centered_bank_function_cosine'] for r in rows),scope='Descriptive four-fit restart audit, width8. Permutation/sign/scale handled. Centered feature matching excludes Gaussian means to prevent constant-dominated similarity. Same surrogate and warmstarts; no monosemanticity claim.');(P/'SHARED_BANK_IDENTITY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
