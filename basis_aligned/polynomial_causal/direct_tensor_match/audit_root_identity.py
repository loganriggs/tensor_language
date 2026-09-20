"""Gauge-aware root feature identity under the exact centered root covariance."""
import itertools,json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from root_product_fit import coefficients
P=Path(__file__).resolve().parent

def compare(f,g,G):
 H=coefficients(f['a'],f['b']);J=coefficients(g['a'],g['b']);C=f['c'];D=g['c'];hn=((H@G)*H).sum(1).sqrt();jn=((J@G)*J).sum(1).sqrt();cos=(H@G@J.T)/(hn[:,None]*jn[None,:]);i,j=linear_sum_assignment(-cos.abs().numpy());outcos=(C.T@D)/(C.norm(dim=0)[:,None]*D.norm(dim=0)[None,:]);component=cos*outcos;F=C@H;K=D@J
 return dict(mean_feature_cosine=float(cos[i,j].abs().mean()),minimum_feature_cosine=float(cos[i,j].abs().min()),matched_output_direction_minimum_cosine=float(outcos[i,j].abs().min()),matched_component_minimum_cosine=float(component[i,j].min()),centered_function_cosine=float(((F@G)*K).sum()/((((F@G)*F).sum()*((K@G)*K).sum()).sqrt())))

def main():
 torch.set_num_threads(1);allfits=torch.load(P/'ROOT_PRODUCT_REFACTOR_V1.pt',weights_only=True)['allfits'];G=torch.load(P/'ROOT_ARCHIVE_METRIC_V1.pt',weights_only=True)['covariance'];fits={k:v for k,v in allfits.items() if k[0]==4};pairs=[]
 for (ka,a),(kb,b) in itertools.combinations(fits.items(),2):pairs.append(dict(first=list(ka),second=list(kb),**compare(a,b,G)))
 f=next(iter(fits.values()));perm=torch.tensor([2,0,3,1]);scale=torch.tensor([-2.,3.,-.5,4.],dtype=torch.float64);g=dict(a=f['a'][perm]*scale[:,None],b=f['b'][perm],c=f['c'][:,perm]/scale);control=compare(f,g,G);assert min(control.values())>1-1e-10;high=next(r for r in pairs if r['first']==[4,'muon',.03,0] and r['second']==[4,'muon',.03,1]);out=dict(pairs=pairs,gauge_control=control,paired_high_rate_muon=high,scope='Root4 centered Gaussian feature identity on fixed bank, with mean freedom removed. All-rate pairs include underfit fits; paired high-rate comparison reported separately. Output directions measured in exact output-frame metric. No semantics inferred.');(P/'ROOT_IDENTITY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(high)
if __name__=='__main__':main()
