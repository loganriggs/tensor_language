"""Compare coefficient versus exact noncentral Gaussian bank errors."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def embedding(T,m):
 # E[(x^T Q x)^2] = 2||Q||²+4||Qm||²+(trQ+m^T Qm)².
 return torch.cat([2**.5*T.flatten(1),2*torch.einsum('vij,j->vi',T,m),(T.diagonal(dim1=-2,dim2=-1).sum(-1)+torch.einsum('i,vij,j->v',m,T,m))[:,None]],1)

def main():
 torch.set_num_threads(1);core=torch.load(P/'QUARTIC_BANK_CORE_V1.pt',weights_only=True);fits=torch.load(P/'QUARTIC_BANK_REFACTOR_V1.pt',weights_only=True);result=json.loads((P/'QUARTIC_BANK_REFACTOR_V1.json').read_text());mu=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['mean'].double();m=core['input_mapback']@mu;T=core['weighted_core'];base=embedding(T,m);rows=[]
 for w in result['winners']:
  f=fits['allfits'][(w['width'],w['optimizer'],w['lr'],w['seed'])];a,b,c=f['a'],f['b'],f['c'];features=.5*(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:]);pred=torch.einsum('vk,kij->vij',c,features);delta=T-pred;e=embedding(delta,m);parts=[float(e[:,:1024].square().sum()),float(e[:,1024:1056].square().sum()),float(e[:,-1].square().sum())];rows.append(dict(width=w['width'],coefficient_relative_error=float(delta.norm()/T.norm()),gaussian_function_relative_error=float(e.norm()/base.norm()),gaussian_error_energy_parts=dict(centered_quadratic=parts[0],linear_from_mean=parts[1],constant=parts[2]),mean_coordinate_norm=float(m.norm())))
 # Independent four-dimensional Gaussian quadrature oracle for the embedding.
 from dag_square_optimizer import rule
 torch.manual_seed(2044);Q=torch.randn(2,4,4,dtype=torch.float64);Q=(Q+Q.transpose(-1,-2))/2;mean=torch.randn(4,dtype=torch.float64);xx,weights=rule(5);x=xx[:,:4]+mean;y=torch.einsum('ni,vij,nj->nv',x,Q,x);actual=(weights[:,None]*y.square()).sum();expected=embedding(Q,mean).square().sum();error=float(abs(actual-expected)/actual);assert error<1e-12
 out=dict(records=rows,quadrature_relative_error=error,scope='Exact Gaussian functional bank error with fixed root sensitivity, not composed quartic error. Separates coefficient error from linear/constant contributions induced by nonzero mean.');(P/'BANK_FUNCTION_METRIC_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
