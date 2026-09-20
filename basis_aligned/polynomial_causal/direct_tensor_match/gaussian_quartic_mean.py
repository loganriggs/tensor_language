"""Exact noncentral Gaussian means for products of quadratic features."""
import torch

def quadratic_moments(A,B,mu,covariance):
 a,b=A@mu,B@mu;AS=A@covariance;AA=AS@A.T;AB=AS@B.T;BB=(B@covariance)@B.T
 mean=a*b+AB.diag()
 cov=AA*BB+AB*AB.T+(a[:,None]*a[None,:])*BB+(a[:,None]*b[None,:])*AB.T+(b[:,None]*a[None,:])*AB+(b[:,None]*b[None,:])*AA
 return mean,cov

def native_mean(teacher,mu,covariance):
 C,L,R,D,A,B=teacher;m,G=quadratic_moments(A,B,mu,covariance);h=D@m;H=D@G@D.T
 return C@((L@h)*(R@h)+((L@H)*R).sum(1))

def bank_mean(U,V,C,mu,covariance):
 bank,width,d=U.shape;m,G=quadratic_moments(U.flatten(0,1),V.flatten(0,1),mu,covariance);m=m.reshape(bank,width).sum(1);G=G.reshape(bank,width,bank,width).sum((1,3));i,j=torch.triu_indices(bank,bank,device=U.device)
 return C@(m[i]*m[j]+G[i,j])

def check():
 import json
 from pathlib import Path
 from dag_square_optimizer import rule
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1937);mu=torch.randn(4);raw=torch.randn(4,4);cov=raw@raw.T;L=torch.linalg.cholesky(cov);xx,w=rule(5);x=xx[:,:4]@L.T+mu;teacher=[torch.randn(*s) for s in [(2,3),(3,5),(3,5),(5,6),(6,4),(6,4)]];C,l,r,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;y=((h@l.T)*(h@r.T))@C.T;actual=(w[:,None]*y).sum(0);got=native_mean(teacher,mu,cov);native_error=float((actual-got).norm()/actual.norm());U,V=torch.randn(3,2,4),torch.randn(3,2,4);c=torch.randn(2,6);q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),3,2).sum(2);i,j=torch.triu_indices(3,3);expected=(w[:,None]*((q[:,i]*q[:,j])@c.T)).sum(0);pred=bank_mean(U,V,c,mu,cov);bank_error=float((pred-expected).norm()/expected.norm());assert max(native_error,bank_error)<1e-12
 out=dict(native_relative_error=native_error,bank_relative_error=bank_error,scope='Exact noncentral Gaussian mean, checked against independent tensor-product Gaussian quadrature. Does not equate Gaussian mean to empirical text mean.');(Path(__file__).resolve().parent/'GAUSSIAN_QUARTIC_MEAN_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out);return out
if __name__=='__main__':check()
