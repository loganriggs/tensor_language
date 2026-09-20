"""Registered two-component input mixture versus single Gaussian and random split."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def moments(groups):
 third=0;fourth=0;qm=0;qs=0
 for weight,mu,cov in groups:
  var=cov.diag();third=third+weight*(mu.pow(3)+3*mu*var);fourth=fourth+weight*(mu.pow(4)+6*mu.square()*var+3*var.square());a,b=mu[8:12],mu[12:16];AA=cov[8:12,8:12];BB=cov[12:16,12:16];AB=cov[8:12,12:16];mean=a*b+AB.diag();Q=AA*BB+AB*AB.T+(a[:,None]*a[None,:])*BB+(b[:,None]*b[None,:])*AA+(a[:,None]*b[None,:])*AB.T+(b[:,None]*a[None,:])*AB;qm=qm+weight*mean;qs=qs+weight*(Q+mean[:,None]*mean[None,:])
 return third,fourth,qs-qm[:,None]*qm[None,:]

def main():
 torch.set_num_threads(2);s={k:v.double() for k,v in torch.load(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt',weights_only=True)['programs']['centered'].items()};panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];A=torch.cat([s['linear_reader'],s['quadratic_left'],s['quadratic_right']]);M=panels[0]['covariance'].double();A=A/((A@M)*A).sum(1).sqrt()[:,None];zs=[(p['rows'].double()-s['mu'])@A.T for p in panels];z=zs[0];direction=int(z.pow(3).mean(0).abs().argmax());split=z[:,direction]>z[:,direction].median();gen=torch.Generator().manual_seed(2035);random=split[torch.randperm(len(z),generator=gen)]
 def params(mask):
  x=z[mask];m=x.mean(0);c=x-m;return float(len(x)/len(z)),m,c.T@c/len(x)
 groups={'single':[params(torch.ones(len(z),dtype=torch.bool))],'mixture':[params(split),params(~split)],'random_split':[params(random),params(~random)]};records=[]
 for law,g in groups.items():
  third,fourth,Q=moments(g)
  for i,t in enumerate(zs):
   q=t[:,8:12]*t[:,12:16];q=q-q.mean(0);emp=q.T@q/len(q);records.append(dict(law=law,panel=i,third_moment_rmse=float((third-t.pow(3).mean(0)).square().mean().sqrt()),fourth_moment_rmse=float((fourth-t.pow(4).mean(0)).square().mean().sqrt()),quadratic_covariance_relative_error=float((Q-emp).norm()/emp.norm())))
 base=next(r for r in records if r['law']=='single' and r['panel']==1);test=next(r for r in records if r['law']=='mixture' and r['panel']==1);pred={name:test[key]<=.9*base[key] for name,key in [('pred_fourth','fourth_moment_rmse'),('pred_quadratic_covariance','quadratic_covariance_relative_error')]};out=dict(split_direction=direction,split_threshold=float(z[:,direction].median()),component_counts=[int(split.sum()),int((~split).sum())],records=records,predictions=pred,scope='Input-only low-dimensional mixture diagnostic; calibration-only split/moments, reused second panel. No student fitting.');(P/'MIXTURE_MOMENT_DIAGNOSTIC_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
