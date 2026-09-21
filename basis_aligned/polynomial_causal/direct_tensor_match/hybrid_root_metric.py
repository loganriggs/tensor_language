"""Relative-error mixtures of empirical sensitivity and synthetic native targets."""
import torch

def combine(text_x,text_y,text_w,synthetic_x,synthetic_y,coefficient):
 if coefficient<0:raise ValueError('negative mixture weight')
 if coefficient==0:return text_x,text_y,text_w
 a=text_w/(text_w*text_y.square()).sum(0,keepdim=True)
 b=torch.ones_like(synthetic_y)/synthetic_y.square().sum(0,keepdim=True)
 return torch.cat([text_x,synthetic_x]),torch.cat([text_y,synthetic_y]),torch.cat([a,coefficient*b])

def controls():
 torch.manual_seed(881);dtype=torch.float64;x=torch.randn(13,5,dtype=dtype);g=torch.randn(17,5,dtype=dtype);y=torch.randn(13,4,dtype=dtype);z=torch.randn(17,4,dtype=dtype);w=torch.rand_like(y)+.01;h=torch.randn_like(y);k=torch.randn_like(z);rows=[]
 for coefficient in [.1,1.,10.]:
  _,target,weight=combine(x,y,w,g,z,coefficient);p=torch.cat([h,k]);actual=((weight*(p-target).square()).sum(0)/(weight*target.square()).sum(0)).mean();expected=(((w*(h-y).square()).sum(0)/(w*y.square()).sum(0)+coefficient*(k-z).square().sum(0)/z.square().sum(0))/(1+coefficient)).mean();error=float(abs(actual-expected));assert error<1e-12;rows.append(dict(coefficient=coefficient,mixture_replay=error))
 xx,yy,ww=combine(x,y,w,g,z,0);assert torch.equal(xx,x) and torch.equal(yy,y) and torch.equal(ww,w)
 return rows
if __name__=='__main__':
 import json
 from pathlib import Path
 rows=controls();Path(__file__).with_name('HYBRID_ROOT_METRIC_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
