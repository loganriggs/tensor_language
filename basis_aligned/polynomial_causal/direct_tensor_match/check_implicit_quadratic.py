import json,torch
from pathlib import Path
from implicit_quadratic import inner
from core import packed_quadratic,metric

def main():
 torch.manual_seed(311);torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 c,a,b=torch.randn(3,5),torch.randn(5,4),torch.randn(5,4)
 d,l,r=torch.randn(3,2,requires_grad=True),torch.randn(2,4,requires_grad=True),torch.randn(2,4,requires_grad=True)
 records=[]
 for name,gaussian in [('gaussian',True),('frobenius',False)]:
  exact=inner(c,a,b,d,l,r,gaussian);coeffa=c@packed_quadratic(a,b);coeffb=d@packed_quadratic(l,r);direct=((coeffa@metric(4,2,name))*coeffb).sum()
  g1=torch.autograd.grad(exact,(d,l,r),retain_graph=True);g2=torch.autograd.grad(direct,(d,l,r),retain_graph=True)
  error=float(abs(exact-direct).detach());grad=max(float((x-y).abs().max()) for x,y in zip(g1,g2));assert error<1e-10 and grad<1e-10
  records.append(dict(metric=name,inner_error=error,gradient_error=grad))
 return dict(records=records)
if __name__=='__main__':
 x=main();Path(__file__).with_name('IMPLICIT_QUADRATIC_CHECK_V1.json').write_text(json.dumps(x,indent=2)+'\n');print(x)
