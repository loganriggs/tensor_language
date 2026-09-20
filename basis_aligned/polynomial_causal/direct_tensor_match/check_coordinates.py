import json,torch
from pathlib import Path
from core import Model,metric,covariance_metric,evaluate
from coordinates import transform_matrix,transform_model_state

def main():
 torch.set_num_threads(1);torch.manual_seed(841);torch.set_default_dtype(torch.float64);L=torch.randn(3,3)+2*torch.eye(3);rows=[]
 for kind,n in [('monomial',2),('cp',2),('tucker',2),('tree',4),('dag',4)]:
  m=Model(3,2,kind,2,n);b=Model(3,2,kind,2,n);b.load_state_dict(transform_model_state(m,L));T=transform_matrix(L,n)
  coeff=float((b()-m()@T).detach().abs().max());z=torch.randn(8,3);function=float((evaluate(b(),z,n)-evaluate(m(),z@L.T,n)).detach().abs().max());M=covariance_metric(3,n,L@L.T);gram=float((T@metric(3,n)@T.T-M).norm()/M.norm())
  assert coeff<1e-9 and function<1e-8 and gram<1e-12;rows.append(dict(kind=kind,coefficient_error=coeff,function_error=function,metric_relative_error=gram))
 return dict(records=rows)
if __name__=='__main__':
 r=main();Path(__file__).with_name('COORDINATE_CHECK_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
