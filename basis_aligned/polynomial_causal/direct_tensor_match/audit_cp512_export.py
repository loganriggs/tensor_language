"""Audit the physical saved CP coefficients and literal storage on CPU."""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);rows=[]
 for seed in [1001,1002]:
  p=torch.load(P/f'QUARTIC_CP512_SEED{seed}_V1.pt',weights_only=True,map_location='cpu')
  x=torch.randn(64,1152,generator=torch.Generator().manual_seed(991),dtype=torch.float64)
  def evaluate(dtype):
   z=torch.ones(64,512,dtype=dtype)
   for f in p['factors']:z=z*(x.to(dtype)@f.to(dtype).T)
   return z@p['coefficients'].to(dtype).T@p['writer'].to(dtype).T
  double=evaluate(torch.float64);single=evaluate(torch.float32).double()
  coefficients=sum(t.numel() for t in p['factors'])+p['coefficients'].numel()+p['writer'].numel()
  row=dict(seed=seed,physical_export_precision_error=float((single-double).norm()/double.norm()),stored_coefficients=coefficients,variable_products=3*p['factors'][0].shape[0])
  assert row['physical_export_precision_error']<1e-4 and coefficients==2385920
  rows.append(row)
 (P/'QUARTIC_CP512_EXPORT_AUDIT_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
if __name__=='__main__':main()
