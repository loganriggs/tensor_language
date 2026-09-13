"""Sparse core after charged spectral truncation, plus cheap fixed DCT frame."""
from pathlib import Path
import json,time,torch
from scipy.fft import dct
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();t,_=build();energy=float(t.square().sum());budget=4896936;bases=[]
 for axis in range(3):
  x=t.movedim(axis,0).reshape(t.shape[axis],-1);bases.append(torch.linalg.eigh(x@x.T).eigenvectors.flip(1))
 output,residual,head=bases
 partial=torch.einsum('op,oia,ab->pib',output,t,head)
 full=torch.einsum('ir,pib->prb',residual,partial)
 rows=[]
 def evaluate(core,adapter_bytes,label,r,h):
  bitmap=(core.numel()+7)//8;count=max(0,min(core.numel(),(budget-adapter_bytes-bitmap)//4))
  values=core.flatten().square().sort(descending=True).values;captured=float(values[:count].sum());projection=float(core.square().sum());err=max(0,1-captured/energy)**.5
  return dict(feasible=adapter_bytes+bitmap<=budget,frame=label,residual_rank=r,head_rank=h,adapter_bytes=adapter_bytes,bitmap_bytes=bitmap,retained_coefficients=count,total_bytes=adapter_bytes+bitmap+4*count,rank_only_error=max(0,1-projection/energy)**.5,relative_error=err,pred_a=err<=.1)
 for r in [869,900,960,1000,1024]:
  for h in [125,128]:rows.append(evaluate(full[:,:r,:h],4*(144+1152*r+128*h),'truncated_spectral',r,h))
 fixed=torch.from_numpy(dct(partial.numpy(),type=2,norm='ortho',axis=1))
 dct_identity=abs(float(fixed.square().sum())/energy-1);assert dct_identity<1e-12
 rows.append(evaluate(fixed,4*(144+128*128),'fixed_dct_residual',1152,128))
 rows.append(evaluate(partial,4*(144+128*128),'identity_residual_reference',1152,128))
 result={'rows':rows,'dct_energy_error':dct_identity,'seconds':time.perf_counter()-tic,'scope':'Fixed spectral/DCT frames, exact fixed-count core support after adapter/bitmap charge. DCT has specified algorithm, no dense parameter matrix; runtime unpriced. No learned sparseframe or behavior claim.'}
 (P/'INTERACTION_CHARGED_SPARSE_TUCKER_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
