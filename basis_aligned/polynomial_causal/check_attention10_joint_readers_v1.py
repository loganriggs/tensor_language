"""Weights-only optimal common input subspace for three folded consumers."""
from pathlib import Path
import torch,json,time
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter()
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
 o=sd['transformer.h.10.attn.c_proj.weight'].double()
 l,r=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right')]
 maps=[o,l@o,r@o];grams=[m.T@m for m in maps];norms=[g.trace() for g in grams]
 gram=sum(g/n for g,n in zip(grams,norms))/3
 eigen,q=torch.linalg.eigh(gram);rows=[]
 for rank in (256,512,768,1024,1152):
  v=q[:,-rank:];errors=[float(((n-(v*(g@v)).sum()).clamp_min(0)/n).sqrt()) for g,n in zip(grams,norms)]
  rows.append(dict(rank=rank,relative_errors=errors,joint_relative_error=float(eigen[:-rank].clamp_min(0).sum().sqrt()) if rank<1152 else 0.,
   direct_change_macs=1152*1152+2*4608*1152,shared_change_macs=rank*(1152+1152+2*4608),
   additional_factor_scalars=rank*(1152+1152+2*4608),
   scope='Original L/R/O remain needed for pristine context and other paths; factor storage is additional unless those dependencies are separately replaced.'))
 result=dict(prediction='Some rank<=768 gives <=10% relative error for each of O,LO,RO.',pred_a=any(x['rank']<=768 and max(x['relative_errors'])<=.1 for x in rows),rows=rows,seconds=time.perf_counter()-tic,
  block_order=['O','L O','R O'],scope='Exact optimal common input subspace for equally Frobenius-normalized block objective, FP64 actual weights. No text fit, native fidelity, or claim about sparse/block/DAG alternatives.')
 (P/'ATTENTION10_JOINT_READERS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
