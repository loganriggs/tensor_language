"""Exact mode-rank lower bound at matched dense-Tucker storage budget."""
from pathlib import Path
import json,time,torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();t,ids=build();energy=t.square().sum();tails=[]
 for axis in [1,2]:
  x=t.movedim(axis,0).reshape(t.shape[axis],-1);eig=torch.linalg.eigvalsh(x@x.T).flip(0).clamp_min(0)
  tails.append(torch.cat([eig.sum()[None],eig.sum()-eig.cumsum(0)]).clamp_min(0)/energy)
 budget=4896936;best=None;passing=[]
 for head in range(1,129):
  for residual in range(1,1153):
   scalars=1152*residual+128*head+12*residual*head
   if 4*scalars>budget:break
   bound=float(torch.maximum(tails[0][residual],tails[1][head]).sqrt())
   row=dict(residual_rank=residual,head_rank=head,bytes=4*scalars,relative_error_lower_bound=bound,input_product_nodes=residual*head)
   if best is None or bound<best['relative_error_lower_bound']:best=row
   if bound<=.1:passing.append(row)
 out={'pred_a':best['relative_error_lower_bound']<=.1,'best_lower_bound':best,'feasible_pairs_not_ruled_out_at_10pct':len(passing),'cheapest_not_ruled_out':min(passing,key=lambda x:x['bytes']) if passing else None,'budget_bytes':budget,'mode_tails_squared':[v.tolist() for v in tails],'seconds':time.perf_counter()-tic,'scope':'Necessary rank bounds for dense-core Tucker with residual/head adapters and all12outputs. No fit, sparsity in core, output-mode reduction, shared external adapters or behavioral claim. Bound applies regardless orthogonal/nonorthogonal factors at these multilinear ranks.'}
 (P/'INTERACTION_TUCKER_BUDGET_BOUND_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='mode_tails_squared'},indent=2))
if __name__=='__main__':main()
