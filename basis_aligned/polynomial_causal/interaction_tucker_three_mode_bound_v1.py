from pathlib import Path
import json,torch,time
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();t,_=build();energy=t.square().sum();x=t.reshape(12,-1);eig=torch.linalg.eigvalsh(x@x.T).flip(0).clamp_min(0);output_tail=torch.cat([eig.sum()[None],eig.sum()-eig.cumsum(0)]).clamp_min(0)/energy
 prior=json.loads((P/'INTERACTION_TUCKER_BUDGET_BOUND_V1_RESULT.json').read_text());rt,ht=[torch.tensor(v,dtype=torch.float64) for v in prior['mode_tails_squared']];budget=prior['budget_bytes']
 residual=torch.arange(1,1153)[:,None];head=torch.arange(1,129)[None,:];best=None;counts=0;cheapest=None
 for output in range(1,13):
  cost=4*(1152*residual+128*head+12*output+output*residual*head);bound=torch.maximum(torch.maximum(rt[1:,None],ht[None,1:]),output_tail[output]).sqrt();valid=cost<=budget;passing=valid&(bound<=.1);counts+=int(passing.sum());masked=bound.masked_fill(~valid,float('inf'));index=int(masked.argmin());ri,hi=divmod(index,128)
  row=dict(output_rank=output,residual_rank=ri+1,head_rank=hi+1,bytes=int(cost[ri,hi]),relative_error_lower_bound=float(bound[ri,hi]))
  if best is None or row['relative_error_lower_bound']<best['relative_error_lower_bound']:best=row
  if passing.any():
   cheapest_cost=cost.masked_fill(~passing,2**60);index=int(cheapest_cost.argmin());ri,hi=divmod(index,128);row=dict(output_rank=output,residual_rank=ri+1,head_rank=hi+1,bytes=int(cost[ri,hi]),relative_error_lower_bound=float(bound[ri,hi]))
   if cheapest is None or row['bytes']<cheapest['bytes']:cheapest=row
 out={'pred_a':counts>0,'best_lower_bound':best,'feasible_pairs_not_ruled_out_at_10pct':counts,'cheapest_not_ruled_out':cheapest,'output_tail_squared':output_tail.tolist(),'seconds':time.perf_counter()-tic,'scope':'Allthree denseTuckermode ranks, alladapterscharged. Necessarybounds only; a passingranktriple still requires jointfit andbehavioral validation. Sparsecore/LL1/shared external adapters outside this representation.'}
 (P/'INTERACTION_TUCKER_THREE_MODE_BOUND_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
