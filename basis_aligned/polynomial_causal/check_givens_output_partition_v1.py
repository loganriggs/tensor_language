from pathlib import Path
import json,torch
from head17_output_block_objective_v1 import build
from sparse_interaction_executor_v1 import compile_program
from reconstruct_givens_candidate_v1 import reconstruct
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);t,ids=build();_,baseline=compile_program(t,.1);learned,_=reconstruct(t,4896936)
 def split(x):return (x[0::2]+x[1::2])/2**.5,(x[0::2]-x[1::2])/2**.5
 common,difference=split(t);rows=[]
 for name,f in [('previous_sparse',baseline),('matched_givens',learned)]:
  c,d=split(f);ec=c-common;ed=d-difference
  identity=float(abs((ec.square().sum()+ed.square().sum())-(f-t).square().sum())/t.square().sum())
  rows.append(dict(name=name,total_error=float((f-t).norm()/t.norm()),common_error=float(ec.norm()/common.norm()),difference_error=float(ed.norm()/difference.norm()),per_pair_difference_errors=[float(e.norm()/v.norm()) for e,v in zip(ed,difference)],partition_identity_error=identity))
 out={'pred_a':rows[1]['difference_error']>rows[0]['difference_error'],'pred_b':max(r['partition_identity_error'] for r in rows)<1e-12,'rows':rows,'scope':'Weight-only partition of fixed errors, no activation fit. Contrast coefficient error is not a causal/behavioral metric.'}
 (P/'INTERACTION_GIVENS_OUTPUT_PARTITION_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
