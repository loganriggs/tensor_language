from pathlib import Path
import json,torch
from head17_output_block_objective_v1 import build
from sparse_interaction_executor_v1 import compile_program
from reconstruct_givens_candidate_v1 import reconstruct
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);t,ids=build();_,baseline=compile_program(t,.1);learned,_=reconstruct(t,4896936)
 data=torch.load(P/'COMPOSED_LAST_BLOCK_STATES_V1_ARTIFACT.pt',weights_only=True);z=data['linear_parts'][:,3].double();v=data['linear_parts'][:,2].double()-z
 w=torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt',weights_only=True)['output_matrix'].double();a=torch.linalg.lstsq(w,v.T).solution.T
 rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'][:24]+json.loads((P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ROWS.json').read_text())['rows'];lookup={v:i for i,v in enumerate(ids)};uk=torch.tensor([lookup[r['uk_id']] for r in rows]);us=torch.tensor([lookup[r['us_id']] for r in rows]);index=torch.arange(120)
 def evaluate(x):
  raw=torch.einsum('oia,ni,na->no',x,z,a);return raw[index,uk]-raw[index,us]
 ref=evaluate(t);eps=torch.finfo(torch.float32).eps;den=(data['linear_parts'][:,2].double().square().mean(-1)+eps)*(data['states'][:,2].double().square().mean(-1)+eps).sqrt();result=[]
 for name,f in [('previous_sparse',baseline),('matched_givens',learned)]:
  err=evaluate(f)-ref;cells=[]
  for group in range(5):
   sl=slice(24*group,24*(group+1));cells.append(dict(group=group,numerator_error=float(err[sl].norm()/ref[sl].norm()),normalized_precap_error=float((err[sl]/den[sl]).norm()/(ref[sl]/den[sl]).norm())))
  result.append(dict(name=name,cells=cells))
 out={'results':result,'scope':'Frozen mixedoperator on prior120retainedports, designatedoutputpairs. Raw numerator and normalized pre-softcap ownterm errors; no newnativeforward or datafit.'}
 (P/'INTERACTION_GIVENS_PORT_ERROR_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
