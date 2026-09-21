"""Empirical necessary output-rank floors on cached true native quartic targets."""
import torch,json,time,math
from pathlib import Path

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent;data=torch.load(p/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);rows=[]
 for index,raw in enumerate(data['targets']):
  for centering in [False,True]:
   y=raw.double();y=y-y.mean(0) if centering else y
   ev=torch.linalg.eigvalsh(y.T@y).flip(0);negative=float(ev.min()/ev.max());assert negative>-1e-10;ev=ev.clamp_min(0);tail=ev.flip(0).cumsum(0).flip(0);energy=ev.sum();errors={str(k):float((tail[k]/energy).sqrt()) for k in [1,3,6,8,10,16,32,64,128]};requirements={str(tol):int((tail/energy>tol**2).sum()) for tol in [.01,.05,.1]}
   rows.append(dict(panel=index,samples=len(y),centered=centering,rank_errors=errors,necessary_ranks=requirements,necessary_quadratic_dictionary_widths={tol:math.ceil((math.sqrt(1+8*r)-1)/2) for tol,r in requirements.items()},minimum_relative_eigenvalue=negative))
 result=dict(records=rows,seconds=time.monotonic()-start,scope='Necessary output-rank bounds on cached true native quartic path targets, not archivedstudent or fullnormalizedmodel. Pure m-quadratic-feature homogeneous quartic readout has <=m(m+1)/2outputdirections. Centered result permits a freely fitted mean; uncentered pure polynomial bound is primary. Empirical matrix lower bounds on these rows, not coefficient bounds or guarantees about fresh text. Independent rank or dictionary-width bounds are not sufficient for attainable reconstruction.')
 (p/'NATIVE_QUARTIC_OUTPUT_CAPACITY_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
