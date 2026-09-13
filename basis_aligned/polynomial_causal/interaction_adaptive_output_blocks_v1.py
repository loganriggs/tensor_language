"""Exact fixed-frame rate allocation with dense-or-factored block choices."""
from pathlib import Path
import json,time,sys,numpy as np,torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();t,ids=build();flat=t.flatten(1);energy=float(t.square().sum());budget=4896936;unit=256;capacity=(budget//4-144)//unit
 spectral=torch.linalg.eigh(flat@flat.T).eigenvectors
 old=torch.load(P/'HEAD17_OUTPUT_BLOCK_FIT_V1_PROGRAM.pt',weights_only=True);assert ids==old['token_ids'].tolist()
 pair=torch.zeros(12,12,dtype=torch.float64)
 for j in range(6):pair[2*j:2*j+2,2*j:2*j+2]=torch.tensor([[1.,1.],[1.,-1.]],dtype=torch.float64)/2**.5
 balanced='--balanced' in sys.argv
 frames={'pair_sum_difference':pair} if balanced else {'native':torch.eye(12,dtype=torch.float64),'spectral':spectral,'prior_learned':old['output_basis'].double(),'pair_sum_difference':pair};records=[]
 ranks=list(range(116))+[128];costs=[5*r for r in range(116)]+[576]
 for name,q in frames.items():
  mixed=(q.T@flat).reshape_as(t);u,s,vh=torch.linalg.svd(mixed,full_matrices=False);captures=torch.cat([torch.zeros(12,1,dtype=torch.float64),s.square().cumsum(1)],1).numpy()/energy
  if balanced:
   weights=torch.empty(12,dtype=torch.float64)
   weights[0::2]=energy/(2*mixed[0::2].square().sum());weights[1::2]=energy/(2*mixed[1::2].square().sum())
   captures=captures*weights.numpy()[:,None]
  dp=np.full(capacity+1,-np.inf);dp[0]=0;backs=[]
  for block in range(12):
   nxt=np.full_like(dp,-np.inf);chosen=np.full(capacity+1,-1,dtype=np.int16)
   for option,(rank,cost) in enumerate(zip(ranks,costs)):
    candidate=dp[:capacity+1-cost]+captures[block,rank];better=candidate>nxt[cost:];nxt[cost:][better]=candidate[better];chosen[cost:][better]=option
   dp=nxt;backs.append(chosen)
  end=int(np.argmax(dp));spent=end;allocated=[]
  for block in range(11,-1,-1):
   option=int(backs[block][end]);allocated.append(ranks[option]);end-=costs[option]
  allocated.reverse();assert end==0
  fitted=torch.stack([(u[j,:,:r]*s[j,:r])@vh[j,:r] for j,r in enumerate(allocated)]);reconstructed=(q@fitted.flatten(1)).reshape_as(t);error=float((reconstructed-t).norm()/t.norm());expected=float(max(0,1-dp[spent])**.5)
  objective_error=float((((fitted-mixed).square().sum((1,2))*weights).sum()/energy).sqrt()) if balanced else error
  assert abs(objective_error-expected)<1e-10
  contrast=t[0::2]-t[1::2];predcontrast=reconstructed[0::2]-reconstructed[1::2]
  records.append(dict(frame=name,ranks=allocated,dense_blocks=sum(r==128 for r in allocated),bytes=4*(144+unit*spent),relative_error=error,contrast_error=float((predcontrast-contrast).norm()/contrast.norm()),objective_error=objective_error,energy_replay_error=abs(objective_error-expected),pred_a=error<=.1))
 out={'records':records,'seconds':time.perf_counter()-tic,'scope':'Exact rank/dense choice for listed frozen outputframes and explicit objective. Outputframe not optimized atthisbudget. Metadata/runtime/nativecontext costs excluded; no behavior or generalLL1 impossibility claim.'}
 out['objective']='equal relative squared common/difference error' if balanced else 'total squared Frobenius error'
 destination='INTERACTION_ADAPTIVE_OUTPUT_BALANCED_V1_RESULT.json' if balanced else 'INTERACTION_ADAPTIVE_OUTPUT_BLOCKS_V1_RESULT.json'
 (P/destination).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
