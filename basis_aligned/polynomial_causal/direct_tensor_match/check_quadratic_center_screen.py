from pathlib import Path
import json,torch
from quadratic_center_screen import screen
P=Path(__file__).parent
torch.set_num_threads(2)
torch.set_grad_enabled(False)
def main():
 rows=[]
 for seed,sizes in enumerate([[1]*12,[3]*4,[4,8],[12],[2,3,7]]):
  g=torch.Generator().manual_seed(23000+seed);n=sum(sizes);Q=torch.zeros(6,n,n,dtype=torch.float64);Q[0]=torch.eye(n,dtype=Q.dtype)
  start=0
  for k in sizes:
   raw=torch.randn(5,k,k,dtype=Q.dtype,generator=g);Q[1:,start:start+k,start:start+k]=(raw+raw.transpose(-1,-2))/2;start+=k
  U=torch.linalg.qr(torch.randn(n,n,dtype=Q.dtype,generator=g)).Q
  V=torch.linalg.qr(torch.randn(n,n,dtype=Q.dtype,generator=g)).Q
  change=(U*torch.logspace(0,1,n,dtype=Q.dtype))@V.T
  # Nonorthogonal coordinate change hides the disjoint blocks.
  native=change.T@Q@change
  a=torch.eye(6,dtype=Q.dtype)[0];b=torch.eye(6,dtype=Q.dtype)[1]
  result=screen(native,a,b)
  assert result['instrument']=='PASS' and result['primary_sizes']==sorted(sizes),result
  # Cross-check center dimension by independent dense linear equations in X.
  cols=[]
  for index in range(n*n):
   X=torch.zeros(n,n,dtype=Q.dtype);X.flatten()[index]=1
   Z=native@X;cols.append((Z-Z.transpose(-1,-2)).flatten())
  constraint=torch.stack(cols,1);sv=torch.linalg.svdvals(constraint);nullity=int((sv<1e-9*sv[0]).sum())
  assert nullity==len(sizes),(nullity,sizes)
  rows.append(dict(seed=seed,planted_block_sizes=sizes,dense_center_nullity=nullity,screen=result))
 # Degenerate pencil must not fabricate a verdict.
 Q=torch.stack([torch.eye(4,dtype=torch.float64)]*3);a=torch.tensor([1.,0.,0.],dtype=Q.dtype);b=torch.tensor([0.,1.,0.],dtype=Q.dtype)
 invalid=screen(Q,a,b);assert invalid['instrument']=='INCONCLUSIVE'
 out=dict(five_nonorthogonal_structural_controls=rows,degenerate_pencil_control=invalid)
 (P/'QUADRATIC_CENTER_PREFLIGHT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print('PASS five nonorthogonal planted structures, independent center nullities, and degenerate-pencil rejection')
if __name__=='__main__':main()
