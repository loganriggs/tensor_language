"""Exact adjacent-boundary local mixed generation and native contrast audit."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(913217);dtype=torch.float64
 # Signed bilinear/RMS residual blocks with a shared fixed re-entry input.
 n,d,m=12,7,11;x0=torch.randn(n,d,dtype=dtype);base=torch.randn(n,d,dtype=dtype);c=.1*torch.randn(n,d,dtype=dtype);r=.2*torch.randn(n,d,dtype=dtype)
 weights=[(torch.randn(m,d,dtype=dtype),torch.randn(m,d,dtype=dtype),.05*torch.randn(d,m,dtype=dtype)) for _ in range(3)]
 def block(x,k):
  a,b,D=weights[k];z=.8*x+.2*x0;u=z/(z.square().mean(-1,keepdim=True)+1e-7).sqrt();return z+((u@a.T)*(u@b.T))@D.T
 states=[base,base+c,base+c+r,base+r]
 states=[block(x,0) for x in states];N,C,Pp,R=states;add=C+R-N
 out=[block(x,1) for x in states];oN,oC,oP,oR=out;local=block(add,1)-oC-oR+oN;prop=oP-block(add,1);mixed=oP-oC-oR+oN
 def suffix(x):return torch.tanh(block(x,2)).sum(-1)
 earlier=suffix(block(add,1));later=suffix(oC+oR-oN);local_effect=suffix(oC+oR-oN+local)-suffix(oC+oR-oN)
 rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
 control=dict(mixed_partition_error=rel(local+prop,mixed),adjacent_effect_error=rel(earlier-later,local_effect))
 assert max(control.values())<1e-12
 artifact=torch.load(P/'CROSSFIRST_BOUNDARY_CURVE_V1_ARTIFACT.pt',weights_only=True);a=artifact['measures'];cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=a[lo:hi,:,0];total=z[:,2]-z[:,1]-z[:,3]+z[:,0];step=z[:,11]-z[:,12]
  cells.append(dict(cell=label,last_block_local_effect_norm_over_total=float(step.norm()/total.norm()),last_block_local_aligned_over_total=float((step*total).sum()/total.square().sum()),last_block_local_cosine_total=float((step*total).sum()/(step.norm()*total.norm())),maxabs_last_block_local_effect=float(step.abs().max())))
 result=dict(control=control,cells=cells,scope='Adjacent reset contrast equals causal effect of newly generated block mixed state around additive output background, conditional on original N/C/R states and fixed shared x0/v1. It is not local attribution on native parent state and need not sum across layers. No new fitting or native forwards.')
 (P/'CROSSFIRST_ADJACENT_BOUNDARY_IDENTITY_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
