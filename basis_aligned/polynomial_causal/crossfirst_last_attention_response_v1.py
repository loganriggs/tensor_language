"""Actual-weight conditional attention-write response control; no fitting."""
from pathlib import Path
import torch,json,time
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(913218);tic=time.perf_counter();binding=json.loads((P/'CROSSFIRST_LAST_BLOCK_SPLIT_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True)
 L=sd['transformer.h.17.mlp.Left.weight'].double();R=sd['transformer.h.17.mlp.Right.weight'].double();D=sd['transformer.h.17.mlp.Down.weight'].double();z=torch.randn(24,1152,dtype=torch.float64);v=torch.randn_like(z)*torch.linspace(-.15,.15,24,dtype=torch.float64)[:,None];eps=torch.finfo(torch.float32).eps
 rho=z.square().mean(-1,keepdim=True)+eps;newrho=(z+v).square().mean(-1,keepdim=True)+eps;lz=z@L.T;rz=z@R.T;lv=v@L.T;rv=v@R.T;base=(lz*rz)@D.T/rho
 cross=(lz*rv+lv*rz)@D.T/newrho;square=(lv*rv)@D.T/newrho;norm=(rho/newrho-1)*base
 pred=v+cross+square+norm;direct=v+(((z+v)@L.T)*((z+v)@R.T))@D.T/newrho-base
 error=float((pred-direct).norm()/direct.norm());assert error<1e-12
 a=torch.load(P/'CROSSFIRST_LAST_BLOCK_SPLIT_V1_ARTIFACT.pt',weights_only=True)['measures'];cells=[]
 for k in range(4):
  z=a[24*k:24*(k+1),:,0];full=z[:,4]-z[:,5];m=z[:,6]-z[:,5];att=z[:,7]-z[:,5]
  cells.append(dict(group=k,MLP_attention_cosine=float((m*att).sum()/(m.norm()*att.norm())),MLP_same_sign_full=int(((m*full)>0).sum()),attention_same_sign_full=int(((att*full)>0).sum()),MLP_norm_over_full=float(m.norm()/full.norm()),attention_norm_over_full=float(att.norm()/full.norm())))
 result=dict(actual_weight_FP64_response_error=error,control_contexts=24,seconds=time.perf_counter()-tic,regional_component_audit=cells,scope='Random-state actual-MLP17weight expansion for context-dependent attention perturbation: direct write + bilinear cross + square + normalization rescale. Native attention-branch term mediation not yet tested. Full L/R/D and baseline states retained; no smaller model or new theorem.')
 (P/'CROSSFIRST_LAST_ATTENTION_RESPONSE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
