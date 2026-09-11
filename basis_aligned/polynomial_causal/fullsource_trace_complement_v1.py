"""Weights-only isotropic complement test; fixed spectral program, no text fitting."""
from pathlib import Path
import torch,json,time
from fullsource_quartic_program_v1 import run
P=Path(__file__).resolve().parent
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter()
 programs=torch.load(P/'FULLSOURCE_QUARTIC_SPECTRAL_V1_PROGRAMS.pt',weights_only=True)['programs']
 binding=json.loads((P/'FULLSOURCE_QUARTIC_SPECTRAL_V1_BINDING.json').read_text())['files']
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 u=state['lm_head.weight'].double();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double() for layer in (16,17) for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0]);trace_read=scale*d0@((l0*r0).sum(1))
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double()
 den=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
 producer=((x@l0.T)*(x@r0.T))@d0.T*scale
 refs=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted']
 reports=[];identities=[]
 for idx,p in enumerate(programs):
  writer=p['output_writers'];v=metric@writer;alphas=[];exact=[]
  for m in range(2):
   c=d1.T@v[:,m];a=l1.T@(c[:,None]*r1);M=(a+a.T)/2
   vals,axes=torch.linalg.eigh(M);sel=vals.abs().topk(16).indices;outer=axes[:,sel]
   signs=[]
   for j in range(16):
    b=p['input_readers'][m,j];nu=p['inner_weights'][m,j]
    nb=scale*((b.T@l0.T)*(b.T@r0.T))@(d0.T@outer[:,j])
    signs.append(torch.sign((nb*nu).sum()))
   outer=outer*torch.stack(signs)[None,:]
   alphas.append((trace_read@outer-p['inner_weights'][m].sum(-1))/(x.shape[1]-16))
   exact.append((producer*(producer@M)).sum(-1))
  reads=torch.einsum('nd,mjdr->nmjr',x,p['input_readers'])
  q=(reads.square()*p['inner_weights']).sum(-1)
  omitted_norm=x.square().sum(-1)[:,None,None]-reads.square().sum(-1)
  corrected=q+omitted_norm*torch.stack(alphas)
  scalar=(corrected.square()*p['outer_weights']).sum(-1)
  write=scalar@writer.T/den[:,None]
  identities.append(rel(torch.stack(exact,1)@writer.T/den[:,None],refs[idx]))
  reports.append(dict(seed=p['seed'],baseline_error=rel(run(p,x,den),refs[idx]),corrected_error=rel(write,refs[idx]),alpha_norm=float(torch.stack(alphas).norm())))
 result=dict(pred_a=max(identities)<=1e-8,pred_b=all(r['corrected_error']<=.5*r['baseline_error'] for r in reports),pred_c=all(r['corrected_error']<=.1 for r in reports),maximum_identity_error=max(identities),reports=reports,wall_seconds=time.perf_counter()-start)
 out=P/'FULLSOURCE_TRACE_COMPLEMENT_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 with torch.no_grad():main()
