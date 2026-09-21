"""Native midpoint channel dictionary, empirical joint-moment output refit.
Baseline: retain native readers and sharing, fit all output coordinates together.
"""
from pathlib import Path
import torch,json,time
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True)
L=state['transformer.h.17.mlp.Left.weight'].double();R=state['transformer.h.17.mlp.Right.weight'].double()
r=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=r['n'].flatten(0,1).double();m=r['m'].flatten(0,1).double();y=r['y'].flatten(0,1).double();mean=y.mean(0);y=y-mean
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();ys=y@S
phi=(n@L.T)*(m@R.T)+(n@R.T)*(m@L.T);phimean=phi.mean(0);phi-=phimean;energy=phi.square().sum(0);cross=phi.T@ys;score=cross.square().sum(1)/energy.clamp_min(1e-30)
orders={'correlation':score.argsort(descending=True),'random0':torch.randperm(4608,generator=torch.Generator().manual_seed(261204))};records=[];programs={}
for policy,order in orders.items():
 for width in [128,512,1024]:
  ids=order[:width];x=phi[:,ids];gram=x.T@x;cross=x.T@y;val,V=torch.linalg.eigh(gram);keep=val>val[-1]*1e-10;writer=(V[:,keep]/val[keep])@(V[:,keep].T@cross);prediction=x@writer;err=float(((prediction-y)@S).norm()/ys.norm());normal=float((gram@writer-cross).norm()/cross.norm())
  records.append(dict(policy=policy,channels=width,products=2*width,stored_weight_coefficients=3456*width,linear_coefficient_multiplications=5760*width,calibration_full_variation_error=err,normal_equation_error=normal,retained_rank=int(keep.sum())))
  if width==512:programs[policy]=dict(L=L[ids].float(),R=R[ids].float(),channel_mean=phimean[ids],reduced_writers=writer.T,full_mean=mean,indices=ids)
  print(policy,width,err,flush=True)
out=p/'MIDPOINT_CHANNEL_REFIT_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,seconds=time.perf_counter()-start,scope='Original calibration only, empirical paired-input moment; native reader dictionary with exact convex output refit, no output-subspace truncation. Correlation ignores cancellation; random comparator. Not held evidence or novel feature discovery.'),indent=2)+'\n');torch.save(programs,p/'MIDPOINT_CHANNEL_REFIT_V1.pt')
