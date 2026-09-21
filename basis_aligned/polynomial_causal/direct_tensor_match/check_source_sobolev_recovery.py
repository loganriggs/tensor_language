"""Envelope-gradient replay and five known mixed-product recovery controls."""
from pathlib import Path
import torch,json,math,time,argparse
from shared_quadratic_products import materialize_mixed
from source_sobolev import SourceSobolev
def profiled_loss(T,L,R,detach_weights=True):
 H=torch.diag(torch.logspace(-1,1,T.shape[-1],dtype=T.dtype));return SourceSobolev(T,H,1).loss(L,R,detach=detach_weights)
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--only-case',type=int);parser.add_argument('--restart-seed',type=int);parser.add_argument('--learning-rate',type=float,default=.02);args=parser.parse_args()
P=Path(__file__).resolve().parent;torch.set_num_threads(2);start=time.perf_counter();records=[]
for seed in (range(5) if args.only_case is None else [args.only_case]):
 g=torch.Generator().manual_seed(470+seed);d=8;r=4;o=4
 A=torch.randn(d,r,generator=g,dtype=torch.float64);B=torch.randn(d,r,generator=g,dtype=torch.float64);W=torch.randn(o,r,generator=g,dtype=torch.float64)
 if seed==1:W=torch.eye(4,dtype=torch.float64)
 if seed==2:A[:,1:]=A[:,:1]
 if seed==3:W[1]=-W[0]
 if seed==4:A,B=A+B,A-B
 T=materialize_mixed(A,B,W)
 if args.restart_seed is not None:g=torch.Generator().manual_seed(args.restart_seed)
 L=torch.randn(d,r,generator=g,dtype=torch.float64);R=torch.randn(d,r,generator=g,dtype=torch.float64);L=(L/L.norm(dim=0)).requires_grad_();R=(R/R.norm(dim=0)).requires_grad_()
 loss,_=profiled_loss(T,L,R);reference,_=profiled_loss(T,L,R,detach_weights=False)
 a=torch.autograd.grad(loss,[L,R]);b=torch.autograd.grad(reference,[L,R]);grad=max(float((x-y).norm()/x.norm()) for x,y in zip(a,b));assert grad<1e-9
 opt=torch.optim.Adam([L,R],lr=args.learning_rate)
 for step in range(args.steps):
  opt.param_groups[0]['lr']=args.learning_rate*.5*(1+math.cos(math.pi*step/args.steps));opt.zero_grad();value,_=profiled_loss(T,L,R);value.backward();opt.step()
  with torch.no_grad():L.div_(L.norm(dim=0));R.div_(R.norm(dim=0))
 with torch.no_grad():
  _,weights=profiled_loss(T,L,R);error=float((materialize_mixed(L,R,weights)-T).norm()/T.norm())
 record=dict(seed=470+seed,envelope_gradient_replay=grad,coefficient_error=error,recovered=error<1e-4);records.append(record);print(record,flush=True)
out=dict(records=records,all_gradient_checks_pass=True,all_fits_recovered=all(r['recovered'] for r in records),steps=args.steps,learning_rate=args.learning_rate,restart_seed=args.restart_seed,ridge=1e-10,seconds=time.perf_counter()-start)
(P/('SOURCE_SOBOLEV_RECOVERY_V1.json' if args.only_case is None else f'SOURCE_SOBOLEV_CASE{args.only_case}_SEED{args.restart_seed}_V1.json')).write_text(json.dumps(out,indent=2)+'\n')
