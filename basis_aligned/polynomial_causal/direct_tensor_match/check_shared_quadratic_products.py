"""Five planted structures: implicit loss/gradient and actual fitted recovery.
Random starts, Adam lr.05, 1200steps; no changing capacity from results.
Explicit dense residual scores the final fit to avoid cancellation floors.
"""
import json,torch,time,argparse,math
from pathlib import Path
from shared_quadratic_products import coefficient_loss,materialize
parser=argparse.ArgumentParser();parser.add_argument('--schedule',choices=['constant','cosine'],default='constant');parser.add_argument('--steps',type=int,default=1200);args=parser.parse_args()
P=Path(__file__).resolve().parent;torch.set_num_threads(2);start=time.perf_counter();records=[]
for index,name in enumerate(['shared_outputs','private_outputs','rotated_inputs','signed_overlap','repeated_products']):
 g=torch.Generator().manual_seed(270+index);d=8;r=4;o=6
 V=torch.randn(d,r,generator=g,dtype=torch.float64)/d**.5
 W=torch.randn(o,r,generator=g,dtype=torch.float64)
 if name=='private_outputs':W.zero_();W[:r,:]=torch.eye(r,dtype=torch.float64)
 if name=='rotated_inputs':V=torch.linalg.qr(torch.randn(d,d,generator=g,dtype=torch.float64)).Q@V
 if name=='signed_overlap':W[1]=-W[0];W[3]=W[2]+W[0]
 if name=='repeated_products':V[:,1]=V[:,0];W[:,1]=-0.75*W[:,0]
 teacher=materialize(V,W)
 reader=(torch.randn(d,r,generator=g,dtype=torch.float64)/d**.5).requires_grad_()
 weights=torch.randn(o,r,generator=g,dtype=torch.float64,requires_grad=True)
 implicit=coefficient_loss(teacher,reader,weights)
 explicit=(materialize(reader,weights)-teacher).square().sum()/teacher.square().sum()
 grads1=torch.autograd.grad(implicit,[reader,weights],retain_graph=True)
 grads2=torch.autograd.grad(explicit,[reader,weights])
 graderror=max(float((a-b).norm()/a.norm()) for a,b in zip(grads1,grads2))
 valueerror=float((implicit-explicit).abs().detach());assert graderror<1e-10 and valueerror<1e-10
 optimizer=torch.optim.Adam([reader,weights],lr=.05)
 # Fit the implicit objective; final explicit residual avoids cancellation floors.
 for step in range(args.steps):
  if args.schedule=='cosine':
   optimizer.param_groups[0]['lr']=.05*.5*(1+math.cos(math.pi*step/args.steps))
  optimizer.zero_grad();loss=coefficient_loss(teacher,reader,weights);loss.backward();optimizer.step()
 error=float(((materialize(reader,weights)-teacher).norm()/teacher.norm()).detach())
 record=dict(structure=name,gradient_replay=graderror,value_replay=valueerror,random_start_recovery_error=error,recovered=error<1e-4,steps=args.steps,schedule=args.schedule)
 records.append(record);print(record,flush=True)
out=dict(records=records,all_instrument_checks_pass=True,all_fits_recovered=all(r['recovered'] for r in records),seconds=time.perf_counter()-start,scope='Five planted symmetric shared-product structures; loss and gradients independently match dense tensors. Recovery failures retained; no trained-model result.')
(P/('SHARED_QUADRATIC_PRODUCTS_TOYS_V1.json' if args.schedule=='constant' and args.steps==1200 else f'SHARED_QUADRATIC_PRODUCTS_TOYS_{args.schedule.upper()}_{args.steps}_V1.json')).write_text(json.dumps(out,indent=2)+'\n')
