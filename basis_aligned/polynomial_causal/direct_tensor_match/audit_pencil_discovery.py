"""Positive-result redteam: coordinate changes, output mixing, actual graph cost."""
from pathlib import Path
import json,time,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS,source_reads,price
from pencil_shared_discovery import discover
from joint_orthogonal_private import JointOrthogonalPrivateMetric
from export_orthogonal_private import export
P=Path(__file__).parent;torch.set_num_threads(2);start=time.monotonic();rows=[]
for case in range(5):
 original,bases,private,_,_=fixture(case);d=len(original[0]);r=bases[0].shape[1];rng=torch.Generator().manual_seed(47000+case)
 left=torch.linalg.qr(torch.randn(d,d,dtype=original.dtype,generator=rng)).Q;right=torch.linalg.qr(torch.randn(d,d,dtype=original.dtype,generator=rng)).Q;change=(left*torch.logspace(0,1,d,dtype=original.dtype))@right.T
 for arm in ('identity','input_condition10','output_mix'):
  A=change if arm=='input_condition10' else torch.eye(d,dtype=original.dtype);T=A@original@A.T
  if arm=='output_mix':
   for j in range(3):
    mix=torch.tensor([[1.,.7],[-.2,1.3]],dtype=T.dtype);T[2*j:2*j+2]=torch.einsum('ab,bij->aij',mix,T[2*j:2*j+2])
  try:
   found=discover(T,r);row=dict(case=case,arm=arm,unique=found['success'])
   if not found['success']:row.update(passed=False,reason=found['reason']);rows.append(row);continue
   params=found['bases']+found['private'];metric=JointOrthogonalPrivateMetric(T,[torch.cat([params[a],params[b]],1) for a,b in GROUPS]);loss,states=metric.loss(params,dense=True)
   template=dict(input_bases={str(j):params[j] for j in range(3)},pairs={str(j):dict(a_linear=T.new_zeros(d),b_linear=T.new_zeros(d),a_bias=T.new_zeros(()),b_bias=T.new_zeros(())) for j in range(3)})
   I=torch.eye(d,dtype=T.dtype);graph,diagnostics=export(states,metric.scales,template,I,I);z=torch.randn(32,d,dtype=T.dtype,generator=rng);actual=source_reads(z,graph);target=torch.einsum('ni,oij,nj->no',z,T,z);replay=float((actual-target).norm()/target.norm())
   span=[]
   for true,found_basis in zip(bases,found['bases']):
    q=torch.linalg.qr(A@true,mode='reduced').Q;span.append(float((q@q.T-found_basis@found_basis.T).norm()))
   cost=price(graph);q=2*r+private[0].shape[1];baseline_multiplications=3*d*q+9*q
   row.update(coefficient_error=float(loss.sqrt()),execution_replay=replay,shared_projector_errors=span,source_multiplications=cost['source_total_multiplications'],independent_pair_source_multiplications=baseline_multiplications,products=cost['activation_products'],passed=float(loss.sqrt())<1e-8 and replay<1e-8 and max(span)<1e-6 and cost['source_total_multiplications']<baseline_multiplications)
  except (ValueError,RuntimeError,AssertionError) as error:row=dict(case=case,arm=arm,passed=False,failure=str(error))
  rows.append(row);print(json.dumps(row),flush=True)
out=dict(records=rows,all_passed=all(r['passed'] for r in rows),seconds=time.monotonic()-start,scope='Five unchanged targets under identity, invertible input condition10, and output mixing. Actual pairwise executor versus dense targets on32 artificial probes. Savings concern shared linear projections, not fewer quadratic products. Known planted directions used only after discovery for audit; no native or semantic claim.')
(P/'PENCIL_SHARED_DISCOVERY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
