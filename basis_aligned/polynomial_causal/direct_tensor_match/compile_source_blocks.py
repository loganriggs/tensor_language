"""Five structural controls and exact common-product rewrite of source cores."""
from pathlib import Path
import json,torch
from quadratic_pair_blocks import compile_pair,products
from source_interface import residual_write
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(2240)
Q=torch.linalg.qr(torch.randn(6,6,dtype=torch.float64)).Q;mix=Q@torch.diag(torch.linspace(.7,1.4,6,dtype=torch.float64));toy=[]
def test(name,A,B,expect_reject=False):
 try:
  c=compile_pair(A,B);x=torch.randn(51,len(A),dtype=torch.float64);pred=products(x@c['input_transform'],c['product_indices'])@c['product_weights'];truth=torch.stack([((x@M)*x).sum(1) for M in [A,B]],1);err=float((pred-truth).norm()/truth.norm());assert err<1e-10;assert not expect_reject;toy.append(dict(name=name,status='PASS',replay=err,products=len(c['product_weights'])))
 except ValueError as e:
  if not expect_reject:raise
  toy.append(dict(name=name,status='EXPECTED_REJECTION',reason=str(e)))
A=torch.eye(6,dtype=torch.float64);B=torch.diag(torch.arange(1,7,dtype=torch.float64));test('positive_definite',mix.T@A@mix,mix.T@B@mix)
A=torch.diag(torch.tensor([1.,-2.,3.,-4.,5.,-6.],dtype=torch.float64));B=A@torch.diag(torch.linspace(.2,1.8,6,dtype=torch.float64));test('indefinite_real',mix.T@A@mix,mix.T@B@mix)
A=torch.diag(torch.tensor([1.,-1.],dtype=torch.float64));B=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64);test('complex_pair',A,B)
A=torch.diag(torch.tensor([1.,-2.,3.,-4.,5.,-6.],dtype=torch.float64));B=A@torch.diag(torch.tensor([.2,.2,.2,1.3,1.3,1.3],dtype=torch.float64));test('repeated_real',mix.T@A@mix,mix.T@B@mix)
A=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64);B=torch.tensor([[0.,0.],[0.,1.]],dtype=torch.float64);test('defective_pencil',A,B,True)
programs=torch.load(p/'MIDPOINT_SOURCE_SHARED_DICTIONARY_V1.pt',weights_only=True);cal=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);z=cal['z'].flatten(0,1).double();exports={};records=[]
for name in ['16','24']:
 e=programs[name];A=(e['a_inner_reader']*e['a_eigenvalues'])@e['a_inner_reader'].T;B=(e['b_inner_reader']*e['b_eigenvalues'])@e['b_inner_reader'].T;c=compile_pair(A,B)
 out={k:e[k].clone() for k in ['a_linear','a_bias','b_linear','b_bias','h_reader','residual_writer','alpha','beta']};out.update(shared_reader=e['shared_reader']@c['input_transform'],product_indices=c['product_indices'].clone(),product_weights=c['product_weights'].clone())
 t=z@out['shared_reader'];quad=products(t,out['product_indices'])@out['product_weights'];qa=out['a_bias']+z@out['a_linear']+quad[:,0];qb=out['b_bias']+z@out['b_linear']+quad[:,1];scale=cal['recipient_scale'].flatten().double();hread=cal['h_reader'].flatten().double();phi=((hread-.5*qa)/scale-out['alpha'])*(qb/scale-out['beta'])
 orig=[]
 for k in ['a','b']:orig.append(e[k+'_bias']+z@e[k+'_linear']+(((z@e['shared_reader'])@e[k+'_inner_reader']).square()*e[k+'_eigenvalues']).sum(1))
 ref=((hread-.5*orig[0])/scale-e['alpha'])*(orig[1]/scale-e['beta']);err=float((phi-ref).norm()/ref.norm());assert err<1e-10
 floats=sum(v.numel() for v in out.values() if v.is_floating_point());integers=sum(v.numel() for v in out.values() if not v.is_floating_point());records.append(dict(shared_rank=int(name),**c['diagnostics'],calibration_scalar_replay=err,source_products=len(c['product_weights']),previous_source_products=32,stored_float_scalars=floats,stored_integer_indices=integers,blocks=c['blocks']));exports[name]=out
result=dict(predictions=dict(pred_a_controls=len(toy)==5,pred_b_exact=max(max(r['matrix_replay']+[r['calibration_scalar_replay']]) for r in records)<1e-10,pred_c_product_saving=all(r['source_products']<32 for r in records)),toys=toy,records=records,scope='Exact numerical rewrite of existing approximations; all prior fidelity failures preserved. Shared products reused across both source forms. Native z,h inputs and RMS remain.');(p/'MIDPOINT_SOURCE_BLOCK_COMPILER_V1.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(exports,p/'MIDPOINT_SOURCE_BLOCK_PROGRAMS_V1.pt');print(json.dumps(result,indent=2))
