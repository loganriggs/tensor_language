"""Independent scalar-product decode, both metrics, saved storage and native derivatives."""
from pathlib import Path
import json,torch
from local_shared_reader_graph import component_scalars
from pack_reader_graph_artifacts import counts
P=Path(__file__).parent;torch.set_num_threads(2)
def main(prefix='JOINT_OVERLAP'):
 meta=json.loads((P/f'{prefix}_V1.json').read_text());programs=torch.load(P/f'{prefix}_PROGRAMS_V1.pt',weights_only=True);d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);energies=[t.square().sum((-1,-2)).reshape(3,2).sum(1) for t in (Q,S@Q@S)];ids=d['indices'];z=d['z'][ids];h=d['h'][ids];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();truth=torch.stack([p['truth'][ids] for p in d['pairs']],1);rows=[]
 for rec in meta['records']:
  program=programs[rec['key']];basis=program['input_basis'];hats=[];linear=[];bias=[]
  for j in range(3):
   p=program['pairs'][str(j)];reader=basis.new_empty(1152,384);reader[:,p['shared_indices']]=basis@p['shared_map'];reader[:,p['private_indices']]=p['private_reader'];core=reader.new_zeros(2,384,384)
   for row,(a,b,kind) in enumerate(p['product_indices'].T.tolist()):
    w=p['product_weights'][row]
    if kind==0:core[:,a,a]+=w
    elif kind==1:core[:,a,a]+=w;core[:,b,b]-=w
    else:core[:,a,b]+=.5*w;core[:,b,a]+=.5*w
   hats.append(reader@core@reader.T)
   for key in ('a','b'):linear.append(p[key+'_linear']);bias.append(p[key+'_bias'])
  hat=torch.cat(hats);linear=torch.stack(linear,1);bias=torch.stack(bias);reads=torch.einsum('ni,oij,nj->no',z,hat,z)+z@linear+bias;grad=2*torch.einsum('oij,nj->noi',hat,z)+linear.T[None];native_reads=torch.einsum('ni,oij,nj->no',z,Q,z);native_grad=2*torch.einsum('oij,nj->noi',Q,z);phis=[];J=[];N=[]
  for j in range(3):
   p=program['pairs'][str(j)];a=(h@p['h_reader']-.5*reads[:,2*j])/scale-p['alpha'];b=reads[:,2*j+1]/scale-p['beta'];phis.append(a*b);J.append(-.5*(b/scale)[:,None]*grad[:,2*j]+(a/scale)[:,None]*grad[:,2*j+1]);na=(h@p['h_reader']-.5*native_reads[:,2*j])/scale-p['alpha'];nb=native_reads[:,2*j+1]/scale-p['beta'];N.append(-.5*(nb/scale)[:,None]*native_grad[:,2*j]+(na/scale)[:,None]*native_grad[:,2*j+1])
  phi=torch.stack(phis,1);values=((phi-truth).norm(dim=0)/(truth-truth.mean(0)).norm(dim=0)).tolist();jac=[float((a-b).norm()/b.norm()) for a,b in zip(J,N)];actual=component_scalars(z,h,program);execution=float((actual-phi).norm()/phi.norm());E=hat-Q;errors=[float((e.square().sum((-1,-2)).reshape(3,2).sum(1)/energy).mean().sqrt()) for e,energy in zip((E,S@E@S),energies)];replay=max(execution,abs(errors[0]-rec['native_error']),abs(errors[1]-rec['covariance_error']),max(abs(a-b) for a,b in zip(values,rec['per_mode_errors'])),max(abs(a-b) for a,b in zip(jac,rec['euclidean_jacobian_errors'])))
  test=z[:5].clone().requires_grad_();outputs=component_scalars(test,h[:5],program);auto=[]
  for j in range(3):auto.append(torch.autograd.grad(outputs[:,j].sum(),test,retain_graph=j<2)[0])
  gradient_replay=max(float((g-J[j][:5]).norm()/J[j][:5].norm()) for j,g in enumerate(auto));storage=counts(program);assert storage['logical_float_coefficients']==storage['backing_storage_floats']==996876
  assert replay<1e-8 and gradient_replay<1e-8
  rows.append(dict(key=rec['key'],maximum_metric_and_component_replay=replay,autodiff_derivative_replay=gradient_replay,packed_float_storage=storage['backing_storage_floats'],per_mode_errors=values,native_error=errors[0],covariance_error=errors[1]))
 primary=next(r for r in meta['records'] if r['key']==meta['winners']['calibration_shaped']);base=meta['plan']['baseline'];pred=dict(pred_a_instrument=all(max(r['execution_replay'],r['dense_loss_replay'])<1e-8 for r in meta['records']),pred_b_components=all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),pred_c_metrics=all(primary[k]<=1.1*base[k] for k in ('native_error','covariance_error')),pred_d_arithmetic=primary['source_total_multiplications']<=.8*base['source_total_multiplications'] and primary['stored_floats']<base['stored_floats']);assert pred==meta['predictions']
 (P/f'{prefix}_AUDIT_V1.json').write_text(json.dumps(dict(rows=rows,predictions=pred,scope='Independent product-index quadratic decode, native component and analytic/autodiff response checks, both coefficient norms and packed storage. No new data; opened448notfresh; not a circuit-adoption audit.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
