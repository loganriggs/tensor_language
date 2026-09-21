"""CPU diagnostic: saved export replay and numerator versus normalized error."""
import json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from source_interface import source_read
from quartic_pair_metric import inner
torch.set_num_threads(2)
x=torch.load(P/'JOINT_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=x['common']
z=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True)['z'].flatten(0,1).double()
programs=torch.load(P/'JOINT_QUARTIC_MODE3_PROGRAMS_V1.pt',weights_only=True)
fit=json.loads((P/'JOINT_QUARTIC_MODE3_FIT_V1.json').read_text());truth=c['true_phi'];s=c['scale'];records=[]
def stats(actual,pred):
 e=pred-actual;center=actual-actual.mean();den=center.norm()
 return dict(relative_variation_error=float(e.norm()/den),relative_centered_error=float((e-e.mean()).norm()/den),bias_fraction_of_error_energy=float(e.mean().square()*e.numel()/e.square().sum()))
for name in ['initial',*fit['selected_by_coefficient_loss'].values()]:
 p=programs[name];a=source_read(z,p,'a');b=source_read(z,p,'b')
 phi=((c['t']-.5*a)/s-c['alpha'])*(b/s-c['beta'])
 native=stats(truth,phi);num=stats(truth*s.square(),phi*s.square())
 expected=fit['initial_native_calibration_variation_error'] if name=='initial' else next(r['native_calibration_variation_error'] for r in fit['records'] if r['label']==name)
 assert abs(native['relative_variation_error']-expected)<1e-10
 record=dict(program=name,normalized=native,numerator=num,export_native_error_replay=abs(native['relative_variation_error']-expected))
 if name!='initial':
  metric=next(k for k,v in fit['selected_by_coefficient_loss'].items() if v==name);f=x['frames'][metric];forms=[]
  for k,factor in [('a',-.5),('b',1.)]:
   U=torch.linalg.solve(c['source_inverse_root'],p[k+'_reader']);sign=p[k+'_eigenvalues']
   X=torch.cat([f['J']@U,f[k+'_fixed_vectors'],f['unit'][:,None]],1)
   weights=torch.cat([factor*sign,f[k+'_fixed_values'],(-factor*(U.square().sum(0)*sign).sum()).reshape(1)])
   forms.append((X*weights)@X.T)
  A,B=f['A'],f['B'];C,D=forms
  dense=(inner(A,B,A,B)+inner(C,D,C,D)-2*inner(A,B,C,D))/inner(A,B,A,B)
  expected_coeff=next(r['coefficient_relative_error'] for r in fit['records'] if r['label']==name)
  record['independent_dense_coefficient_error']=float(dense.clamp_min(0).sqrt())
  assert abs(record['independent_dense_coefficient_error']-expected_coeff)<1e-8
 records.append(record)
result=dict(records=records,rms_min=float(s.min()),rms_max=float(s.max()),rms_mean=float(s.mean()),scope='Calibration-only diagnostic. Dense-matrix contraction independently checks lowrank implementation; no fresh or native intervention evidence.')
(P/'JOINT_QUARTIC_MODE3_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
