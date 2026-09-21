"""Independent saved-program replay and train/evaluation covariance geometry."""
from pathlib import Path
import json
import torch
from gaussian_quartic_moment import squared_error_by_degree,teacher_norm_by_degree
from source_interface import source_read
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=data['common'];f=data['frames']['covariance']
fit=json.loads((P/'GAUSSIAN_QUARTIC_MODE3_FIT_V1.json').read_text());name=fit['selected_by_gaussian_loss']['covariance'];row=next(r for r in fit['records'] if r['label']==name)
program=torch.load(P/'GAUSSIAN_QUARTIC_MODE3_PROGRAMS_V1.pt',weights_only=True)[name];forms=[]
for k,factor in [('a',-.5),('b',1.)]:
 U=torch.linalg.solve(c['source_inverse_root'],program[k+'_reader']);sign=program[k+'_eigenvalues']
 X=torch.cat([f['J']@U,f[k+'_fixed_vectors'],f['unit'][:,None]],1);w=torch.cat([factor*sign,f[k+'_fixed_values'],(-factor*(U.square().sum(0)*sign).sum()).reshape(1)])
 forms.append((X*w)@X.T)
degrees=squared_error_by_degree(f['A'],f['B'],*forms);norm=teacher_norm_by_degree(f['A'],f['B'])[1:].sum();dense=float((degrees.sum()/norm).sqrt())
assert abs(dense-row['gaussian_numerator_variation_error'])<1e-8
affine_errors=[float((M[:-1,-1]-T[:-1,-1]).norm()/T[:-1,-1].norm()) for M,T in zip(forms,[f['A'],f['B']])]
assert max(affine_errors)<1e-10
cal=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);z=cal['z'].flatten(0,1).double()
indices=torch.tensor([j for i in fit['evaluation_documents'] for j in range(i*64,(i+1)*64)])
qa=source_read(z,program,'a');qb=source_read(z,program,'b')
phi=((c['t']-.5*qa)/c['scale']-c['alpha'])*(qb/c['scale']-c['beta']);truth=c['true_phi'][indices]
native=float((phi[indices]-truth).norm()/(truth-truth.mean()).norm())
assert abs(native-row['native_opened_evaluation_error'])<1e-8
delta=z-c['mu'];train=delta[:24*64];evaluation=delta[indices];cov=train.T@train/len(train)
ev,Q=torch.linalg.eigh(cov);trainproj=(train@Q).square().mean(0);evalproj=(evaluation@Q).square().mean(0)
geometry=[]
for shrink in [0.,.001,.01,.1,1.]:
 diag=(1-shrink)*ev+shrink*ev.mean()
 a=float((trainproj/diag).sum());b=float((evalproj/diag).sum())
 geometry.append(dict(isotropic_shrinkage=shrink,train_mean_squared_mahalanobis=a,evaluation_mean_squared_mahalanobis=b,evaluation_train_ratio=b/a))
out=dict(selected_program=name,dense_gaussian_error=dense,export_native_error=native,fixed_affine_relative_errors=affine_errors,covariance_eigenvalue_min=float(ev.min()),covariance_eigenvalue_max=float(ev.max()),covariance_condition=float(ev.max()/ev.min()),geometry=geometry,scope='Independent dense/export audit; covariance shift diagnostic uses input states only and does not refit programs or establish that shrinkage improves functional fidelity.')
(P/'GAUSSIAN_FIT_GEOMETRY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
