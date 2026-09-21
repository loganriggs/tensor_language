"""Evaluate existing fits in the exact Gaussian numerator moment metric."""
import json
from pathlib import Path
import torch
from gaussian_quartic_moment import squared_error_by_degree,teacher_norm_by_degree
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=data['common'];f=data['frames']['covariance']
fits=json.loads((P/'FUNCTIONAL_QUARTIC_MODE3_FIT_V1.json').read_text())
programs=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_PROGRAMS_V1.pt',weights_only=True)
fixed=json.loads((P/'FIXED_QUARTIC_PRODUCT_FIT_V1.json').read_text())
fixedprograms=torch.load(P/'FIXED_QUARTIC_PRODUCT_PROGRAMS_V1.pt',weights_only=True)
entries=[('initial',programs['initial'])]
for family in ['coefficient_0.0','normalized_0.0','normalized_0.01']:
 name=fits['selected_by_training_objective'][family];entries.append((name,programs[name]))
name=fixed['selected_by_training_objective']['0.01'];entries.append((name,fixedprograms[name]))
norms=teacher_norm_by_degree(f['A'],f['B']);variance=norms[1:].sum();records=[]
for name,program in entries:
 matrices=[]
 for k,factor in [('a',-.5),('b',1.)]:
  U=torch.linalg.solve(c['source_inverse_root'],program[k+'_reader']);w=program[k+'_eigenvalues']
  X=torch.cat([f['J']@U,f[k+'_fixed_vectors'],f['unit'][:,None]],1)
  weights=torch.cat([factor*w,f[k+'_fixed_values'],(-factor*(U.square().sum(0)*w).sum()).reshape(1)])
  matrices.append((X*weights)@X.T)
 error=squared_error_by_degree(f['A'],f['B'],*matrices)
 record=dict(program=name,gaussian_numerator_variation_error=float((error.sum()/variance).sqrt()),
  error_energy_fractions_by_hermite_degree=(error/error.sum()).tolist())
 records.append(record);print(record,flush=True)
out=dict(teacher_energy_fractions_by_hermite_degree=(norms/norms.sum()).tolist(),records=records,
 scope='Exact Gaussian numerator moment metric with final coordinate fixed to one, using training joint covariance and means. Does not average inverse sampled RMS or assert Gaussian inputs match native states.')
(P/'GAUSSIAN_QUARTIC_MODE3_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
