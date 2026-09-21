"""Necessary condition for sharing the same real squares across both source cores.
If invertible A and B have a real simultaneous congruence diagonalization, A^-1 B
has only real eigenvalues. Robustly complex eigenvalues obstruct that restricted
exact rewrite; not general product sharing, block forms or approximate circuits.
"""
from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);programs=torch.load(p/'MIDPOINT_SOURCE_SHARED_DICTIONARY_V1.pt',weights_only=True);records=[]
for k in ['8','16','24']:
 e=programs[k];A=(e['a_inner_reader']*e['a_eigenvalues'])@e['a_inner_reader'].T;B=(e['b_inner_reader']*e['b_eigenvalues'])@e['b_inner_reader'].T
 if int(k)>16:
  # Individual rank16 cores are singular here; use a regular generic pencil base.
  A,B=A+B,A-0.37*B
 cond=float(torch.linalg.cond(A));M=torch.linalg.solve(A,B);values,V=torch.linalg.eig(M);res=float((M.to(torch.complex128)@V-V*values).norm()/M.norm());complex_mask=values.imag.abs()>1e-7*(1+values.abs());records.append(dict(shared_rank=int(k),base_condition=cond,eigen_residual=res,complex_eigenvalues=int(complex_mask.sum()),max_imaginary=float(values.imag.abs().max()),eigenvalues=[[float(v.real),float(v.imag)] for v in values]))
# Two planted controls: shared diagonal pair and indefinite non-SDC complex pencil.
A=torch.eye(3,dtype=torch.float64);B=torch.diag(torch.tensor([1.,2.,3.],dtype=torch.float64));assert torch.linalg.eigvals(torch.linalg.solve(A,B)).imag.abs().max()==0
A=torch.diag(torch.tensor([1.,-1.],dtype=torch.float64));B=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64);assert (torch.linalg.eigvals(torch.linalg.solve(A,B)).imag.abs()>.9).all()
out=dict(records=records,controls='PASS real diagonal pair and complex indefinite pair',scope='Numerical necessary-condition screen for exact common-square congruence. Complex eigenvalues with low residual obstruct simultaneous real diagonalization when pencil base is nonsingular; no claim of impossibility for broader arithmetic DAGs.');(p/'MIDPOINT_SHARED_SOURCE_PENCIL_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({**out,'records':[{k:v for k,v in r.items() if k!='eigenvalues'} for r in records]},indent=2))
