"""Dense coefficient check of best toy fits, avoiding subtractive loss cancellation."""
from pathlib import Path
import torch,json,itertools
from quartic_pair_metric import numerator_forms
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);fits=torch.load(p/'QUARTIC_NUMERATOR_TOY_FITS_V1.pt',weights_only=True);rows=[]
def dense(A,B):
 raw=torch.einsum('ij,kl->ijkl',A,B);return sum(raw.permute(order) for order in itertools.permutations(range(4)))/24
for name,v in fits.items():
 A,B=numerator_forms(v['Qa'],v['Qb'],.7,-.4);qa=(v['U']*v['signA'])@v['U'].T;qb=(v['V']*v['signB'])@v['V'].T;C,D=numerator_forms(qa,qb,.7,-.4);ref=dense(A,B);pred=dense(C,D);error=float((pred-ref).norm()/ref.norm());rows.append(dict(structure=name,dense_relative_coefficient_error=error,source_matrix_errors=[float((qa-v['Qa']).norm()/v['Qa'].norm()),float((qb-v['Qb']).norm()/v['Qb'].norm())]));assert error<1e-3
out=dict(records=rows,recovered_structures=sum(r['dense_relative_coefficient_error']<1e-3 for r in rows),scope='Independent explicit symmetric coefficient tensors for bestsavedtoyfits. Implicit subtraction can report zero near machine precision; use these noncancelling residual norms. Best-of-sweep toy recovery is not a universal optimizer comparison.');(p/'QUARTIC_NUMERATOR_TOY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
