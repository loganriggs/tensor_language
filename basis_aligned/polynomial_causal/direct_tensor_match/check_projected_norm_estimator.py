"""Exhaustive independent-sign validation of projected-plus-residual tensor energy."""
import itertools,json
from pathlib import Path
import torch
from implicit_quartic import entries
from quartic_cp import directional
from projected_quartic_energy import projected_energy
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1858);d=3;teacher=[torch.randn(*shape) for shape in [(2,4),(4,3),(4,3),(3,5),(5,d),(5,d)]];idx=torch.cartesian_prod(*[torch.arange(d)]*4);exact=entries(*teacher,idx).square().sum();signs=torch.tensor(list(itertools.product([-1.,1.],repeat=d)));choices=torch.cartesian_prod(*[torch.arange(len(signs))]*4);vectors=[signs[choices[:,j]] for j in range(4)];raw=directional(*teacher,vectors).square().sum(1);Q=torch.linalg.qr(torch.randn(d,d))[0];rows=[]
 for rank in [1,2]:
  B=Q[:,:rank];projected=[x@B@B.T for x in vectors];residual=directional(*teacher,vectors)-directional(*teacher,projected);known=projected_energy(teacher,B);estimate=known+residual.square().sum(1).mean();error=float(abs(estimate-exact)/exact);assert error<1e-12;rows.append(dict(rank=rank,relative_error=error,projected_energy_fraction=float(known/exact),raw_variance=float(raw.var()),residual_variance=float(residual.square().sum(1).var())))
 out=dict(records=rows,scope='Exhaustive allindependentRademacher inputs verify orthogonal coefficient projection plus residualenergy identity. Variance reduction is targetdependent, not guaranteed for every projection.')
 (P/'PROJECTED_NORM_ESTIMATOR_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
