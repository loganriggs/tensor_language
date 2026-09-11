"""All-token native-legal atom updates equal residual-width whitened updates."""
import json
from pathlib import Path
import torch
from sparse_product_dictionary_v1 import atom_sweep,form
from joint_quadratic_fit_v1 import product_cross


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1061)
    u=torch.randn(17,5);root=torch.linalg.cholesky(u.T@u)
    p=torch.linalg.solve_triangular(root,u.T,upper=False).T
    l,r,down=torch.randn(11,7),torch.randn(11,7),torch.randn(5,11)
    z=torch.randn(5,4);a=p@z
    pl,pr=torch.randn(4,7),torch.randn(4,7)
    norm=product_cross(pl,pr,pl,pr).diag().sqrt()
    pl/=norm.sqrt()[:,None];pr/=norm.sqrt()[:,None]
    full_l,full_r=pl.clone(),pr.clone();small_l,small_r=pl.clone(),pr.clone()
    full=atom_sweep(u,l,r,down,full_l,full_r,a)
    small=atom_sweep(root.T,l,r,down,small_l,small_r,z)
    full_forms=torch.stack([form(full_l[j:j+1],full_r[j:j+1],torch.ones(1)) for j in range(4)])
    small_forms=torch.stack([form(small_l[j:j+1],small_r[j:j+1],torch.ones(1)) for j in range(4)])
    function_error=float((full_forms-small_forms).norm()/full_forms.norm())
    scaled_gain_error=abs(full['sum_conditional_gain']-small['sum_conditional_gain']*5/17)/max(1.,full['sum_conditional_gain'])
    full_target=torch.stack([form(l,r,c) for c in u@down])
    small_target=torch.stack([form(l,r,c) for c in root.T@down])
    full_loss=(full_target-torch.einsum('vk,kij->vij',a,full_forms)).square().sum()
    small_loss=(small_target-torch.einsum('vk,kij->vij',z,small_forms)).square().sum()
    loss_error=float(abs(full_loss-small_loss)/full_loss)
    result=dict(instrument_passed=max(function_error,scaled_gain_error,loss_error)<1e-10,
                atom_function_error=function_error,normalized_gain_error=scaled_gain_error,
                full_coefficient_loss_error=loss_error,
                scope='Exact full-U legal writer and product update equivalence in whitened output coordinates; fixes interface by parameterization, does not prove joint convergence or physical circuit properties.')
    Path(__file__).with_name('LEGAL_PRODUCT_ATOM_UPDATE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
