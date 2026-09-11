"""Original-coordinate block-error bound for a real block-scalar witness."""
import json
from pathlib import Path
import torch


def measure(forms,change,labels,block_values):
    """Z=W diag(block_values[labels]) W^-1; no fitting or native model access."""
    inverse=torch.linalg.inv(change)
    diagonal=torch.diag(block_values[labels])
    z=change@diagonal@inverse
    residual=forms@z-z.T@forms
    transformed=change.T@forms@change
    off=labels[:,None]!=labels[None,:]
    kept=transformed.clone();kept[:,off]=0
    reconstructed=inverse.T@kept@inverse
    energy=forms.square().sum()
    error=float((forms-reconstructed).square().sum()/energy)
    gaps=(block_values[:,None]-block_values[None,:]).abs()
    separation=float(gaps[~torch.eye(len(block_values),dtype=torch.bool)].min())
    condition=float(torch.linalg.cond(change))
    residual_energy=float(residual.square().sum()/energy)
    bound=condition**4*residual_energy/separation**2
    return dict(relative_squared_block_error=error,relative_squared_algebraic_residual=residual_energy,
                bound_on_relative_squared_error=bound,condition=condition,separation=separation)


def control():
    torch.manual_seed(977);torch.set_default_dtype(torch.float64)
    labels=torch.tensor([0,0,0,1,1,1,2,2]);d=len(labels)
    q=[]
    for _ in range(5):
        x=torch.randn(d,d);x=(x+x.T)/2;x[labels[:,None]!=labels[None,:]]=0;q.append(x)
    q=torch.stack(q)
    a=torch.linalg.qr(torch.randn(d,d)).Q;b=torch.linalg.qr(torch.randn(d,d)).Q
    change=a@torch.diag(torch.linspace(1.,3.,d))@b.T;inverse=torch.linalg.inv(change)
    forms=inverse.T@q@inverse
    noise=torch.randn_like(forms);noise=(noise+noise.transpose(1,2))/2
    rows={}
    for strength in [0.,.01,.1]:
        rows[str(strength)]=measure(forms+strength*noise,change,labels,torch.tensor([-1.,.5,2.]))
    # Witness rescaling and scalar identity shifts must preserve the bound.
    scaled=measure(forms+.1*noise,change,labels,torch.tensor([-1.,.5,2.])*7+13)
    scale_error=abs(scaled['bound_on_relative_squared_error']/rows['0.1']['bound_on_relative_squared_error']-1)
    passed=all(r['relative_squared_block_error']<=r['bound_on_relative_squared_error']+1e-25 for r in rows.values()) and rows['0.0']['relative_squared_block_error']<1e-25 and rows['0.1']['relative_squared_block_error']>1e-3 and scale_error<1e-10
    result=dict(instrument_passed=passed,rows=rows,witness_scale_shift_relative_error=scale_error,
                scope='Synthetic real block-scalar witness bound. General complex blocks require a Sylvester-separation bound; native block recovery untested.')
    Path(__file__).with_name('CONGRUENCE_BLOCK_ERROR_TRANSFER_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    assert passed,result
    return result


if __name__=='__main__':print(json.dumps(control(),indent=2))
