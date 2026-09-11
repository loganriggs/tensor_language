"""Exact norm-squared correction of an existing bilinear reader program.

Input is already at the normalized MLP interface. Feature edits keep this
separate radial background fixed. Input-response coefficients instead perturb
the interface input itself; the caller owns upstream normalization.
"""
import torch
from torch import nn

class RadialCorrectedReader(nn.Module):
    def __init__(self,base,delta):
        super().__init__();self.base=base
        assert delta.shape==base.bias.shape
        self.register_buffer('delta',delta.to(base.down).detach())

    def correction(self,x):
        return x.square().sum(-1,keepdim=True)/x.shape[-1]*self.delta

    def forward(self,x):return self.base(x)+self.correction(x)

    def remove_features(self,x,indices):
        return self.base.remove_features(x,indices)+self.correction(x)

    def interchange_features(self,x,donor,indices):
        return self.base.interchange_features(x,donor,indices)+self.correction(x)

    def response_coefficients(self,x,feature_delta):
        y,b,a=self.base.response_coefficients(x,feature_delta)
        return y+self.correction(x),b,a

    def input_response_coefficients(self,x,dx):
        y,b,a=self.base.response_coefficients(x,self.base.features(dx))
        linear=2*(x*dx).sum(-1,keepdim=True)/x.shape[-1]*self.delta
        return y+self.correction(x),b+linear,a+self.correction(dx)

    def price(self):
        return dict(base=self.base.price(),additional_float_coefficients=self.delta.numel(),
            input_squares_per_example=self.base.analysis_basis.shape[1],
            radial_output_scalings_per_example=self.delta.numel())


if __name__=='__main__':
    import json
    from pathlib import Path
    from rectangular_sparse_reader_v1 import RectangularSparseReaderProgram
    torch.manual_seed(841);torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    n,dim,out=11,7,4
    l,r,down=torch.randn(n,dim),torch.randn(n,dim),torch.randn(out,n)
    basis=torch.randn(10,dim)
    ids=torch.stack([torch.randperm(10)[:3] for _ in range(2*n)])
    values=torch.randn(2*n,3);bias=torch.randn(out)
    base=RectangularSparseReaderProgram(basis,ids,values,down,bias)
    reader=torch.sparse.mm(base.codes,basis);a,b=reader.split(n)
    native=torch.einsum('oj,ji,jk->oik',down,l,r);native=(native+native.transpose(1,2))/2
    candidate=torch.einsum('oj,ji,jk->oik',down,a,b);candidate=(candidate+candidate.transpose(1,2))/2
    diff=native-candidate;delta=diff.diagonal(dim1=1,dim2=2).sum(1)
    corrected=candidate+delta[:,None,None]*torch.eye(dim)/dim
    program=RadialCorrectedReader(base,delta)
    x,dx=torch.randn(5,dim),torch.randn(5,dim)
    direct=torch.einsum('ni,oij,nj->no',x,corrected,x)+bias
    execution=float((program(x)-direct).norm()/direct.norm())
    projection=float(abs((native-corrected).square().sum()-(diff.square().sum()-delta.square().sum()/dim))/diff.square().sum())
    y0,bb,aa=program.input_response_coefficients(x,dx)
    dose=max(float((program(x+t*dx)-(y0+t*bb+t*t*aa)).norm()/direct.norm()) for t in (-1.,.25,2.))
    trace=float((native-corrected).diagonal(dim1=1,dim2=2).sum(1).abs().max())
    result=dict(predictions=dict(pred_a_execution=execution<=1e-10,pred_b_projection=projection<=1e-10,
        pred_c_dose=dose<=1e-10,pred_d_trace=trace<=1e-10),execution_error=execution,
        projection_error=projection,input_dose_error=dose,trace_error=trace,price=program.price(),
        scope='Exact synthetic coefficient projection and interface-input dose algebra, not native behavioral validation')
    Path(__file__).with_name('RADIAL_CORRECTED_READER_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert all(result['predictions'].values())
