"""Executable shared-reader bilinear program and exact feature-edit algebra.

Input is the bilinear module's already-normalized input. Feature edits happen
inside that module; they do not renormalize the input or edit the residual.
Bias is retained. The caller owns residual addition and final readout.
"""
import torch
from torch import nn


class SparseReaderProgram(nn.Module):
    def __init__(self, analysis_basis, code_indices, code_values, down, bias=None):
        super().__init__()
        assert analysis_basis.ndim == 2 and analysis_basis.shape[0] == analysis_basis.shape[1]
        dim = analysis_basis.shape[0]
        assert code_indices.shape == code_values.shape and code_indices.ndim == 2
        assert code_indices.shape[0] == 2 * down.shape[1]
        assert code_indices.dtype in (torch.int16,torch.int32,torch.int64)
        assert bool((code_indices >= 0).all() and (code_indices < dim).all())
        n, k = code_indices.shape
        rows = torch.arange(n,device=analysis_basis.device).repeat_interleave(k)
        indices = torch.stack((rows,code_indices.to(device=analysis_basis.device,dtype=torch.long).reshape(-1)))
        codes = torch.sparse_coo_tensor(indices,code_values.to(analysis_basis).reshape(-1),
            (n,dim),device=analysis_basis.device,dtype=analysis_basis.dtype).coalesce()
        self.register_buffer('analysis_basis',analysis_basis.detach())
        self.register_buffer('codes',codes.detach())
        self.register_buffer('down',down.to(analysis_basis).detach())
        self.register_buffer('bias',torch.zeros(down.shape[0],device=analysis_basis.device,dtype=analysis_basis.dtype)
                             if bias is None else bias.to(analysis_basis).detach())
        assert self.bias.shape == (down.shape[0],)

    @classmethod
    def from_artifact(cls, artifact, down, bias=None):
        basis=artifact['analysis_basis'].to(device=down.device)
        return cls(basis,artifact['code_indices'],artifact['code_values'],down,bias)

    def features(self,x):
        return x @ self.analysis_basis.T

    def reads(self,h):
        flat=h.reshape(-1,h.shape[-1])
        output=torch.sparse.mm(self.codes,flat.T.contiguous()).T
        output=output.reshape(*h.shape[:-1],self.codes.shape[0])
        return output.split(self.down.shape[1],dim=-1)

    def write(self,products):
        return products @ self.down.T

    def forward_features(self,h):
        left,right=self.reads(h)
        return self.write(left*right)+self.bias

    def forward(self,x):
        return self.forward_features(self.features(x))

    def selected(self,h,indices):
        ids=torch.as_tensor(indices,device=h.device,dtype=torch.long)
        assert ids.ndim==1 and bool(((ids>=0)&(ids<h.shape[-1])).all())
        mask=h.new_zeros(h.shape[-1]).index_fill_(0,ids,1)
        return h*mask

    def remove_features(self,x,indices):
        h=self.features(x)
        return self.forward_features(h-self.selected(h,indices))

    def interchange_features(self,x,donor,indices):
        assert donor.shape==x.shape
        h,other=self.features(x),self.features(donor)
        return self.forward_features(h+self.selected(other-h,indices))

    def response_coefficients(self,x,feature_delta):
        """Return y0,b,a such that y(t)=y0+t*b+t^2*a exactly."""
        h=self.features(x)
        assert feature_delta.shape==h.shape
        left,right=self.reads(h)
        dl,dr=self.reads(feature_delta)
        return (self.write(left*right)+self.bias,
                self.write(dl*right+left*dr),self.write(dl*dr))

    def disjoint_interaction(self,x,first,second):
        assert not set(first).intersection(second),'Feature sets must be disjoint'
        h=self.features(x)
        ls,rs=self.reads(self.selected(h,first))
        lt,rt=self.reads(self.selected(h,second))
        return self.write(ls*rt+lt*rs)

    def price(self):
        return dict(float_coefficients=self.analysis_basis.numel()+self.codes._nnz()+self.down.numel()+self.bias.numel(),
            code_nonzeros=self.codes._nnz(),runtime_coo_index_entries=self.codes.indices().numel(),
            products=self.down.shape[1],input_features=self.analysis_basis.shape[0],
            scope='Program buffers only; residual, unembedding and other native operations separately retained.')
