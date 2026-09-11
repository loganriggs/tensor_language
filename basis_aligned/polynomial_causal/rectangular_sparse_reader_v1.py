"""Rectangular input feature interface; reuse all existing execution/edit algebra."""
import torch
from torch import nn
from sparse_reader_program_v1 import SparseReaderProgram

class RectangularSparseReaderProgram(SparseReaderProgram):
    def __init__(self,analysis_basis,code_indices,code_values,down,bias=None):
        nn.Module.__init__(self)
        assert analysis_basis.ndim==2
        features,input_dimension=analysis_basis.shape
        assert features>0 and input_dimension>0
        assert code_indices.ndim==2 and code_indices.shape==code_values.shape
        assert code_indices.shape[0]==2*down.shape[1]
        assert code_indices.dtype in (torch.int16,torch.int32,torch.int64)
        assert bool(((code_indices>=0)&(code_indices<features)).all())
        n,k=code_indices.shape
        rows=torch.arange(n,device=analysis_basis.device).repeat_interleave(k)
        indices=torch.stack((rows,code_indices.to(device=analysis_basis.device,dtype=torch.long).reshape(-1)))
        codes=torch.sparse_coo_tensor(indices,code_values.to(analysis_basis).reshape(-1),
            (n,features),device=analysis_basis.device,dtype=analysis_basis.dtype).coalesce()
        self.register_buffer('analysis_basis',analysis_basis.detach())
        self.register_buffer('codes',codes.detach())
        self.register_buffer('down',down.to(analysis_basis).detach())
        self.register_buffer('bias',torch.zeros(down.shape[0],device=analysis_basis.device,dtype=analysis_basis.dtype)
                             if bias is None else bias.to(analysis_basis).detach())
        assert self.bias.shape==(down.shape[0],)

    def price(self):
        result=super().price()
        result['input_dimension']=self.analysis_basis.shape[1]
        return result


if __name__=='__main__':
    import json
    from pathlib import Path
    torch.manual_seed(816);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    basis=torch.randn(24,12);basis/=basis.norm(dim=1,keepdim=True)
    ids=torch.stack([torch.randperm(24)[:3] for _ in range(20)])
    values=torch.randn(20,3);down=torch.randn(7,10);bias=torch.randn(7)
    artifact=dict(analysis_basis=basis,code_indices=ids,code_values=values)
    program=RectangularSparseReaderProgram.from_artifact(artifact,down,bias)
    code=torch.zeros(20,24).scatter_(1,ids,values);readers=code@basis
    x=torch.randn(9,12)
    direct=((x@readers[:10].T)*(x@readers[10:].T))@down.T+bias
    execution=float((program(x)-direct).norm()/direct.norm())
    interaction=program(x)-program.remove_features(x,[0,2])-program.remove_features(x,[1,3])+program.remove_features(x,[0,1,2,3])
    expected=program.disjoint_interaction(x,[0,2],[1,3])
    removal=float((interaction-expected).norm()/direct.norm())
    delta=torch.randn(9,24);y0,b,a=program.response_coefficients(x,delta)
    dose=max(float((program.forward_features(program.features(x)+t*delta)-(y0+t*b+t*t*a)).norm()/direct.norm()) for t in (-1.,.25,2.))
    result=dict(predictions=dict(pred_a_execution=execution<=1e-10,pred_b_interaction=removal<=1e-10,pred_c_dose=dose<=1e-10),
        execution_error=execution,interaction_error=removal,dose_error=dose,price=program.price(),
        scope='Rectangular synthetic feature execution; internal feature edits can leave the realizable input-feature subspace')
    Path(__file__).with_name('RECTANGULAR_SPARSE_READER_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert all(result['predictions'].values())
