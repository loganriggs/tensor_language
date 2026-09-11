"""Output-sharing symmetric LL1 conditional solves; dense small controls only."""
import json
from pathlib import Path
import torch


def input_update(residual, writer, rank):
    matrix=torch.einsum('o,oij->ij',writer,residual)/writer.square().sum()
    values,vectors=torch.linalg.eigh((matrix+matrix.T)/2)
    ids=values.abs().argsort(descending=True)[:rank]
    return vectors[:,ids],values[ids]


def output_update(residual, matrix):
    return torch.einsum('oij,ij->o',residual,matrix)/matrix.square().sum()


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1351)
    residual=torch.randn(7,9,9);residual=(residual+residual.transpose(1,2))/2
    c=torch.randn(7);a,h=input_update(residual,c,3);q=(a*h)@a.T
    prediction=c[:,None,None]*q
    gain=residual.square().sum()-(residual-prediction).square().sum()
    identity=float(abs(gain-c.square().sum()*h.square().sum())/residual.square().sum())
    x=torch.randn(13,9)
    computed=((x@a).square()@h)[:,None]*c
    direct=torch.einsum('ni,oij,nj->no',x,prediction,x)
    execution=float((computed-direct).norm()/direct.norm())
    planted=[];groups=[]
    for j in range(2):
        frame=torch.linalg.qr(torch.randn(9,2)).Q
        matrix=(frame*torch.tensor([1.,-.6]))@frame.T
        writer=torch.randn(7);planted.append((writer,matrix))
        groups.append((writer+.03*torch.randn(7),matrix.clone()))
    truth=sum(w[:,None,None]*m for w,m in planted);total=truth.square().sum()
    previous=float('inf');increase=0.
    for step in range(200):
        for j in range(2):
            target=truth-sum(w[:,None,None]*m for k,(w,m) in enumerate(groups) if k!=j)
            frame,values=input_update(target,groups[j][0],2);matrix=(frame*values)@frame.T
            writer=output_update(target,matrix);groups[j]=(writer,matrix)
            error=float((truth-sum(w[:,None,None]*m for w,m in groups)).square().sum()/total)
            increase=max(increase,error-previous);previous=error
        if error<1e-12:break
    result=dict(predictions=dict(pred_a_conditional=identity<=1e-10,pred_b_executor=execution<=1e-10,
                                 pred_c_near_two_block_recovery=error<=1e-8),
                capture_identity_error=identity,executor_error=execution,planted_error=error,
                sweeps=step+1,maximum_error_increase=increase,
                scope='Symmetric indefinite output-sharing LL1, dense small conditional/recovery controls. No native full fit, generic global guarantee, or identified circuit.')
    Path(__file__).with_name('SYMMETRIC_LL1_CONDITIONAL_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert identity<=1e-10 and execution<=1e-10


if __name__=='__main__':control()
