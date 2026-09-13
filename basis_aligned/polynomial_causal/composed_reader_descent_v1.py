"""Three-step CPU escape screen; not a converged fit or adopted circuit.

Fixed 4492 product budget, native norm-ranked initialization, free readers.
Pred_a >=1% decrease in squared coefficient error; pred_b final error <=10%.
No text fitting. Backtracking accepts only objective decrease.
"""
import json,time
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from fixed_writer_products_v1 import prepare_global
from symmetric_product_varpro_v1 import gram,normalized,solve

def main():
    torch.set_num_threads(2);start=time.perf_counter();p=Path(__file__).resolve().parent
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    def maps(layer):return [sd[f'transformer.h.{layer}.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    tl,tr,d=maps(9);l10,r10,d10=maps(10)
    w=torch.load(p/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].double()*float(sd['transformer.h.10.lambdas'][0])
    tw=(prepare_global(w,l10,r10,d10,True)['mixed_map']@d).T
    h=gram(tl,tr,tl,tr);energy=(tw*(h@tw)).sum()
    order=(tw.square().sum(1)*h.diag()).argsort(descending=True)[:4492]
    del h
    l=normalized(tl[order]).detach().requires_grad_();r=normalized(tr[order]).detach().requires_grad_()
    history=[];initial=None
    for iteration in range(4):
        loss,_,_,_=solve(l,r,tl,tr,tw,energy)
        gl,gr=torch.autograd.grad(loss,(l,r));value=float(loss.detach())
        if initial is None:initial=value
        rms=float(((gl.square().sum(1)+gr.square().sum(1)).mean()/2).sqrt())
        history.append(dict(iteration=iteration,relative_squared_error=value,reader_gradient_rms=rms,elapsed_seconds=time.perf_counter()-start))
        if iteration==3:break
        step=.01/max(rms,1e-30);accepted=False
        for backtrack in range(10):
            with torch.no_grad():
                nl=normalized(l-step*gl);nr=normalized(r-step*gr)
                trial=solve(nl,nr,tl,tr,tw,energy)[0]
            if float(trial)<value:
                accepted=True;break
            step*=.5
        history[-1].update(step=step,backtracks=backtrack,accepted=accepted)
        if not accepted:break
        l=nl.detach().requires_grad_();r=nr.detach().requires_grad_()
    final=history[-1]['relative_squared_error']
    result=dict(pred_a=(initial-final)/initial>=.01,pred_b=final<=.01,history=history,
        squared_error_improvement=(initial-final)/initial,final_relative_error=final**.5,
        seconds=time.perf_counter()-start,scope=__doc__)
    (p/'COMPOSED_READER_DESCENT_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
