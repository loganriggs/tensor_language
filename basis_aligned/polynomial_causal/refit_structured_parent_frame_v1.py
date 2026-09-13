"""Five fixed-support two-of-four sparse complete-even fits, weights only.
200 L-BFGS iterations/300 evaluations per arm,180seconds overall.
Pred_a final loss <=initial each arm. Pred_b >=10%gain vs rotated baseline.
Pred_c normalized-column coordinate gradient RMS<=1e-8 for all five starts.
Pred_d FP32 stored payload normalized loss delta<=1e-6.
"""
import time,json
from pathlib import Path
import numpy as np
import torch
from sparse_complete_even_v1 import reflect,query_grams,inner,loss
from sparse_parent_reader_v1 import unpack
from shared_query_product_objective_v1 import rotation


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    p=Path(__file__).resolve().parent;start=time.perf_counter()
    n=torch.load(p/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    b=n['key_basis'][1].double();k1,k2=[n[k][1].double() for k in ('k1','k2')]
    l1,l2=[torch.stack([n[q][1].double().T@rotation(pos).T for pos in (1,4,16,63)]) for q in ('q1','q2')]
    grams=query_grams(l1,l2);o1,o2=reflect(k1,b),reflect(k2,b);norm=inner(grams,k1,k2,k1,k2)
    objective=lambda s:loss(s,grams,k1,k2,o1,o2,norm)/norm.sum()
    rotitem=torch.load(p/'STRUCTURED_PARENT_FRAME_V1_PROGRAM.pt',weights_only=True)['2of4']
    rs,_,_=unpack(rotitem);baseline=float(objective(rs));rows=[];best=None
    for frame,file in [('2of4','STRUCTURED_PARENT_FRAME_V1_PROGRAM.pt')]:
        item=torch.load(p/file,weights_only=True)[frame];ss,_,_=unpack(item)
        mask=torch.from_numpy(np.unpackbits(item['mask'].numpy(),bitorder='little').copy()).bool().reshape(ss.shape)
        for perturb in (0,.01,.03,.1,.3):
            arm=len(rows);gen=torch.Generator().manual_seed(7131347+arm)
            param=(ss+perturb*torch.randn(ss.shape,generator=gen)/1152**.5).requires_grad_(True)
            def matrix():
                s=param*mask
                return s/s.norm(dim=0,keepdim=True)
            initial=float(objective(matrix()).detach());evals=0
            opt=torch.optim.LBFGS([param],lr=1,max_iter=200,max_eval=300,
                tolerance_grad=1e-10,tolerance_change=1e-13,history_size=8,line_search_fn='strong_wolfe')
            def closure():
                nonlocal evals
                if time.perf_counter()-start>180:raise TimeoutError()
                opt.zero_grad();v=objective(matrix());v.backward();evals+=1;return v
            timedout=False
            try:opt.step(closure)
            except TimeoutError:timedout=True
            s=matrix();v=objective(s);grad=torch.autograd.grad(v,param)[0]
            # Evaluate gradient in unit column coordinates, independent of parameter scaling.
            unit=s.detach().requires_grad_(True)
            us=unit*mask;us=us/us.norm(dim=0,keepdim=True)
            ug=torch.autograd.grad(objective(us),unit)[0]
            grms=float(ug[mask].square().mean().sqrt())
            packed=dict(shape=list(s.shape),mask=item['mask'],values=s.detach()[mask].float())
            rounded,_,_=unpack(packed);delta=float(abs(objective(rounded)-v.detach()))
            record=dict(arm=arm,frame=frame,perturb=perturb,initial_loss=initial,final_loss=float(v.detach()),
                evaluations=evals,iterations=opt.state[param].get('n_iter',0),
                gradient_rms=grms,converged=grms<=1e-8,payload_loss_delta=delta,
                gram_condition=float(torch.linalg.cond(s.detach().T@s.detach())),timed_out=timedout)
            rows.append(record);print(json.dumps(record),flush=True)
            if best is None or record['final_loss']<best[0]:best=(record['final_loss'],packed,arm)
            if timedout:break
        if time.perf_counter()-start>180:break
    torch.save({'2of4':best[1]},p/'STRUCTURED_FRAME_REFIT_V1_PROGRAM.pt')
    result=dict(pred_a=all(r['final_loss']<=r['initial_loss']+1e-12 for r in rows),
        pred_b=best[0]<=.9*baseline,pred_c=len(rows)==5 and all(r['converged'] for r in rows),
        pred_d=all(r['payload_loss_delta']<=1e-6 for r in rows),
        baseline_loss=baseline,best_arm=best[2],relative_gain=1-best[0]/baseline,
        rows=rows,seconds=time.perf_counter()-start,scope=__doc__)
    (p/'STRUCTURED_FRAME_REFIT_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
