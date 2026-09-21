"""Dense identity checks and paired recovery tests for exact quartic CP."""
import json
import math
import time
from pathlib import Path
import torch
from quartic_cp import cp_entries, cp_gram
from quartic_cp_profile import cp_objective, normalize_factors
P = Path(__file__).resolve().parent


def main(rank=3, rate=.03, suffix="V1"):
    torch.set_num_threads(2)
    dtype = torch.float64
    indices = torch.cartesian_prod(*[torch.arange(4) for _ in range(4)])
    rows, controls = [], []
    for family_index, family in enumerate(['independent', 'shared_inputs', 'shared_outputs', 'squares', 'cancellation']):
        torch.manual_seed(9120 + family_index)
        teacher = normalize_factors([torch.randn(3, 4, dtype=dtype) for _ in range(4)])
        output = torch.randn(2, 3, dtype=dtype)
        if family == 'shared_inputs':
            teacher[0][1] = teacher[0][0]
        if family == 'shared_outputs':
            output = torch.randn(2, 1, dtype=dtype) @ torch.randn(1, 3, dtype=dtype)
        if family == 'squares':
            teacher = [teacher[0].clone() for _ in range(4)]
        if family == 'cancellation':
            for f in teacher:
                f[1] = f[0]
            output[:, 1] = -output[:, 0]
        target = cp_entries(output, teacher, indices)
        energy = target.square().sum()
        assert energy > 1e-6
        parameters = [torch.randn(3, 4, dtype=dtype, requires_grad=True) for _ in range(4)]
        factors = normalize_factors(parameters)
        phi = cp_entries(torch.eye(3, dtype=dtype), factors, indices)
        implicit_gram = cp_gram(factors, factors)
        gram_error = float(((implicit_gram-phi.T@phi).norm()/implicit_gram.norm()).detach())
        loss, coeff = cp_objective(output, teacher, factors)
        dense = (phi@coeff.T-target).square().sum()-energy+1e-8*coeff.square().sum()
        grads = torch.autograd.grad(loss, parameters, retain_graph=True)
        dense_grads = torch.autograd.grad(dense, parameters, retain_graph=True)
        full_loss, _ = cp_objective(output, teacher, factors, envelope=False)
        full_grads = torch.autograd.grad(full_loss, parameters)
        def grad_error(other):
            return float(torch.stack([(a-b).square().sum() for a,b in zip(grads,other)]).sum().sqrt()/torch.stack([a.square().sum() for a in grads]).sum().sqrt().clamp_min(1e-15))
        _, planted = cp_objective(output, teacher, teacher)
        planted_error = float((cp_entries(planted, teacher, indices)-target).norm()/target.norm())
        control = dict(family=family,gram_error=gram_error,loss_error=float((loss-dense).abs().detach()/energy),dense_gradient_error=grad_error(dense_grads),full_solve_gradient_error=grad_error(full_grads),planted_error=planted_error)
        controls.append(control)
        assert max(control[k] for k in ['gram_error','loss_error','dense_gradient_error','full_solve_gradient_error']) < 1e-8, control
        assert planted_error < 1e-4, control
        for optimizer in ['adam','muon']:
            for restart in [0,1]:
                torch.manual_seed(9340+restart)
                parameters = [torch.nn.Parameter(torch.randn(rank,4,dtype=dtype)/2) for _ in range(4)]
                opt = torch.optim.Adam(parameters,lr=rate) if optimizer=='adam' else torch.optim.Muon(parameters,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
                start=time.monotonic();best=None;initial=None
                for step in range(401):
                    factors=normalize_factors(parameters)
                    loss,coeff=cp_objective(output,teacher,factors)
                    score=float(loss.detach())
                    if best is None or score<best[0]:
                        with torch.no_grad():error=float((cp_entries(coeff,factors,indices)-target).norm()/target.norm())
                        best=(score,step,error)
                    if step==0:initial=best[2]
                    if step==400:break
                    opt.zero_grad();(loss/energy).backward();opt.step()
                    for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/400)))
                row=dict(rank=rank,rate=rate,family=family,optimizer=optimizer,restart=restart,initial_error=initial,error=best[2],selected_step=best[1],seconds=time.monotonic()-start)
                rows.append(row);print(json.dumps(row),flush=True)
    summary={}
    for optimizer in ['adam','muon']:
        subset=[r for r in rows if r['optimizer']==optimizer]
        summary[optimizer]=dict(below1percent=sum(r['error']<.01 for r in subset),median_error=float(torch.tensor([r['error'] for r in subset]).median()),median_initial=float(torch.tensor([r['initial_error'] for r in subset]).median()),seconds=sum(r['seconds'] for r in subset))
    result=dict(controls=controls,rows=rows,summary=summary,predictions=dict(a=True,b=any(v['below1percent']>=8 for v in summary.values()),c=all(v['median_error']<v['median_initial'] for v in summary.values())),scope='CPU five-family CP rank3 exact coefficient recovery; 400 scheduled steps, two starts, one rate; no native fit or general optimizer superiority.')
    (P/f'QUARTIC_CP_PROFILE_CONTROLS_{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    for rank,rate,suffix in [(3,.1,'R3_LR01'),(6,.03,'R6_LR003'),(6,.1,'R6_LR01')]:
        main(rank,rate,suffix)
