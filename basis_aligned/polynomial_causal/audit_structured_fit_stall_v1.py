"""CPU red-team of the frozen weight-product checkpoint; no checkpoint mutation.

Registered instrument: CPU/GPU loss replay <=1e-8, alternate objective gradient
relative discrepancy <=1e-5. Descriptive probes: cancellation, directional finite
differences and a short fresh L-BFGS solve. A decrease >1e-8 with no penalty or
capacity change demonstrates remaining optimization headroom. Failure to find
such a decrease is inconclusive, never evidence against computational structure.
"""
import hashlib,json,time
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel,QuadraticObjective
from joint_quadratic_fit_v1 import product_cross

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    start=time.perf_counter();torch.set_num_threads(2)
    cp=P/'STRUCTURED_FIT_V1_weight_product_s0_CHECKPOINT.pt'
    receipt=json.loads((P/'STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00.json').read_text())
    digest=hashlib.sha256(cp.read_bytes()).hexdigest()
    assert digest==receipt['checkpoint_sha256']
    state=torch.load(cp,map_location='cpu',weights_only=False)
    model=QuadraticModel('product',1152);model.load_state_dict(state['model'])
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double();metric=u.T@u;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']]
    print('Computing independent native coefficient norm',flush=True)
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    objective=QuadraticObjective(metric,total,l=l,r=r,d=d)
    base,w,g=objective.loss(model);base_value=float(base.detach())
    base.backward();grad=torch.cat([p.grad.flatten() for p in model.parameters()]).detach().clone()
    kg=w.T@metric@w
    with torch.no_grad():
        diag=g.diag().clamp_min(1e-30);corr=g/(diag[:,None]*diag[None,:]).sqrt()
        corr.fill_diagonal_(0);ij=torch.argmax(corr.abs()).item();i,j=divmod(ij,len(g))
        writer_cos=float(kg[i,j]/(kg[i,i]*kg[j,j]).sqrt())
        cancellation=float((kg.diag()*g.diag()).sum()/(kg*g).sum())
    def centered_loss():
        cross,gram=objective.cross_gram(model)
        with torch.no_grad():writer=torch.linalg.solve(gram,cross.T).T
        return (((writer.T@metric@writer)*gram).sum()-2*((metric@cross)*writer).sum())/total
    def projected_loss():
        cross,gram=objective.cross_gram(model)
        writer=torch.linalg.solve(gram,cross.T).T
        return -((metric@cross)*writer).sum()/total
    model.zero_grad(set_to_none=True);projected=projected_loss();projected.backward()
    alternate_grad=torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
    grad_difference=float((alternate_grad-grad).norm()/grad.norm())
    initial={k:v.clone() for k,v in model.state_dict().items()}
    direction={};offset=0
    for name,p in model.named_parameters():
        direction[name]=-grad[offset:offset+p.numel()].reshape_as(p)/grad.norm();offset+=p.numel()
    probes=[]
    for step in [1e-5,1e-4,1e-3,1e-2,.1,1.]:
        values={}
        for sign in [-1,1]:
            model.load_state_dict({k:initial[k]+sign*step*direction[k] for k in initial})
            with torch.no_grad():values[sign]=float(objective.loss(model)[0])
        probes.append(dict(step=step,plus_loss=values[1],minus_loss=values[-1],
            observed_central_slope=(values[1]-values[-1])/(2*step),predicted_slope=-float(grad.norm()),
            decrease=base_value-values[1]))
    model.load_state_dict(initial)
    print('Running short independent fresh L-BFGS probes',flush=True)
    refinement=[]
    for variant in ['original','centered','projected']:
        model.load_state_dict(initial)
        opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=50,history_size=8,
            tolerance_grad=1e-12,tolerance_change=1e-16,line_search_fn='strong_wolfe')
        calls=0
        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True)
            loss=objective.loss(model)[0] if variant=='original' else centered_loss() if variant=='centered' else projected_loss()
            loss.backward();calls+=1;return loss
        opt.step(closure)
        diagnostics,_=objective.diagnostics(model)
        refinement.append(dict(variant=variant,closures=calls,**diagnostics,
            decrease=base_value-diagnostics['squared_relative_error']))
    saved_optimizer=next(iter(state['optimizer']['state'].values()))
    result=dict(schema='structured.fit.stall.audit.v1',checkpoint_sha256=digest,
        independent_native_total=float(total),cpu_replay_error=abs(base_value-receipt['last']['squared_relative_error']),
        projected_gradient_relative_error=grad_difference,gradient_norm=float(grad.norm()),
        sum_component_energy_over_joint_energy=cancellation,most_correlated_pair=[i,j],
        pair_input_cosine=float(corr[i,j]),pair_writer_cosine=writer_cos,
        gram_condition=float(torch.linalg.cond(g)),parameter_row_norms={k:dict(min=float(v.norm(dim=1).min()),max=float(v.norm(dim=1).max())) for k,v in initial.items()},
        saved_optimizer_direction_norm=float(saved_optimizer['d'].norm()),saved_optimizer_step=float(saved_optimizer['t']),
        directional_probes=probes,fresh_lbfgs_probes=refinement,
        predictions=dict(instrument=abs(base_value-receipt['last']['squared_relative_error'])<=1e-8 and grad_difference<=1e-5,
            remaining_optimization_headroom=max([p['decrease'] for p in probes]+[p['decrease'] for p in refinement])>1e-8),
        scope='Numerical optimization diagnostic only. No structural absence or circuit identification claim.',
        wall_seconds=time.perf_counter()-start)
    out=P/'STRUCTURED_FIT_STALL_V1_AUDIT.json'
    with out.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
