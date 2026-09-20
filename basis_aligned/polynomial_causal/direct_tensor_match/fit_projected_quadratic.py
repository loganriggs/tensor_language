"""Fit normalized product features to an implicit projected quadratic operator."""
import math
import torch
from quadratic_student_fit import gram

def objective(operator,a,b,penalty=.001):
    # Normalize in the declared covariance metric, retaining original coordinates.
    aw,bw=a@operator.L,b@operator.L
    scale=(.5*(aw.square().sum(1)*bw.square().sum(1)+(aw*bw).sum(1).square())).clamp_min(1e-24).pow(.25)
    u,v=a/scale[:,None],b/scale[:,None]
    G=gram(u@operator.L,v@operator.L).double()
    X=operator.cross(u,v).double()
    writer=torch.linalg.solve(G+penalty*torch.eye(len(a),device=a.device,dtype=G.dtype),X.T).T
    loss=((writer@G)*writer).sum()-2*(writer*X).sum()+penalty*writer.square().sum()
    return loss,(u,v,writer)

def fit(operator,a0,b0,optimizer='adam',lr=.005,steps=100,penalty=.001):
    a=torch.nn.Parameter(a0.clone());b=torch.nn.Parameter(b0.clone())
    opt=torch.optim.Adam([a,b],lr=lr) if optimizer=='adam' else torch.optim.Muon([a,b],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
    best=float('inf');saved=None;history=[]
    for step in range(steps+1):
        opt.zero_grad();loss,program=objective(operator,a,b,penalty)
        value=float(loss.detach());assert math.isfinite(value)
        if step==0:initial=value;normalizer=max(abs(value),1e-20)
        if value<best:best=value;saved=tuple(t.detach().clone() for t in program);selected=step
        if step%20==0 or step==steps:history.append(dict(step=step,objective=value))
        if step==steps:break
        (loss/normalizer).backward();opt.step()
        for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
    return dict(initial_objective=initial,best_objective=best,selected_step=selected,history=history),saved

def check():
    import json
    from pathlib import Path
    from gaussian_projected_quadratic_cross import ProjectedQuadratic
    torch.set_num_threads(1);torch.manual_seed(2023)
    teacher=[torch.randn(*s,dtype=torch.float64) for s in [(2,3),(3,5),(3,5),(5,6),(6,4),(6,4)]]
    op=ProjectedQuadratic(teacher,torch.randn(4,dtype=torch.float64),torch.eye(4,dtype=torch.float64))
    a=torch.randn(2,4,dtype=torch.float64,requires_grad=True);b=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    loss,(u,v,c)=objective(op,a,b);G=gram(u@op.L,v@op.L);X=op.cross(u,v)
    normal=float((c@(G+.001*torch.eye(2,dtype=G.dtype))-X).norm()/X.norm())
    grads=torch.autograd.grad(loss,[a,b]);eps=1e-5;ap=a.detach().clone();am=ap.clone();ap[0,0]+=eps;am[0,0]-=eps
    fd=(objective(op,ap,b)[0]-objective(op,am,b)[0])/(2*eps);error=float(((fd-grads[0][0,0]).abs()/grads[0][0,0].abs()).detach())
    result,_=fit(op,a.detach(),b.detach(),steps=100)
    assert normal<1e-12 and error<1e-5 and result['best_objective']<result['initial_objective']
    receipt=dict(normal_equation_error=normal,finite_difference_relative_error=error,fit=result,scope='CPU toy objective/gradient and optimization check. No native improvement claim.')
    (Path(__file__).resolve().parent/'PROJECTED_QUADRATIC_FIT_CHECK_V1.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
if __name__=='__main__':check()
