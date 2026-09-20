import torch
from running_mean_head import execute


def test_running_mean_matches_independent_dense_pattern_and_chunked_state():
    torch.manual_seed(636)
    for length in [1,2,17]:
        v=torch.randn(3,length,7,dtype=torch.float64)
        pos=torch.arange(length)
        for diagonal in [0.,.7,torch.randn(3,length,dtype=torch.float64)]:
            for mass in [-1.,0.,1.]:
                pattern=-mass*torch.ones(length,length,dtype=v.dtype).tril(-1)/pos.clamp_min(1)[:,None]
                want=torch.einsum('ts,bsd->btd',pattern,v)
                want=want+(diagonal[...,None] if torch.is_tensor(diagonal) else diagonal)*v
                torch.testing.assert_close(execute(v,diagonal,mass),want)
        total=torch.zeros_like(v[:,0]);steps=[]
        for i in range(length):
            steps.append(.7*v[:,i]-total/max(i,1));total=total+v[:,i]
        torch.testing.assert_close(execute(v,.7),torch.stack(steps,1))


def test_value_projection_fold_commutes_with_shared_mean():
    from running_mean_head import execute_projected
    torch.manual_seed(637)
    a=torch.randn(2,13,5,dtype=torch.float64);b=torch.randn_like(a)
    v=torch.randn(3,5,dtype=torch.float64);v0=torch.randn_like(v)
    o=torch.randn(5,3,dtype=torch.float64);mix=.37
    got=execute_projected(a,b,v,v0,o,mix,diagonal=.01)
    # Independent fold: average residual sources first, then apply composed maps.
    want=(1-mix)*(execute(a,.01)@(o@v).T)+mix*(execute(b,.01)@(o@v0).T)
    torch.testing.assert_close(got,want)
