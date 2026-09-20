import torch
import torch.nn.functional as F
from normalized_bilinear_face import decompose


def test_face_matches_direct_normalized_map_and_keeps_denominator_correction():
    torch.manual_seed(638)
    L=torch.randn(9,5,dtype=torch.float64);R=torch.randn_like(L);D=torch.randn(4,9,dtype=torch.float64)
    b,m,e=[torch.randn(3,7,5,dtype=torch.float64) for _ in range(3)]
    b[0,0]=0.;eps=1e-4
    out=decompose(L,R,D,b,m,e,eps)
    bias=torch.randn(4,dtype=torch.float64)
    def native(x):
        n=F.rms_norm(x,(5,),eps=eps)
        return ((n@L.T)*(n@R.T))@D.T+bias
    direct=native(b+m+e)-native(b+m)-native(b+e)+native(b)
    torch.testing.assert_close(out['interaction'],direct)
    torch.testing.assert_close(out['cross']+out['normalization'],direct)
    assert (out['cross']-direct).norm()/direct.norm()>.1
    none=decompose(L,R,D,b,m,torch.zeros_like(e),eps)
    torch.testing.assert_close(none['interaction'],torch.zeros_like(direct))
    torch.testing.assert_close(none['cross']+none['normalization'],torch.zeros_like(direct))


def test_head_output_projection_folds_into_both_cross_readers():
    torch.manual_seed(639)
    L=torch.randn(11,7,dtype=torch.float64);R=torch.randn_like(L)
    D=torch.randn(7,11,dtype=torch.float64);O=torch.randn(7,3,dtype=torch.float64)
    m=torch.randn(2,5,3,dtype=torch.float64);e=torch.randn_like(m)
    a,b=m@O.T,e@O.T
    direct=((a@L.T)*(b@R.T)+(b@L.T)*(a@R.T))@D.T
    A,B=L@O,R@O
    folded=((m@A.T)*(e@B.T)+(e@A.T)*(m@B.T))@D.T
    torch.testing.assert_close(direct,folded)
    # The denominator geometry also folds exactly, but background is still needed.
    background=torch.randn_like(a);G=O.T@O;z=m+e
    squared=background.square().sum(-1)+2*((background@O)*z).sum(-1)+torch.einsum('bti,ij,btj->bt',z,G,z)
    torch.testing.assert_close(squared,(background+a+b).square().sum(-1))


def test_conditional_head_coordinates_reuse_background_and_norm_geometry():
    from normalized_bilinear_face import prepare_head_coordinates,evaluate_head_coordinates
    torch.manual_seed(640)
    L=torch.randn(9,7,dtype=torch.float64);R=torch.randn_like(L)
    D=torch.randn(7,9,dtype=torch.float64);O=torch.randn(7,3,dtype=torch.float64)
    b=torch.randn(2,5,7,dtype=torch.float64);m=torch.randn(2,5,3,dtype=torch.float64);e=torch.randn_like(m)
    bias=torch.randn(7,dtype=torch.float64);eps=1e-5
    program=prepare_head_coordinates(L,R,D,O,b,eps,bias)
    for z in [torch.zeros_like(m),m,e,m+e]:
        x=b+z@O.T;n=F.rms_norm(x,(7,),eps=eps)
        target=((n@L.T)*(n@R.T))@D.T+bias
        torch.testing.assert_close(evaluate_head_coordinates(program,z),target)
