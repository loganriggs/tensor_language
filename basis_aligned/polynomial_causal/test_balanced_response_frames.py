import torch
from balanced_response_frames import balanced_frames


def test_dual_frames_preserve_optimal_cross_snapshot_truncation():
    torch.manual_seed(653)
    x=torch.randn(17,11,dtype=torch.float64)
    y=torch.randn(19,11,dtype=torch.float64)
    V,W,s=balanced_frames(x,y,4)
    torch.testing.assert_close(W.T@V,torch.eye(4,dtype=torch.float64),atol=1e-12,rtol=1e-12)
    u,sv,vh=torch.linalg.svd(y@x.T,full_matrices=False)
    expected=(u[:,:4]*sv[:4])@vh[:4]
    actual=(y@V)@(W.T@x.T)
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)


def test_small_observable_direction_beats_large_unobserved_state():
    x=torch.diag(torch.tensor([100.,1.],dtype=torch.float64))
    y=torch.diag(torch.tensor([.001,1.],dtype=torch.float64))
    V,W,_=balanced_frames(x,y,1)
    probe=torch.tensor([2.,3.],dtype=torch.float64)
    recovered=V@(W.T@probe)
    torch.testing.assert_close(recovered[1],probe[1])
    _,_,vh=torch.linalg.svd(x)
    pod=vh[:1].T
    assert float(((pod@pod.T)@probe)[1])==0.
