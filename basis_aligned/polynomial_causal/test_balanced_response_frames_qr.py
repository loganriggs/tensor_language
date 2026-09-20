import torch
from balanced_response_frames import balanced_frames,balanced_frames_qr


def test_qr_reduction_preserves_cross_snapshot_operator():
    torch.manual_seed(677)
    X=torch.randn(37,11,dtype=torch.float64);Y=torch.randn(53,11,dtype=torch.float64)
    V,W,s=balanced_frames(X,Y,4);v,w,sv=balanced_frames_qr(X,Y,4)
    torch.testing.assert_close(s[:11],sv,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(w.T@v,torch.eye(4,dtype=torch.float64),atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(V@W.T,v@w.T,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close((Y@V)@(W.T@X.T),(Y@v)@(w.T@X.T),atol=1e-10,rtol=1e-10)
