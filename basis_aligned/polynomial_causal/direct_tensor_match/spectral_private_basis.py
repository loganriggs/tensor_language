"""Choose private directions from the full-space rank relaxation, shared span fixed."""
import torch

def choose(targets,shared,width):
 basis=torch.linalg.qr(shared,mode='complete').Q;r=shared.shape[1];V=basis[:,r:];T=basis.T@targets@basis;B=T[:,:r,r:];C=T[:,r:,r:]
 M=torch.cat([2**.5*torch.cat(list(B),1),torch.cat(list(C),1)],0)
 eigenvalues,U=torch.linalg.eigh(M@M.T);bottom=U[r:,-width:];sv=torch.linalg.svdvals(bottom)
 if float(sv[-1]/sv[0])<=1e-10:raise ValueError('Rank-relaxation private projection is numerically deficient')
 directions=V@bottom
 return directions,dict(strategy='full_space_private_branch_spectral',width=width,private_bottom_condition=float(sv[0]/sv[-1]),rank_relaxation_root_bound=float((eigenvalues[:-width].clamp_min(0).sum()/targets.square().sum()).sqrt()),scope='Spectral initialization only; projected C blocks must still be symmetric and jointly fitted. Private directions may leave the earlier learned pair span.')
