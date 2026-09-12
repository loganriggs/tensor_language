"""One-producer-slot covariance bound for the fixed graded coefficient metric."""
import torch
from graded_source_projection_v1 import graded_norms

def covariance(forms,root,writer_gram):
    full=graded_norms(forms,root@root.T,writer_gram).detach()
    q=torch.eye(len(root),dtype=root.dtype,device=root.device,requires_grad=True)
    grades=graded_norms(forms,root@q@root.T,writer_gram)
    k=torch.arange(1,5,dtype=root.dtype,device=root.device)
    potential=(grades[1:]/(4*k*full[1:])).sum()
    s=torch.autograd.grad(potential,q)[0];return ((s+s.T)/2).detach()

def bound(forms,root,writer_gram,rank):
    s=covariance(forms,root,writer_gram);ev=torch.linalg.eigvalsh(s)
    return dict(loss_lower_bound=float(1-ev[-rank:].sum()),trace=float(ev.sum()),minimum_eigenvalue=float(ev[0]))
