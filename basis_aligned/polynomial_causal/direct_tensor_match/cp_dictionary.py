"""Greedy exact output-variable-projection selection from a fixed CP dictionary."""
import torch

def select_atoms(K,Q,width):
 K=K.double();Q=Q.double();selected=[];receipts=[]
 for step in range(width):
  if selected:
   block=K[selected][:,selected];cross=K[selected];projection=torch.linalg.solve(block,cross);residual=Q-Q[:,selected]@projection;schur=K.diag()-(cross*projection).sum(0)
  else:residual=Q;schur=K.diag().clone()
  valid=schur>1e-10*K.diag().max();scores=residual.square().sum(0)/schur.clamp_min(1e-30);scores[~valid]=-float('inf');scores[selected]=-float('inf');j=int(scores.argmax())
  if not bool(torch.isfinite(scores[j])):break
  selected.append(j);block=K[selected][:,selected];writer=torch.linalg.solve(block,Q[:,selected].T).T;gain=float(2*(Q[:,selected]*writer).sum()-((writer@block)*writer).sum());receipts.append(dict(width=len(selected),selected=selected.copy(),gain=gain,increment=float(scores[j]),condition=float(torch.linalg.cond(block))))
 return receipts
