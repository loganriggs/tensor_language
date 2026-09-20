"""Evaluate fixed native quartic and archived quadratic/quartic programs."""
import torch

def native(teacher,x):
 C,l,r,D,A,B=teacher;z=((x@A.T)*(x@B.T))@D.T
 return ((z@l.T)*(z@r.T))@C.T

def quadratic(s,x):
 d=x-s['mu']
 return s['constant']+(d@s['linear_reader'].T)@s['linear_writer'].T+((d@s['quadratic_left'].T)*(d@s['quadratic_right'].T))@s['quadratic_writer'].T

def quartic(s,x):
 if 'output_writer' in s:
  primitive=(x@s['A'].T)*(x@s['B'].T);bank=primitive@s['bank_writer'].T if 'bank_writer' in s else primitive
  result=((bank@s['root_left'].T)*(bank@s['root_right'].T))@s['output_writer'].T+s['constant']
  if 'skip_writer' in s:
   skip=primitive@s['skip_reader'].T if 'skip_reader' in s else primitive
   result=result+skip@s['skip_writer'].T
  return result
 if 'bank_writer' in s:bank=((x@s['A'].T)*(x@s['B'].T))@s['bank_writer'].T
 else:bank=((x@s['U'].flatten(0,1).T)*(x@s['V'].flatten(0,1).T)).reshape(len(x),4,4).sum(2)
 i,j=torch.triu_indices(4,4,device=x.device)
 return ((bank[:,i]*bank[:,j])@s['Z'].T)@s['W'].T+s['constant']

def load_teacher(path,scale,device='cuda'):
 state=torch.load(path,weights_only=True)
 def w(k):return state[k].to(device=device,dtype=torch.float32)
 _,ru=torch.linalg.qr(w('lm_head.weight'))
 return [ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
