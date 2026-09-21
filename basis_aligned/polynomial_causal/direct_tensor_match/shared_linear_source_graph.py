"""Shared linear intermediate nodes feeding mixed products and private squares.
The factorized reader is executed as two linear stages; dense readers are audit-only.
"""
import torch

def source_reads(z,p):
 s=z@p['input_basis']
 q=((s@p['left_map'])*(s@p['right_map']))@p['product_weights']
 q[...,int(p['square_output'])]+=(s@p['square_map']).square()@p['square_weights']
 return q+z@p['source_linear']+p['source_bias']

def component_scalars(z,h,p):
 q=source_reads(z,p);scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[...,None]
 return ((h@p['h_readers']-.5*q[...,::2])/scale-p['alpha'])*(q[...,1::2]/scale-p['beta'])

def expand(p):
 out={k:v for k,v in p.items() if k not in ('input_basis','left_map','right_map','square_map')}
 for family in ('left','right','square'):out[family+'_reader']=p['input_basis']@p[family+'_map']
 return out

def factor(p,basis):
 out={k:v for k,v in p.items() if k not in ('left_reader','right_reader','square_reader')}
 out['input_basis']=basis
 for family in ('left','right','square'):out[family+'_map']=basis.T@p[family+'_reader']
 return out

def arithmetic(p):
 # Common affine corrections, later state and final model excluded equally.
 if 'input_basis' in p:
  matrices=[p[k] for k in ('input_basis','left_map','right_map','square_map')]
 else:matrices=[p[k] for k in ('left_reader','right_reader','square_reader')]
 m=p['product_weights'].shape[0];v=p['square_weights'].numel()
 projection=sum(a.numel() for a in matrices)
 readout=p['product_weights'].numel()+v
 adds=sum(a.shape[1]*(a.shape[0]-1) for a in matrices)+6*(m-1)+v
 return dict(stored_floats=sum(a.numel() for a in p.values() if a.is_floating_point()),activation_products=m+v,projection_multiplications=projection,source_total_multiplications=projection+readout+m+v,source_additions=adds)
