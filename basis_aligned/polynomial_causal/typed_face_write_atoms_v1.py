"""Three physical writes retaining the complete routing/inherited product."""
import importlib.util
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('face_native',P/'extracted_circuits/odd_attention8h2_typed_face_v1/native.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)

def channels(p,current,donor_city,recipient_token,donor_token,city):
    a0=native.routing(p,current,current[:,city],city).double()
    a1=native.routing(p,current,donor_city,city).double()
    c=torch.nn.functional.linear(current,p['current_value'].to(current.dtype))[:,city]
    i0=native.inherited(p,recipient_token).to(current.dtype)
    i1=native.inherited(p,donor_token).to(current.dtype)
    v0=((1-p['mixture'])*c+p['mixture']*i0).double()
    v1=((1-p['mixture'])*c+p['mixture']*i1).double()
    da=a1-a0;dv=v1-v0
    return torch.stack([da[...,None]*v0[:,None],a0[...,None]*dv[:,None],da[...,None]*dv[:,None]])

def random_basis(seed):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    matrix=torch.randn(128,128,generator=generator,dtype=torch.float64)
    q,r=torch.linalg.qr(matrix)
    signs=torch.diag(r).sign();signs[signs==0]=1
    return q*signs[None]

def split(total,basis):
    return torch.stack([(total@basis[:,a:b])@basis[:,a:b].T for a,b in [(0,43),(43,86),(86,128)]])

def physical_writes(channel_pieces,p,mask):
    return (channel_pieces@p['output'].double().T)*mask[None,None,:,None]
