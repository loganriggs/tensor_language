"""Weight-derived probes of a quartic coefficient tensor's first input unfolding."""
import torch
from quartic_cp import directional

def input_probes(teacher,b,c,d,output_probe):
 with torch.enable_grad():
  a=torch.zeros_like(b,requires_grad=True);value=directional(*teacher,[a,b,c,d]);return torch.autograd.grad((value*output_probe).sum(),a)[0].detach()
