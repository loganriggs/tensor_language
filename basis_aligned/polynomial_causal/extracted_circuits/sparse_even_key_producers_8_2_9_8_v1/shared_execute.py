"""Known shared-key rewrite applied to portable parent; no fitted weights."""
import torch
import torch.nn.functional as F


class SharedParent:
    def __init__(self, program, share_read=True):
        self.p = program
        self.share_read = share_read
        self.adapters = [[program[k][i].double()@program['key_basis'][i]
                          for k in ['k1','k2']] for i in range(2)]

    def scalar(self, current, tokens, index):
        p = self.p; basis = p['key_basis'][index]
        n = current.shape[1]
        inv = 1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
        angle = torch.outer(torch.arange(n,dtype=torch.float32),inv)
        co = angle.cos().bfloat16().to(current.device)
        si = angle.sin().bfloat16().to(current.device)
        def rot(x):
            a,b = x.chunk(2,-1)
            return torch.cat((a*co+b*si,-a*si+b*co),-1)
        x = current.double(); shared = x@basis if self.share_read else None
        full = []; reflected = []
        for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
            q = rot(F.rms_norm(F.linear(current,p[qn][index]),(128,),eps=torch.finfo(torch.float32).eps)).double()
            native_key = F.linear(current,p[kn][index])
            den = (native_key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
            num = x@p[kn][index].double().T
            inside = (shared if self.share_read else x@basis)@self.adapters[index][j].T
            full.append(q@rot(num/den).transpose(-1,-2)/128)
            reflected.append(q@rot((num-2*inside)/den).transpose(-1,-2)/128)
        gamma = (full[0]*full[1]+reflected[0]*reflected[1])/2
        gamma = gamma.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=current.device).tril(),0)
        value = p['first_token_value'][tokens] if index==0 else x@p['current_value_reader']
        return (gamma@value[...,None])[...,0]


def load_shared(path, device='cpu'):
    """Load alongside the unchanged original execute.py and program.pt."""
    import importlib.util
    from pathlib import Path
    original = Path(__file__).with_name('execute.py')
    spec = importlib.util.spec_from_file_location('portable_parent_reference', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SharedParent(module.load_program(path, device))
