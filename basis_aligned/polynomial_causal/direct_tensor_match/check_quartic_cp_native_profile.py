"""Independent dense expansion for the native folded quartic profile."""
import itertools
import json
from pathlib import Path
import torch
from quartic_cp import cp_entries, directional
from quartic_cp_profile import normalize_factors, native_objective


def controls():
    torch.set_num_threads(2)
    rows=[]
    for seed in range(5):
        torch.manual_seed(9880+seed)
        d=3;dtype=torch.float64
        c=torch.randn(2,4,dtype=dtype);a=torch.randn(4,3,dtype=dtype);b=torch.randn_like(a)
        down=torch.randn(3,4,dtype=dtype);left=torch.randn(4,d,dtype=dtype);right=torch.randn_like(left)
        if seed==1:left[1]=left[0]
        if seed==2:c[1]=c[0]
        if seed==3:right=left.clone()
        if seed==4:left[1]=left[0];right[1]=-right[0]
        teacher=[c,a,b,down,left,right]
        s=.5*torch.einsum('ak,ki,kj->aij',down,left,right)
        s=s+s.transpose(1,2)
        t=.5*(torch.einsum('vk,ka,kb->vab',c,a,b)+torch.einsum('vk,kb,ka->vab',c,a,b))
        raw=torch.einsum('vab,aij,bkl->vijkl',t,s,s)
        dense=sum(raw.permute(0,*[j+1 for j in perm]) for perm in itertools.permutations(range(4)))/24
        target=dense.flatten(1).T
        idx=torch.cartesian_prod(*[torch.arange(d) for _ in range(4)])
        eye=torch.eye(d,dtype=dtype)
        analytic=directional(*teacher,[eye[idx[:,j]] for j in range(4)])
        p=[torch.randn(5,d,dtype=dtype,requires_grad=True) for _ in range(4)]
        f=normalize_factors(p);loss,coeff=native_objective(teacher,f,ridge=1e-6)
        pred=cp_entries(coeff,f,idx)
        explicit=(pred-target).square().sum()-target.square().sum()+1e-6*coeff.square().sum()
        grad=torch.autograd.grad(loss,p,retain_graph=True);g2=torch.autograd.grad(explicit,p)
        error=lambda x,y:float(((x-y).norm()/y.norm().clamp_min(1e-15)).detach())
        row=dict(seed=seed,directional=error(analytic,target),loss=float((loss-explicit).abs().detach()/target.square().sum()),gradient=error(torch.cat([g.flatten() for g in grad]),torch.cat([g.flatten() for g in g2])))
        assert max(row[k] for k in ['directional','loss','gradient'])<1e-8,row
        rows.append(row)
    return rows
if __name__=='__main__':
    rows=controls()
    Path(__file__).with_name('QUARTIC_CP_NATIVE_PROFILE_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows))
