"""Weights-only fixed-partner discriminator for the broad input-mode baseline."""
from pathlib import Path
import torch,json
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    s=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    q=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    J=q['mixed_map'].double();w=q['direction'].double()
    L,R,D=[s['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    maps=[(D*((R@y)[None,:]))@(L@J)+(D*((L@y)[None,:]))@(R@J) for y in (w,J@w)]
    joint=torch.cat([h/h.norm() for h in maps]);result={}
    for name,h in zip(('partner_w','partner_Jw_w','joint_unit_normalized'),maps+[joint]):
        values=torch.linalg.svdvals(h);energy=values.square();total=energy.sum()
        result[name]=dict(relative_errors={str(k):float((energy[k:].sum()/total).sqrt())
                          for k in (4,8,16,32,64,128)},
                          ranks_for_error={str(t):next((k for k in range(len(values)+1)
                          if energy[k:].sum()<=t*t*total),len(values)) for t in (.02,.05,.1)},
                          frobenius_norm=float(h.norm()))
    result['scope']=('Actual fixed directions in response basis, weight-only matrix spectra. '
                     'Joint bank uses equal unit-Frobenius function weighting, not amplitude or text weighting. '
                     'Only K(Jz,w) and K(Jz,Jw*w) paths, not remaining variable-partner terms. '
                     'Spectral rank bounds concern linear matrix representations; no behavioral '
                     'or whole-program storage claim.')
    (P/'INTERACTION_FIXED_PARTNER_SPECTRA_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
