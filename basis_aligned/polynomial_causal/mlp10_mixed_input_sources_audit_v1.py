"""Counter-review of residual-only miss using separately measured partner effects."""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent


def main():
    a=torch.load(P/'MLP10_MIXED_INPUT_SOURCES_V1_ARTIFACT.pt',weights_only=True)['measures'].double();cells=[]
    for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
        z=a[lo:hi,:,0];full=z[:,8]-z[:,5];rr=z[:,11]-z[:,5];mixed=z[:,12]-z[:,5];aa=z[:,13]-z[:,5];sumtwo=rr+mixed
        cells.append(dict(cell=label,residual_full_cosine=float((rr*full).sum()/(rr.norm()*full.norm())),
            residual_same_nonzero_sign=int((rr*full>0).sum()),residual_opposite_sign=int((rr*full<0).sum()),
            mixed_norm_over_full=float(mixed.norm()/full.norm()),aa_norm_over_full=float(aa.norm()/full.norm()),
            sum_two_measured_effects_error=float((sumtwo-full).norm()/full.norm()),sum_two_same_sign=int((sumtwo*full>0).sum()),
            maxabs_sum_two_error=float((sumtwo-full).abs().max()),maxabs_aa_effect=float(aa.abs().max())))
    result=dict(cells=cells,scope='RR+mixed is an arithmetic sum of separate measured effects, not a newly executed joint state intervention. Three-term physical-effect composition already measured. No filtering, fitting or repaired originalRR-only bar.')
    (P/'MLP10_MIXED_INPUT_SOURCES_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
