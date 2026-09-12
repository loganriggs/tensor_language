"""Exact finite intervention algebra on frozen four-sector masks, no fitting."""
import json,torch
from pathlib import Path
P=Path(__file__).resolve().parent
def transform(values):
    result=values.clone()
    for bit in range(4):
        for mask in range(16):
            if mask&(1<<bit):result[mask]-=result[mask^(1<<bit)]
    return result
def reconstruct(terms):
    result=terms.clone()
    for bit in range(4):
        for mask in range(16):
            if mask&(1<<bit):result[mask]+=result[mask^(1<<bit)]
    return result
def main():
    p=P/'SCALAR_SECTOR_MOBIUS_V1_RESULT.json';assert not p.exists()
    a=torch.load(P/'SCALAR_VALUE_SECTOR_FACTORIAL_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    m=a['regional'][:,:,0];contrasts=m[::2]-m[1::2];regional=(contrasts[:,:1]-contrasts).T
    newline=(a['newline_ce'][:,:16]-a['newline_ce'][:,:1]).T
    rm,nm=transform(regional),transform(newline)
    re=float((reconstruct(rm)-regional).norm()/regional.norm());ne=float((reconstruct(nm)-newline).norm()/newline.norm())
    records=[]
    for family in range(2):
        ix=slice(family*12,(family+1)*12);nx=slice(family*16,(family+1)*16)
        records.append(dict(family=family,regional_mask6_mean=float(regional[6,ix].mean()),regional_single_means=[float(rm[k,ix].mean()) for k in (2,4)],regional_pair_interaction_mean=float(rm[6,ix].mean()),regional_pair_interaction_to_full_norm=float(rm[6,ix].norm()/regional[6,ix].norm()),newline_single_meanabs=[float(nm[k,nx].abs().mean()) for k in (2,4)],newline_pair_interaction_meanabs=float(nm[6,nx].abs().mean()),newline_pair_interaction_to_full_norm=float(nm[6,nx].norm()/newline[6,nx].norm())))
    result={'pred_a':max(re,ne)<=1e-10,'regional_reconstruction_error':re,'newline_reconstruction_error':ne,'records':records,'preserved_outlier':dict(index=30,single_8first=float(nm[2,30]),single_9current=float(nm[4,30]),pair_interaction=float(nm[6,30]),joint=float(newline[6,30]),fullmask=float(newline[15,30])),'scope':'Exact Boolean finite-intervention expansion, basis fixed by masks. Descriptive interaction magnitudes, not allocation of independent causal ownership or rescue of failed preservation.'}
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
