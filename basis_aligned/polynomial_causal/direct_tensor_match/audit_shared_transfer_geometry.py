"""Diagnostic calibration support, spectral truncation and evaluation oracle gap.
No evaluation-based model selection and no fitted coefficients exported.
"""
import json,time
from pathlib import Path
import torch
from sparse_quartic_bank import features
from audit_fixed_cp_response_capacity import fit
P=Path(__file__).resolve().parent

def analyze(cal,eva,ycal,yeva,wcal,weva):
    scale=cal.square().mean(0).sqrt().clamp_min(1e-30)
    a=cal/scale*(wcal/wcal.mean()).sqrt()[:,None]
    b=ycal*(wcal/wcal.mean()).sqrt()
    u,s,vh=torch.linalg.svd(a,full_matrices=False)
    z=eva/scale
    coordinates=z@vh.T
    leverage=(coordinates/s).square().sum(1)*len(cal)
    ce,info=fit(eva,yeva,weva);oracle=eva@ce
    target_energy=(weva*yeva.square()).sum()
    oracle_energy=(weva*(oracle-yeva).square()).sum()
    rows=[]
    for cutoff in [0.,.001,.003,.01,.03]:
        keep=s>s[0]*max(cutoff,1e-12)
        pred=coordinates[:,keep]@((u[:,keep].T@b)/s[keep])
        residual=pred-yeva;energy=weva*residual.square()
        excess=(weva*(pred-oracle).square()).sum()
        identity=float(abs(energy.sum()-oracle_energy-excess)/target_energy)
        assert identity<1e-9
        n=max(1,len(eva)//10);idx=torch.topk(leverage,n).indices
        rows.append(dict(cutoff=cutoff,rank=int(keep.sum()),sensitivity_error=float((energy.sum()/target_energy).sqrt()),oracle_error=float((oracle_energy/target_energy).sqrt()),excess_error=float((excess/target_energy).sqrt()),pythagoras_error=identity,high_leverage_error_fraction=float(energy[idx].sum()/energy.sum()),high_leverage_target_energy_fraction=float((weva[idx]*yeva[idx].square()).sum()/target_energy)))
    return dict(rows=rows,leverage_quantiles=torch.quantile(leverage,torch.tensor([.5,.9,.99,1.],dtype=leverage.dtype)).tolist(),singular_ratio=float(s[-1]/s[0]),oracle_solver=info)

def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
    torch.manual_seed(221)
    a=torch.randn(90,7,dtype=torch.float64);e=torch.randn(40,7,dtype=torch.float64)*3;c=torch.randn(7,dtype=torch.float64)
    control=analyze(a,e,a@c,e@c,torch.ones(90,dtype=a.dtype),torch.ones(40,dtype=a.dtype));assert control['rows'][0]['sensitivity_error']<1e-12
    cache=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)
    x=[torch.cat([cache[0]['rows'],extra['rows']]).double(),cache[1]['rows'].double()]
    labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'];y=[r['target'][:,1].double()/19054614563.464127 for r in labels];w=[r['weight'][:,1].double() for r in labels]
    result=[]
    for seed in [1101,1102]:
        program=torch.load(P/f'SPARSE_SUPPORT_EXCHANGE_SEED{seed}_V1.pt',weights_only=True)
        original=torch.load(P/f'SPARSE_QUARTIC_BANK_SEED{seed}_V1.pt',weights_only=True)['pairs'];candidate=torch.triu_indices(144,144,offset=1);occupied=set((original[0]*144+original[1]).tolist());remaining=candidate[:,torch.tensor([int(a)*144+int(b) not in occupied for a,b in candidate.T])];chosen=torch.randperm(remaining.shape[1],generator=torch.Generator().manual_seed(11700))[:256];pool=torch.cat([original,remaining[:,chosen]],1)
        for size,pairs in [(512,program['pairs']),(768,pool)]:
            phi=[features(z,*[t.double() for t in program['factors']],pairs) for z in x]
            result.append(dict(seed=seed,root_products=size,**analyze(*phi,*y,*w)))
    out=dict(rows=result,control=control,seconds=time.monotonic()-start,scope='Root1, opened calibration/evaluation panels. Cutoffs fixed before diagnostic, all reported, no selection/export. Leverage uses weighted calibration design; top10% evaluation leverage group. Pythagoras decomposes weighted evaluation error into evaluation-optimal irreducible residual and calibration-fit excess. Oracle uses evaluation labels, not predictive evidence. Shared producers are calibration-informed. Correlation/concentration is not causation.')
    (P/'SHARED_TRANSFER_GEOMETRY_V1.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in result:print(json.dumps(r))
if __name__=='__main__':main()
