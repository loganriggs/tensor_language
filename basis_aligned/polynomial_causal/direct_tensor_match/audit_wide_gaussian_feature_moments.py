"""Compare exact Gaussian and cached empirical quartic-feature moment geometry.
No new fitting. Scale-invariant Gram comparison distinguishes overall energy
from shape. Prediction: Gaussian/first-panel shape gap exceeds panel-to-panel.
"""
import json,time
from pathlib import Path
import torch
from gaussian_bank_metric import gram_by_degree
from empirical_quartic_dictionary import features
P=Path(__file__).resolve().parent


def aligned_gap(reference,candidate):
    scale=(reference*candidate).sum()/candidate.square().sum()
    return float((reference-scale*candidate).norm()/reference.norm()),float(scale)


def main():
    start=time.monotonic();torch.set_num_threads(2);torch.set_grad_enabled(False)
    eye=torch.eye(4,dtype=torch.float64)
    assert aligned_gap(eye,3*eye)[0]<1e-14
    assert aligned_gap(eye,torch.diag(torch.tensor([1.,2.,3.,4.],dtype=torch.float64)))[0]>.1
    old=torch.load(P/'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt',weights_only=True);U,V=old['U'].double(),old['V'].double()
    xs=[p['rows'].double() for p in torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']]
    phi=[features(x,U,V) for x in xs];empirical=[p.T@p/len(p) for p in phi];means=[p.mean(0) for p in phi]
    panelgap,panel_scale=aligned_gap(empirical[0],empirical[1]);transform=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms']['second_floor01'].double();records=[]
    for name,t in [('isotropic',None),('second_floor01',transform)]:
        a,b=(U,V) if t is None else (U@t,V@t);parts,meta=gram_by_degree(a,b);G=sum(parts.values());gap,scale=aligned_gap(empirical[0],G)
        mean_gap,mean_scale=aligned_gap(means[0],meta['mean'])
        records.append(dict(law=name,calibration_gram_shape_error=gap,optimal_gram_energy_scale=scale,calibration_mean_shape_error=mean_gap,optimal_mean_scale=mean_scale,gaussian_raw_feature_energy=float(G.trace()),calibration_raw_feature_energy=float(empirical[0].trace())))
    weighted=records[1];result=dict(records=records,panel_to_panel_gram_shape_error=panelgap,panel_to_panel_optimal_scale=panel_scale,prediction_gaussian_gap_exceeds_panel_gap=weighted['calibration_gram_shape_error']>panelgap,seconds=time.monotonic()-start,scope='Fixed learned32 dictionary; exact zero-mean Gaussian raw moments versus two cached2048-row panels. Gram shape controls student self geometry only; no teacher cross, covariance-law causal attribution, heldout confirmation or nativecircuitadoption.')
    (P/'WIDE_GAUSSIAN_FEATURE_MOMENTS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
