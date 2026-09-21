"""Disentangle mean direction and centered covariance in local reader bounds."""
import json,time
from pathlib import Path
import torch
from quartic_reader_rank_bound import analyze
P=Path(__file__).resolve().parent


def main():
    start=time.monotonic();torch.set_num_threads(2);torch.set_grad_enabled(False)
    grams=torch.load(P/'QUARTIC_READER_GRAMS_V1.pt',weights_only=True)['grams'];cal=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];mu=cal['mean'].double();cov=cal['covariance'].double();second=cal['second_moment'].double();transforms=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms'];records=[]
    for key in ['centered_floor01','centered_floor10','second_floor01']:
        L=transforms[key].double();r,_=analyze([L.T@g@L for g in grams],[128,256,512]);records.append(dict(metric=key,analysis=r))
    fractions=[]
    for g in grams:
        mean=float(mu@g@mu);variation=float((g*cov.T).sum());raw=float((g*second.T).sum());fractions.append(dict(mean_fraction_of_raw_second_moment_energy=mean/raw,centered_fraction=variation/raw,decomposition_relative_discrepancy=abs(mean+variation-raw)/raw))
    result=dict(records=records,mean_energy_fractions=fractions,prediction_mean_explains_over_half_both=all(r['mean_fraction_of_raw_second_moment_energy']>.5 for r in fractions),seconds=time.monotonic()-start,scope='Same calibration moments for both anchor panels. Centered vs uncentered perturbation laws, not a Gaussian nonlinear closure. Raw covariance normalization may use finite-sample correction; decomposition discrepancy reported.')
    (P/'READER_METRIC_MEAN_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
