"""Post-fit fixed-root1 calibration coverage diagnostic; no validation claim."""
import json
from pathlib import Path
import torch
from empirical_quartic_dictionary import features
from audit_root_feature_conditions import root_features
from paired_root_compiler import cast
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 base=cast(torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True),torch.float64);panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)['rows'];xs=[torch.cat([panels[0]['rows'],extra]).double(),panels[1]['rows'].double()];data=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'];phi=[features(x,base['U'],base['V']) for x in xs];scales=phi[0].square().mean(0).sqrt();z=[v/scales for v in phi];rows=[]
 for arm in ['uniform','sensitive']:
  program=cast(torch.load(P/f'SENSITIVE_ROOT_{arm.upper()}_V2.pt',weights_only=True),torch.float64);weights=torch.ones(len(z[0]),dtype=torch.float64) if arm=='uniform' else data[0]['weight'][:,1]/data[0]['weight'][:,1].mean();normal=z[0].T@(weights[:,None]*z[0])+len(z[0])*1e-6*torch.eye(z[0].shape[1],dtype=torch.float64);chol=torch.linalg.cholesky(normal)
  leverage=[(q*torch.cholesky_solve(q.T,chol).T).sum(1) for q in z];pred=root_features(program,xs[1])[:,1];target=data[1]['target'][:,1];w=data[1]['weight'][:,1];error=(pred-target).square()*w;reference=target.square()*w;top=torch.argsort(leverage[1],descending=True)[:205]
  rows.append(dict(arm=arm,train_feature_leverage_median=float(leverage[0].median()),eval_feature_leverage_median=float(leverage[1].median()),eval_above_train_99percentile=int((leverage[1]>torch.quantile(leverage[0],.99)).sum()),eval_highest_leverage_decile_weighted_error_share=float(error[top].sum()/error.sum()),eval_highest_leverage_decile_reference_share=float(reference[top].sum()/reference.sum()),eval_weighted_error=float((error.sum()/reference.sum()).sqrt()),scope='Root1 only. Ridge inverse-design feature leverage, not calibrated uncertainty. Opened-panel post-fitdiagnostic; higherror concentration alone doesnotproveOOD.'))
 (P/'SENSITIVE_ROOT_LEVERAGE_AUDIT_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
