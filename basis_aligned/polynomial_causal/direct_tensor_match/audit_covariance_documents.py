import json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def stats(x):
 mu=x.mean(0);center=x-mu;cov=center.T@center/len(x);return mu,cov

def main():
 torch.set_num_threads(4);torch.set_default_dtype(torch.float64);start=time.perf_counter();panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];splits=[];full=[];checks=[]
 for panel,p in enumerate(panels):
  x=p['rows'].double().reshape(32,64,1152);mu,cov=stats(x.flatten(0,1));checks.append(float((cov-p['covariance'].double()).norm()/cov.norm()));full.append((mu,cov));g=torch.Generator();g.manual_seed(1850+panel)
  for seed in range(8):
   perm=torch.randperm(32,generator=g);a,A=stats(x[perm[:16]].flatten(0,1));b,B=stats(x[perm[16:]].flatten(0,1));splits.append(dict(panel=panel,split=seed,covariance_relative_difference=float((A-B).norm()/((A.norm()+B.norm())/2)),mean_cosine=float(a@b/a.norm()/b.norm()),centered_rank_ceiling=1023,ambient_dimension=1152))
 ev,V=torch.linalg.eigh(full[0][1]);torch.manual_seed(1850);random=torch.randn(1152,16);random=random/random.norm(dim=0);directions={'calibration_top16':V[:,-16:],'calibration_bottom16':V[:,:16],'random16':random};rows=[];normalized=[c/(c.trace()/1152) for _,c in full]
 for name,Q in directions.items():
  variances=[(Q*(c@Q)).sum(0) for c in normalized];ratio=variances[1]/variances[0];rows.append(dict(directions=name,calibration_variances=variances[0].tolist(),evaluation_variances=variances[1].tolist(),variance_ratios=ratio.tolist(),quartic_penalty_ratios=ratio.pow(4).tolist()));print(name,'variance ratio min/median/max',float(ratio.min()),float(ratio.median()),float(ratio.max()),'quartic',float(ratio.pow(4).min()),float(ratio.pow(4).median()),float(ratio.pow(4).max()))
 median=float(torch.tensor([r['covariance_relative_difference'] for r in splits]).median());predictions=dict(pred_a_rows=max(checks)<1e-6,pred_b_sensitivity=median>.5,pred_c_amplification=any(v>4 or v<.25 for r in rows for v in r['quartic_penalty_ratios']));out=dict(row_stat_replays=checks,document_splits=splits,median_split_covariance_difference=median,directions=rows,predictions=predictions,seconds=time.perf_counter()-start,scope='Descriptive document-split and fixed-direction metric sensitivity; overlapping splits are not independentreplicates. No studentselection or population confidenceclaims.');(P/'COVARIANCE_DOCUMENT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print('median split shift',median,predictions)
if __name__=='__main__':main()
