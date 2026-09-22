"""Label-access least-squares capacity diagnostic, never an evaluation score."""
import json,time
import torch
from audit_conditional_residual_accounting import P,SCALE,load

def solve(A,Y):
 fit=torch.linalg.lstsq(A,Y,rcond=1e-13,driver='gelsd');C=fit.solution
 residual=A@C-Y;normal=float((A.T@residual).norm()/(A.norm()*Y.norm()))
 Q,R=torch.linalg.qr(A,mode='reduced');qr_res=Y-Q@(Q.T@Y)
 feature=(residual.square().sum(0)/Y.square().sum(0)).sqrt();qr_feature=(qr_res.square().sum(0)/Y.square().sum(0)).sqrt()
 agreement=float((feature-qr_feature).abs().max());rank=int(fit.rank)
 assert rank==A.shape[1] and normal<1e-10 and agreement<1e-8
 return dict(rank=rank,columns=A.shape[1],singular_values=fit.singular_values.tolist(),condition_number=float(fit.singular_values[0]/fit.singular_values[-1]),rcond=1e-13,normal_residual=normal,svd_qr_error_agreement=agreement,pooled_minimum=float(residual.norm()/Y.norm()),per_output_minima=feature.tolist(),small_output_rms=float(feature[4:].square().mean().sqrt()))

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();Y=data['target'].double()/SCALE
 pairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T;rows=[]
 for seed in [1001,1002]:
  p,sha=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');fs=p['factors']
  A=torch.cat([torch.stack([xx@f.T for f in fs]).prod(0) for xx in x.split(1024)])
  for kind,design,target in [('values',A,Y),('responses',A[pairs[1]]-A[pairs[0]],Y[pairs[1]]-Y[pairs[0]])]:
   row=dict(seed=seed,kind=kind,parent_sha256=sha,states=len(design),**solve(design,target));rows.append(row);print(seed,kind,row['pooled_minimum'],row['small_output_rms'],min(row['per_output_minima'][4:]),max(row['per_output_minima'][4:]),row['condition_number'],flush=True)
 pred=all(max(r['per_output_minima'][4:])>.1 for r in rows)
 (P/'CURRENT_CP_ORACLE_CAPACITY_V1.json').write_text(json.dumps(dict(rows=rows,prediction=pred,seconds=time.monotonic()-start,scope='Oracle uses evaluation labels. Separate value/response empirical least-squares minima within frozen512-feature spans. Not held-out evaluation or population lower bounds. No candidate export.'),indent=2)+'\n')
if __name__=='__main__':main()
