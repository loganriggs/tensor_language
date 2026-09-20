"""Fixed-cost Gaussian/empirical moment comparison on calibration data only."""
import json
from pathlib import Path
import torch
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_grad_enabled(False)
 source=torch.load(P/'QUADRATIC_SKIP_PROGRAM_V1.pt',weights_only=True);base={k:v.double() for k,v in torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4].items()};cal=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)
 assert cal['token_sha256']==targets['token_sha256'][0];assert abs(float(source['teacher_scale'])/float(targets['teacher_scale'])-1)<1e-6
 x=cal['rows'].double();y=targets['targets'][0].double();c=source['primitive_mean'].double();q=(x@base['A'].T)*(x@base['B'].T)-c;residual=y-quartic(base,x);K1=residual.T@q/len(x);G1=q.T@q/len(x);K0=source['cross_covariance'].double();G0=source['primitive_covariance'].double();exports={};rows=[];replays=[]
 for alpha in [0.,.25,.5,.75,1.]:
  K=(1-alpha)*K0+alpha*K1;G=(1-alpha)*G0+alpha*G1;e,v=torch.linalg.eigh((G+G.T)/2);assert float(e.min())>0;invhalf=(v/e.sqrt())@v.T;u,sv,vh=torch.linalg.svd(K@invhalf,full_matrices=False);writer=u[:,:2]*sv[:2];reader=vh[:2]@invhalf;W=writer@reader;model={**base,'constant':base['constant']-W@c,'skip_writer':writer,'skip_reader':reader};archive={k:t.float() for k,t in model.items()};assert sum(t.numel() for t in archive.values())==21948;exports[alpha]=archive
  gaussian_gain=float(2*(W*K0).sum()-((W@G0)*W).sum());emp_gain=float(2*(W*K1).sum()-((W@G1)*W).sum());emp_error=float((quartic(model,x)-y).norm()/y.norm());rows.append(dict(alpha=alpha,gaussian_gain=gaussian_gain,empirical_gain=emp_gain,calibration_relative_error=emp_error,gram_condition=float(e.max()/e.min()),coefficients=21948,products=10))
  if alpha==0:
   old={k:t.double() for k,t in source['programs'][2].items()};replays.append(float((quartic(model,x[:128])-quartic(old,x[:128])).norm()/quartic(old,x[:128]).norm()))
 pred=dict(pred_a_replay=max(replays)<1e-5,pred_b_calibration=rows[-1]['empirical_gain']>=rows[0]['empirical_gain']-1e-8 and rows[0]['gaussian_gain']>=rows[-1]['gaussian_gain']-1e-8)
 result=dict(alphas=[0,.25,.5,.75,1],primary=.5,records=rows,predictions=pred,alpha0_replay=max(replays),transfer_status='pending',scope='Explicitly data-informed calibration polynomial regression versus exact Gaussian moments; same graph, center, rank and cost. No diagnostic-panel fitting.')
 torch.save(dict(programs=exports,teacher_scale=source['teacher_scale'],empirical_cross=K1,empirical_gram=G1,token_sha256=cal['token_sha256']),P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt');(P/'SKIP_METRIC_BLEND_FIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
