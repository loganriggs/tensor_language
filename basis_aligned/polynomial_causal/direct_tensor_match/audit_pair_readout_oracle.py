"""Hindsight least-squares pair floor in the fixed ten-feature linear readout class."""
import json
from pathlib import Path
import torch
from fit_paired_readout import pairs
from extract_scalar_modes import evaluate
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(3);torch.set_grad_enabled(False)
 s={k:v.double() for k,v in torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True)['program'].items()};view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);u=view['output_directions'].double()[:,1];mu=view['output_mean'].double();cache=P.parents[1]/'bilinear_quotient/.rowcache';cal=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];target=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);panels=[('calibration',cal['rows'],target['targets'][0],torch.load(cache/'fineweb_n96_skip1200.pt',weights_only=True)[:32,:65])]
 for p in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels']:
  t=int(p['context']);panels.append((f'diagnostic_{t}',p['rows'],p['targets'],torch.load(cache/'fineweb_n192_skip7000.pt',weights_only=True)[32:64,:t+1]))
 records=[]
 for name,xx,yy,tokens in panels:
  x=xx.double();y=(yy.double()-mu)@u;p=(x@s['A'].T)*(x@s['B'].T);h=(p@s['root_left'].T)*(p@s['root_right'].T);phi=torch.cat([p,h],1);a,b=pairs(tokens)['same_token'];design=phi[b]-phi[a];dy=y[b]-y[a];scale=design.square().mean(0).sqrt();z=design/scale;solution=torch.linalg.lstsq(z,dy,driver='gelsd');res=dy-z@solution.solution;baseline=evaluate(s,x)[:,1];base=(baseline[b]-baseline[a])-dy;oracle=float(res.square().mean());old=float(base.square().mean());orth=float((z.T@res).norm()/(z.norm()*res.norm()));assert orth<1e-10 and oracle<=old+1e-8
  records.append(dict(panel=name,pairs=len(a),design_rank=int(solution.rank),baseline_pair_mse=old,hindsight_pair_mse=oracle,max_fractional_mse_improvement=1-oracle/old,baseline_relative_error=float(base.norm()/dy.norm()),hindsight_relative_error=float(res.norm()/dy.norm()),normal_equation_relative_residual=orth))
 result=dict(records=records,scope='Exact numerical least-squares minimum on each finite panel within the frozen ten-feature linear-readout class. Diagnostic targets used only for hindsight bound, no exported candidate. Not a bound on changed features, native logit effects, unseen data or arbitrary functions of readers.')
 (P/'PAIR_READOUT_ORACLE_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
