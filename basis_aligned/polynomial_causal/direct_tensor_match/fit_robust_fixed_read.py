"""Cutting-plane minimax over existing correction coefficients, with exact separation."""
import json,time,copy
from pathlib import Path
import numpy as np
import torch
from fixed_read_robust_problem import build
from quadratic_ball_extrema import extrema
from convex_quadratic_minimax import solve
from pairwise_graph_assessment import Assessment
from pairwise_reader_graph import price
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();p=build();d=p['data'];metric=p['metric'];V=p['V'];atoms=p['atoms'];H=p['H'];G,b,c=p['G'],p['b'],p['c'];radius=1152**.5
 worst=json.loads((P/'NATIVE_WORST_READ_V1.json').read_text());limit=1.1*next(r['worst_absolute_error'] for r in worst['rows'] if r['candidate']=='isotropic_baseline' and r['component']==3 and r['read']=='b');x=torch.zeros(32,dtype=H.dtype);history=[];cuts=[]
 def oracle(x):
  form=H[5]+torch.einsum('k,kij->ij',x,atoms);error=form-metric.true[5];linear=-2*error@d['mu'];bias=-torch.trace(d['old_covariance']@error)+d['mu']@error@d['mu'];result=extrema(error,linear,bias,radius);return result,error,linear,bias
 for iteration in range(8):
  out,E,l,k=oracle(x)
  for key in ('minimum','maximum'):
   z=out[key]['x'];psi=((z-d['mu'])@V).square()-torch.einsum('ik,ij,jk->k',V,d['old_covariance'],V);intercept=z@E@z+l@z+k-psi@x;v=psi/limit;v0=intercept/limit
   G=torch.cat((G,torch.outer(v,v)[None]));b=torch.cat((b,(v*v0)[None]));c=torch.cat((c,v0.square()[None]));cuts.append(dict(iteration=iteration,extremum=key,value=out[key]['value'],certificate=out[key]['certificate']))
  result=solve(G.numpy(),b.numpy(),c.numpy(),initial=x.numpy());x=torch.tensor(result['x'],dtype=H.dtype);new,E,l,k=oracle(x);trial=H.clone();trial[5]+=torch.einsum('k,kij->ij',x,atoms);original=metric.ratios_full(trial);worst_ratio=(new['worst_absolute']/limit)**2;upper=max(float(original.max()),worst_ratio);gap=upper-result['maximum']
  row=dict(iteration=iteration,finite_maximum=result['maximum'],global_maximum=upper,separation_gap=gap,worst_read_squared_ratio=worst_ratio,original_maximum=float(original.max()),original_ratios=original.tolist(),solver=result);history.append(row);print(json.dumps({k:v for k,v in row.items() if k!='solver'}),flush=True)
  (P/'ROBUST_FIXED_READ_PARTIAL_V1.json').write_text(json.dumps(dict(history=history,delta_weights=x.tolist()),indent=2)+'\n')
  if upper<=1 or gap<1e-8:break
 graph=copy.deepcopy(p['graph']);directions=graph['pairs']['2']['private_reader'][:,-32:];graph['pairs']['2']['product_weights'][-32:,1]+=x/directions.norm(dim=0).square();graph=Assessment(d).correct(graph);assessment=Assessment(d).assess(graph);assert price(graph)==price(p['graph'])
 # Compare exported form with the solved form, including actual scalar execution via assessment.
 from pairwise_reader_graph import expand
 from local_shared_reader_graph import decode
 exported=decode(expand(graph)['2'])[1];replay=float((exported-trial[5]).norm()/trial[5].norm());assert replay<1e-10
 active=result['active'];w=np.asarray(result['multipliers']);w=w/w.sum();Ga=np.einsum('k,kij->ij',w,G.numpy()[active]);ba=w@b.numpy()[active];ca=float(w@c.numpy()[active]);e,U=np.linalg.eigh(Ga);keep=e>max(float(e.max()),1)*1e-12;proj=U.T@ba;range_residual=float(np.linalg.norm(proj[~keep]));lower=ca-float(np.sum(proj[keep]**2/e[keep]))
 # A finite convex combination gives a mathematical lower bound only if b is in range.
 result_out=dict(history=history,delta_weights=x.tolist(),cuts=cuts,assessment=assessment,export_replay=replay,numerical_dual_lower_bound=lower,dual_discarded_range_residual=range_residual,dual_gap=history[-1]['global_maximum']-lower,predictions=dict(pred_a_instrument=True,pred_b_fidelity=history[-1]['global_maximum']<=1,pred_c_unchanged_cost=True),seconds=time.monotonic()-start,scope='Fixed32directions/one read only; numerical dual bound is not an interval proof. No global DAG impossibility or fresh validation.')
 (P/'ROBUST_FIXED_READ_V1.json').write_text(json.dumps(result_out,indent=2)+'\n');print(json.dumps({k:v for k,v in result_out.items() if k not in ('history','cuts','delta_weights','assessment')}),flush=True)
 if result_out['predictions']['pred_b_fidelity']:torch.save(graph,P/'ROBUST_FIXED_READ_PROGRAM_V1.pt')
if __name__=='__main__':main()
