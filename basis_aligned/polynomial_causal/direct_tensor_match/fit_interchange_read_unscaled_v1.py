"""Compare paired versus Cartesian-context fitting at identical existing graph cost."""
import copy,json,time
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from fixed_read_robust_problem import build
from interchange_quadratic_metric import metric,controls
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
from pairwise_graph_assessment import Assessment
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();check=controls();p=build();d=p['data'];z,h,s=p['metric'].z,p['metric'].h,p['metric'].s;pair=d['pairs'][2];bundle=expand(p['graph']);q=decode(bundle['2']);lin=torch.stack([bundle['2'][k+'_linear'] for k in ('a','b')]);bias=torch.stack([bundle['2'][k+'_bias'] for k in ('a','b')]);reads=torch.einsum('ni,oij,nj->no',z,q,z)+z@lin.T+bias;true=torch.einsum('ni,oij,nj->no',z,torch.stack(pair['Qs']),z);carry=h@pair['a']-true[:,0];V=p['V'];psi=((z-d['mu'])@V).square()-torch.einsum('ik,ij,jk->k',V,d['old_covariance'],V)
 models={}
 for name,cart in [('paired',False),('cartesian',True)]:models[name]=metric(reads[:,0],reads[:,1],true[:,0],true[:,1],carry,s,pair['alpha'],pair['beta'],psi,cart)[:3]
 # Keep original eight safeguards and protect generated paired error within1.1of parent.
 G=np.concatenate((p['G'].numpy(),(models['paired'][0]/(1.1**2*models['paired'][2]))[None].numpy()));b=np.concatenate((p['b'].numpy(),(models['paired'][1]/(1.1**2*models['paired'][2]))[None].numpy()));c=np.r_[p['c'].numpy(),1/1.1**2]
 def constraints(x):return 1-(np.einsum('i,kij,j->k',x,G,x)+2*b@x+c)
 def jac(x):return -2*(np.einsum('kij,j->ki',G,x)+b)
 rows=[]
 for name,(Gt,bt,ct) in models.items():
  A,B,C=Gt.numpy(),bt.numpy(),float(ct);objective=lambda x:float(x@A@x+2*B@x+C);derivative=lambda x:2*(A@x+B)
  fit=minimize(objective,np.zeros(32),jac=derivative,method='SLSQP',constraints=[dict(type='ineq',fun=constraints,jac=jac)],options=dict(maxiter=1500,ftol=1e-12));x=torch.tensor(fit.x,dtype=V.dtype);graph=copy.deepcopy(p['graph']);directions=graph['pairs']['2']['private_reader'][:,-32:];graph['pairs']['2']['product_weights'][-32:,1]+=x/directions.norm(dim=0).square();graph=Assessment(d).correct(graph);assessment=Assessment(d).assess(graph);H=p['H'].clone();H[5]+=torch.einsum('k,kij->ij',x,p['atoms']);replay=float((decode(expand(graph)['2'])[1]-H[5]).norm()/H[5].norm());assert replay<1e-10
  scores={k:dict(before=float(cc.sqrt()),after=float((x@gg@x+2*bb@x+cc).sqrt())) for k,(gg,bb,cc) in models.items()};row=dict(objective=name,success=bool(fit.success),iterations=int(fit.nit),message=fit.message,max_constraint_violation=float(max(0,-constraints(fit.x).min())),scores=scores,assessment=assessment,export_replay=replay,delta_weights=fit.x.tolist());rows.append(row)
  if row['success'] and row['max_constraint_violation']<1e-8:torch.save(graph,P/f'INTERCHANGE_READ_{name.upper()}_V1.pt')
  print(json.dumps({k:v for k,v in row.items() if k not in ('delta_weights','assessment')}),flush=True)
 out=dict(controls=check,records=rows,predictions=dict(pred_a_instrument=all(r['success'] and r['max_constraint_violation']<1e-8 for r in rows),pred_b_cartesian_gain=rows[1]['scores']['cartesian']['after']<=.9*rows[1]['scores']['cartesian']['before'],pred_c_objective_distinction=rows[1]['scores']['cartesian']['after']<rows[0]['scores']['cartesian']['after']),seconds=time.monotonic()-start,scope='Original448calibration states only. Cartesian risk is all200704normalizer-clamped source/context pairings. Only32existing coefficients change; not fresh native donor behavior or circuitidentification.')
 (P/'INTERCHANGE_READ_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
