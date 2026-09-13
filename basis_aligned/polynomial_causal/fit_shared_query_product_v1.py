from pathlib import Path
import json,time,torch
from shared_query_product_objective_v1 import captured,fit,rotation
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(7131121);tic=time.perf_counter()
 native=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
 basis=native['key_basis'][1].double();a,b=[native[k][1].double()@basis for k in ['k1','k2']];grams=[]
 for position in [1,4,16,63]:
  m1=native['q1'][1].double().T@rotation(position).T@a;m2=native['q2'][1].double().T@rotation(position).T@b
  grams.append((m1.T@m1,m2.T@m2,m1.T@m2))
 g1,g2,c=[torch.stack([g[i] for g in grams]) for i in range(3)]
 _,_,vh=torch.linalg.svd(torch.cat([a,b]),full_matrices=False);spectral=vh[:48].T;total=captured(torch.eye(64,dtype=torch.float64),g1,g2,c);baseline=float(captured(spectral@spectral.T,g1,g2,c)/total)
 rows=[];best=None
 for seed in range(10):
  initial=spectral if seed==0 else spectral+.2*torch.randn_like(spectral) if seed<5 else torch.randn_like(spectral)
  u,receipt=fit(g1,g2,c,initial);receipt['start']=seed;rows.append(receipt)
  if best is None or receipt['capture']>best[0]:best=(receipt['capture'],u)
  print(json.dumps(receipt),flush=True)
 gain=(best[0]-baseline)/(1-baseline)
 out={'pred_a':all(r['monotone'] for r in rows),'pred_b':all(r['converged'] for r in rows),'pred_c':gain>=.01,'baseline_capture':baseline,'best_capture':best[0],'relative_residual_improvement':gain,'rows':rows,'seconds':time.perf_counter()-tic,'scope':'Rank48 shared key subspace, inside-inside product with sharedquerysource androundedpositions1/4/16/63. Weight-only local multistart, not complete attention objective or native validation.'}
 (P/'SHARED_QUERY_PRODUCT_FIT_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');torch.save({'basis_rotation':best[1]},P/'SHARED_QUERY_PRODUCT_FIT_V1_PROGRAM.pt');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
