from pathlib import Path
import json,time,torch
from complete_even_key_objective_v1 import grams,loss
from shared_query_product_objective_v1 import rotation
P=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(2);tic=time.perf_counter()
 native=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);basis=native['key_basis'][1].double();allgrams=[]
 for position in [1,4,16,63]:
  m=[native[q][1].double().T@rotation(position).T@native[k][1].double() for q,k in [('q1','k1'),('q2','k2')]]
  allgrams.append(grams(*m,basis))
 packed=[torch.stack([g[i] for g in allgrams]) for i in range(4)]
 adapters=torch.cat([native[k][1].double()@basis for k in ['k1','k2']]);vh=torch.linalg.svd(adapters,full_matrices=False).Vh
 u=vh[:48].T;old=loss(u@u.T,*packed)
 prior=torch.load(P/'SHARED_QUERY_PRODUCT_FIT_V1_PROGRAM.pt',weights_only=True)['basis_rotation'];priorloss=loss(prior@prior.T,*packed)
 eigen=torch.linalg.eigh(packed[3].sum(0)).eigenvectors[:,-48:];eigenloss=loss(eigen@eigen.T,*packed)
 times=[]
 for i in range(12):
  pp=(u@u.T).detach().requires_grad_(True);start=time.perf_counter();value=loss(pp,*packed)/old;grad=torch.autograd.grad(value,pp)[0];times.append(time.perf_counter()-start)
 torch.save(dict(grams=packed,original_basis=u,queryfold_basis=prior,outside_basis=eigen,normalization=old.detach()),P/'COMPLETE_EVEN_KEY_V1_GRAMS.pt')
 out={'original_svd_loss':float(old),'inside_queryfold_loss':float(priorloss),'outside_spectral_loss':float(eigenloss),'queryfold_relative_change':float(priorloss/old-1),'gram_scalars':sum(g.numel() for g in packed),'median_value_gradient_seconds':float(torch.tensor(times[2:]).median()),'total_seconds':time.perf_counter()-tic,'scope':'Actual unnormalized complete even-key replacement coefficient objective atfourpositions; weight-only price and startingpoints, not fit or behavioral validation.'}
 (P/'COMPLETE_EVEN_KEY_V1_PREPARATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
