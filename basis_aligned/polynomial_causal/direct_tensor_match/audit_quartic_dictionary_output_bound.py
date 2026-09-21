"""Output-rank error lower bounds for a three-feature pure quartic readout."""
import json,time,itertools
from pathlib import Path
import torch
from shared_quadratic_bank import bank_gram,bank_entries

def spectrum(c,gram):
 gram=(gram+gram.T)/2;ev,v=torch.linalg.eigh(gram);negative=float(ev.min()/ev.abs().max());assert negative>-1e-10
 sigma=torch.linalg.svdvals(c@(v*ev.clamp_min(0).sqrt()));energy=sigma.square().sum()
 return dict(gram_min_relative_eigenvalue=negative,relative_rank_errors={str(k):float((sigma[k:].square().sum()/energy).sqrt()) for k in range(1,len(sigma)+1)},squared_singular_values=sigma.square().tolist())

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent;torch.manual_seed(1901)
 u=torch.randn(4,2,3,dtype=torch.float64);v=torch.randn_like(u);indices=torch.tensor(list(itertools.product(range(3),repeat=4)));dense=bank_entries(u,v,indices);gram=bank_gram(u,v);control=float((gram-dense.T@dense).norm()/gram.norm());assert control<1e-12
 source=torch.load(p/'NATIVE_MIXED_ROOT_V2.pt',weights_only=True)['students'][('adam',.005,1)];U,V,S,C=[source[k].double() for k in ['U','V','mixing','C']];support=source['support'];i,j=torch.triu_indices(4,4);root_i,root_j=i[support],j[support]
 H=S[root_i][:,i]*S[root_j][:,j]+S[root_i][:,j]*S[root_j][:,i]*(i!=j);c=C@H
 records=[dict(metric='symmetric_coefficient_frobenius',**spectrum(c,bank_gram(U,V)))];checks=[]
 panels=torch.load(p/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 for index,panel in enumerate(panels):
  x=panel['rows'][:512].double();q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);phi=q[:,i]*q[:,j];mix=q@S.T;direct=(mix[:,root_i]*mix[:,root_j])@C.T;replay=float((phi@c.T-direct).norm()/direct.norm());checks.append(replay);assert replay<1e-12
  records.append(dict(metric=f'empirical_panel_{index}',samples=len(x),**spectrum(c,phi.T@phi/len(phi))))
 result=dict(records=records,dense_gram_control=control,native_mixing_replay=max(checks),seconds=time.monotonic()-start,scope='Necessary output-rank bounds for archived pure quartic program, not full model. Three scalar quadratic features have six pairwise products and at most six output directions. Applies even with free input-feature learning. Coefficient and empirical-function metrics are distinct. No claim rank-six bound is attainable with three quadratic features; affine/skip branches change capacity and cost.')
 (p/'QUARTIC_DICTIONARY_OUTPUT_BOUND_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rank6_bounds={r['metric']:r['relative_rank_errors']['6'] for r in records},controls=[control,max(checks)],seconds=result['seconds'])))
if __name__=='__main__':main()
