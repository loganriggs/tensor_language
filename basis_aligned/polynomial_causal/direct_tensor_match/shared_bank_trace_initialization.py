import itertools,json
from pathlib import Path
import torch
from shared_bank_toys import teacher_for
from shared_bank_fit import fit_bank
from signed_quadratic_width import spectral_factors
from implicit_quartic import entries
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=P/'SHARED_BANK_TRACE_INITIALIZATION_V1.json';assert not out.exists();rows=[];checks=[];initializations=[];idx=torch.tensor(list(itertools.product(range(6),repeat=4)))
 for family in ['coordinate','rotated_signed','dense','common_factor','squares']:
  teacher,U,V=teacher_for(family);oracle,_,_,_=fit_bank(teacher,3,2,'adam',.05,0,0,initial=(list(U),list(V)));assert oracle['error']<1e-6;checks.append(dict(family=family,teacher_bank_replay_error=oracle['error']));Hfull=entries(*teacher,idx).T.reshape(3,6,6,6,6);H=torch.einsum('vijkk->vij',Hfull).flatten(1);_,_,right=torch.linalg.svd(H,full_matrices=False);Q=right[:3].reshape(3,6,6);Q=(Q+Q.transpose(-1,-2))/2;trueQ=(U.transpose(-1,-2)@V+V.transpose(-1,-2)@U)/2;cos=torch.linalg.svdvals(torch.linalg.qr(Q.flatten(1).T)[0].T@torch.linalg.qr(trueQ.flatten(1).T)[0]);candidates=[];torch.manual_seed(1752);rotations=[torch.eye(3)]+[torch.linalg.qr(torch.randn(3,3))[0] for _ in range(15)]
  for turn,rotation in enumerate(rotations):
   forms=(rotation@Q.flatten(1)).reshape_as(Q);us=[];vs=[]
   for form in forms:
    u,v=spectral_factors(form,2);u=torch.nn.functional.pad(u,(0,2-u.shape[1]));v=torch.nn.functional.pad(v,(0,2-v.shape[1]));us.append(u.T);vs.append(v.T)
   score,_,_,_=fit_bank(teacher,3,2,'adam',.05,0,0,initial=(us,vs));candidates.append((score['error'],turn,us,vs))
  error,turn,us,vs=min(candidates,key=lambda x:x[0]);initializations.append(dict(family=family,unfolded_span_principal_cosines=cos.tolist(),selected_rotation=turn,initial_error=error))
  for optimizer in ['adam','muon']:
   row,_,_,_=fit_bank(teacher,3,2,optimizer,.05,0,600,initial=(us,vs));row.update(family=family,method='trace16',optimizer=optimizer,bank=3,seed=0);rows.append(row);print(family,optimizer,'trace',row['initial_error'],row['error'],flush=True)
  out.write_text(json.dumps(dict(records=rows,capacity_checks=checks,initializations=initializations,scope='10followups. Trace16 includesextra weights-only initialization search onsmall teacher input-trace contraction; not a native algorithm or equalcomputebudget comparison.'),indent=2)+'\n')
if __name__=='__main__':main()
