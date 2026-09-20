"""Fixed reader-budget dictionary allocation controls; CPU only."""
import itertools,json,time
from pathlib import Path
import torch
from shared_bank_toys import teacher_for
from shared_bank_fit import fit_bank
from shared_quadratic_bank import bank_entries
from implicit_quartic import entries
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 out=P/'SHARED_BANK_ALLOCATION_V1.json';assert not out.exists();start=time.perf_counter();rows=[];witnesses=[]
 prior=json.load(open(P/'SHARED_BANK_TOYS_V1.json'))['records']
 for r in prior:
  rows.append({k:r[k] for k in ['family','optimizer','lr','seed','error','selected_step']}|dict(bank=3,width=2,input_parameters=72,output_parameters=18,source='SHARED_BANK_TOYS_V1'))
 indices=torch.cartesian_prod(*[torch.arange(6)]*4)
 for family in ['coordinate','rotated_signed','dense','common_factor','squares']:
  teacher,U,V=teacher_for(family)
  initial=([u.clone() for u in U.reshape(6,1,6)],[v.clone() for v in V.reshape(6,1,6)])
  result,u,v,c=fit_bank(teacher,6,1,'adam',.005,0,0,initial)
  target=entries(*teacher,indices);pred=bank_entries(u,v,indices)@c.T
  error=float((pred-target).norm()/target.norm());witnesses.append(dict(family=family,coefficient_error=error))
  for bank,width in [(2,3),(6,1)]:
   for optimizer,lr,seed in itertools.product(['adam','muon'],[.005,.05],[0,1]):
    result,_,_,_=fit_bank(teacher,bank,width,optimizer,lr,seed,600)
    row=dict(family=family,bank=bank,width=width,optimizer=optimizer,lr=lr,seed=seed,input_parameters=72,output_parameters=3*bank*(bank+1)//2,source='new',**result);rows.append(row)
    print(family,bank,width,optimizer,lr,seed,result['error'],flush=True)
    out.write_text(json.dumps(dict(records=rows,witnesses=witnesses,complete=False,seconds=time.perf_counter()-start),indent=2)+'\n')
 best={(f,b):min(r['error'] for r in rows if r['family']==f and r['bank']==b) for f in ['coordinate','rotated_signed','dense','common_factor','squares'] for b in [2,3,6]}
 predictions=dict(pred_a_witness=all(r['coefficient_error']<1e-6 for r in witnesses),pred_b_recovery=all(best[f,6]<.01 for f in ['coordinate','rotated_signed','dense','common_factor','squares']),pred_c_tradeoff=any(best[f,2]-best[f,6]>.05 for f in ['coordinate','rotated_signed','dense','common_factor','squares']))
 out.write_text(json.dumps(dict(records=rows,witnesses=witnesses,complete=True,predictions=predictions,seconds=time.perf_counter()-start,scope='Same72input-reader scalars, varying9/18/63output parameters. Exact split capacity witnesses. Random optimization failures do not imply impossibility. No circuit identity.'),indent=2)+'\n');print(predictions,flush=True)
if __name__=='__main__':main()
