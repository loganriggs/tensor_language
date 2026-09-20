import json,math,time
from pathlib import Path
import torch
from shared_quadratic_bank import bank_gram
from root_basis_search import symmetric_square
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.perf_counter();source=torch.load(P/'NATIVE_LEARNED_SHARED_BANK_V1.pt',weights_only=True)['students'];screen=json.load(open(P/'ROOT_BASIS_SEARCH_V1.json'))['records'];rows=[];exports={}
 for source_row in screen:
  if source_row['optimizer']!='adam' or source_row['lr']!=.005:continue
  key=('adam',.005,source_row['seed']);s=source[key];U,V,C=[s[n].double() for n in ['U','V','C']];G=bank_gram(U,V);norm=((C.T@C)*G).sum();S0=torch.tensor(source_row['basis_matrix']);support=torch.tensor(source_row['support']);basecond=float(torch.linalg.cond(S0));bestsource=float('inf')
  for optimizer in ['adam','muon']:
   for lr in [.005,.05]:
    K=torch.nn.Parameter(torch.zeros(4,4));opt=torch.optim.Adam([K],lr=lr) if optimizer=='adam' else torch.optim.Muon([K],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf')
    for step in range(601):
     opt.zero_grad();S=torch.matrix_exp(K-K.T)@S0;T=symmetric_square(S)[support];H=T@G@T.T;cross=C@G@T.T;writer=torch.linalg.solve(H,cross.T).T;loss=1-(writer*cross).sum()/norm;value=float(loss.detach())
     if value<best:best=value;bs=S.detach().clone();bw=writer.detach().clone();beststep=step
     if step==600:break
     loss.backward();opt.step()
     for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
    delta=C-bw@symmetric_square(bs)[support];direct=float(((delta.T@delta)*G).sum()/norm);condition=float(torch.linalg.cond(bs));row=dict(source_seed=key[2],optimizer=optimizer,lr=lr,relative_student_error=max(0,direct)**.5,analytic_squared_error=best,direct_squared_error=direct,selected_step=beststep,screen_error=source_row['best_relative_error'],relative_condition_change=abs(condition/basecond-1));rows.append(row);print(row,flush=True)
    if direct<bestsource:bestsource=direct;exports[key]=dict(U=U,V=V,mixing=bs,support=support,C=bw)
 predictions=dict(pred_a_integrity=all(abs(r['analytic_squared_error']-r['direct_squared_error'])<1e-8 and r['relative_condition_change']<1e-8 for r in rows),pred_b_refinement=all(min(r['relative_student_error'] for r in rows if r['source_seed']==seed)<.8*next(r['screen_error'] for r in rows if r['source_seed']==seed) for seed in [0,1]),pred_c_target=all(min(r['relative_student_error'] for r in rows if r['source_seed']==seed)<.05 for seed in [0,1]))
 torch.save(dict(students=exports,scope='Eight mixedroot approximations of frozen fullbank students; native teacher validation pending.'),P/'ROOT_BASIS_REFINE_V1.pt');(P/'ROOT_BASIS_REFINE_V1.json').write_text(json.dumps(dict(records=rows,predictions=predictions,seconds=time.perf_counter()-start),indent=2)+'\n');print(predictions)
if __name__=='__main__':main()
