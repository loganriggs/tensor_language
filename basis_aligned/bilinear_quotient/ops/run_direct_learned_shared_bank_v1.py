#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_gain pred_c_cp
"""Native learned four-quadratic sharedbank, fourproducts/feature,10roots.
pred_a_finite; pred_b_gain>CP8; pred_c_cp heldout error<CP8.
Eight400step exact-gradient fits; price48,384 plusoutputframe.
"""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(bank=4,bilinear_width=4,root_products=10,steps=400,optimizers=['adam','muon'],rates=[.0005,.005],seeds=[0,1],native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from shared_quadratic_bank import normalize_bank,bank_gram,native_bank_cross,bank_entries,roots
 from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_LEARNED_SHARED_BANK_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight'),w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher[0]/=scale;d=1152;gen=torch.Generator(device='cuda');gen.manual_seed(1651);idx=torch.randint(d,(8192,4),device='cuda',generator=gen);target=torch.cat([entries(*teacher,z) for z in idx.split(512)]);x=torch.randn(256,d,device='cuda',generator=gen);C,L,R,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;fy=((h@L.T)*(h@R.T))@C.T;records=[];students={}
 for optimizer,lr,seed in itertools.product(PLAN['optimizers'],PLAN['rates'],PLAN['seeds']):
  torch.manual_seed(seed);us=[torch.nn.Parameter(torch.randn(4,d,device='cuda')/d**.5) for _ in range(4)];vs=[torch.nn.Parameter(torch.randn(4,d,device='cuda')/d**.5) for _ in range(4)];params=us+vs;opt=torch.optim.Adam(params,lr=lr) if optimizer=='adam' else torch.optim.Muon(params,lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=-float('inf');base=None;history=[];runstart=time.perf_counter()
  for step in range(401):
   opt.zero_grad();U,V=normalize_bank(torch.stack(us),torch.stack(vs));G=bank_gram(U,V).double();cross=native_bank_cross(teacher,U,V).double();writer=torch.linalg.solve(G+1e-6*G.diag().mean()*torch.eye(len(G),device='cuda',dtype=torch.float64),cross.T).T;loss=((writer@G)*writer).sum()-2*(cross*writer).sum();gain=float(-loss.detach());assert math.isfinite(gain)
   if gain>best:best=gain;beststep=step;bu=U.detach().clone();bv=V.detach().clone();bc=writer.detach().float()
   if base is None:base=max(gain,1e-30)
   if step%100==0:history.append(dict(step=step,gain=gain,condition=float(torch.linalg.cond(G).detach())))
   if step==400:break
   (loss/base).backward();opt.step()
   for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/400)))
  with torch.no_grad():
   predicted=torch.cat([bank_entries(bu,bv,z)@bc.T for z in idx.split(512)]);q=((x@bu.flatten(0,1).T)*(x@bv.flatten(0,1).T)).reshape(len(x),4,4).sum(2);i,j=roots(4,x.device);f=(q[:,i]*q[:,j])@bc.T;row=dict(optimizer=optimizer,lr=lr,seed=seed,selected_step=beststep,explained_fraction_using_estimated_teacher_norm=best,estimated_coefficient_error=math.sqrt(max(0,1-best)),evaluation_coefficient_error=float((predicted-target).norm()/target.norm()),gaussian_function_error=float((f-fy).norm()/fy.norm()),seconds=time.perf_counter()-runstart,reduced_parameters=48384,history=history);records.append(row);students[(optimizer,lr,seed)]=dict(U=bu.cpu(),V=bv.cpu(),C=bc.cpu());print(json.dumps(row),flush=True);out.write_text(json.dumps(dict(plan=PLAN,records=records,seconds=time.perf_counter()-start),indent=2)+'\n')
 cp=json.load(open(P/'NATIVE_EXACT_CP_GREEDY_V1.json'))['records'][-1];pred=dict(pred_a_finite=all(math.isfinite(r['explained_fraction_using_estimated_teacher_norm']) for r in records),pred_b_gain=max(r['explained_fraction_using_estimated_teacher_norm'] for r in records)>cp['explained_fraction_using_estimated_teacher_norm'],pred_c_cp=min(r['evaluation_coefficient_error'] for r in records)<cp['evaluation_coefficient_error']);guard_torch_save(dict(students=students,teacher_scale=scale),str(P/'NATIVE_LEARNED_SHARED_BANK_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,seconds=time.perf_counter()-start,predictions=pred,scope='Learned shared low-rank quadratics, exact coefficient self/cross gradients, estimated teacher norm. Dense rootwriters, no sparsecore/identity/Gaussian fitting/circuit claim.'),indent=2)+'\n')
if __name__=='__main__':main()
