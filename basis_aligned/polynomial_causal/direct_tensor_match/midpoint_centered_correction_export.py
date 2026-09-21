"""Compress a centered original-operator correction into shared linear features."""
from pathlib import Path
import json,torch
from midpoint_centered_operator_refit import cross_gram,toy

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);p=Path(__file__).resolve().parent
 out=p/'MIDPOINT_CENTERED_CORRECTION_V1.json';assert not out.exists();checks=toy()
 ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
 state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');ru=torch.linalg.qr(state['lm_head.weight'].double(),mode='r').R
 L=state['transformer.h.17.mlp.Left.weight'].double();R=state['transformer.h.17.mlp.Right.weight'].double();C=ru@state['transformer.h.17.mlp.Down.weight'].double()
 rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n,m,y=[rows[k].flatten(0,1).double() for k in ['n','m','y']]
 S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
 e={k:v.double() for k,v in torch.load(p/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt',weights_only=True)['products512'].items()}
 A=e['Pn']@e['Tn'];B=e['Pm']@e['Tm'];W=e['output_basis']@e['output_core'];nc=n-n.mean(0);mc=m-m.mean(0)
 Mn=nc.T@nc/len(n);Mm=mc.T@mc/len(m);K=(A.T@Mn@A)*(B.T@Mm@B);cross=cross_gram(A,B,L,R,Mn,Mm)@C.T
 scale=K.diag().sqrt().clamp_min(1e-12);Kn=K/scale[:,None]/scale[None,:];eig,V=torch.linalg.eigh(Kn)
 D=(V@((V.T@(cross/scale[:,None]-Kn@(scale[:,None]*W.T)))/eig.clamp_min(1e-10)[:,None]))/scale[:,None]
 # Factor in normalized product coordinates; avoids unnecessary bad conditioning.
 root=(V*eig.clamp_min(1e-10).sqrt()[None,:])@V.T
 invroot=(V*eig.clamp_min(1e-10).rsqrt()[None,:])@V.T
 U,s,Vh=torch.linalg.svd(root@(scale[:,None]*D)@S,full_matrices=False)
 aa,bb,ac,bc=n@A,m@B,nc@A,mc@B;raw=aa*bb;centered=ac*bc
 checks['centered_product_compile']=float((centered-(raw-aa*bb.mean(0)-bb*aa.mean(0)+aa.mean(0)*bb.mean(0))).norm()/centered.norm());assert checks['centered_product_compile']<1e-12
 base=(raw-e['product_mean'])@W.T+e['full_mean'];den=((y-y.mean(0))@S).norm();programs={'baseline':e};records=[]
 for rank in [4,8,16,32,64]:
  left=(invroot@U[:,:rank]*s[:rank])/scale[:,None];writers=torch.linalg.solve(S,Vh[:rank].T)
  pred=base+(centered@left)@writers.T
  records.append(dict(rank=rank,correction_relative_error=float(s[rank:].norm()/s.norm()),paired_error=float(((pred-y)@S).norm()/den),weight_coefficients=1291264+(512+1152)*rank,additional_mean_state=1024))
  if rank in [8,32]:programs[f'rank{rank}']={**e,'centered_correction_left':left,'centered_correction_writers':writers,'left_mean':aa.mean(0),'right_mean':bb.mean(0)}
 checks['full_correction_replay']=float((((invroot@U*s)/scale[:,None])@torch.linalg.solve(S,Vh.T).T-D).norm()/D.norm());assert checks['full_correction_replay']<1e-9
 torch.save(programs,p/'MIDPOINT_CENTERED_CORRECTION_GRAPHS_V1.pt')
 out.write_text(json.dumps(dict(checks=checks,records=records,scope='Original weight centered operator, optimal output-rank compression conditional on fixed product dictionary and separable centered input metric. Additive centered correction shares raw products. Counts extra512+512 means separately; mean multiplications and linear arithmetic not free. No native result yet.'),indent=2)+'\n');print(out.read_text())
if __name__=='__main__':main()
