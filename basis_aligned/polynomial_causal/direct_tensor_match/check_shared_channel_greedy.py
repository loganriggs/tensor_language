import json
from pathlib import Path
import torch
from shared_channel_greedy import gram,greedy,refit

def main():
 torch.set_num_threads(2);torch.manual_seed(31000);rows=[]
 for kind in ['dense','sparse_outputs','duplicated','cancellation','rank_one']:
  L=torch.randn(12,5,dtype=torch.float64);R=torch.randn_like(L);C=torch.randn(3,12,dtype=torch.float64)
  if kind=='sparse_outputs':C[:,4:]=0
  if kind=='duplicated':L[6:]=L[:6];R[6:]=R[:6]
  if kind=='cancellation':L[1]=L[0];R[1]=R[0];C[:,1]=-C[:,0]
  if kind=='rank_one':R=L.clone()
  tensor=.5*(L[:,:,None]*R[:,None,:]+R[:,:,None]*L[:,None,:]);A=tensor.flatten(1);norm=A.norm(dim=1);A/=norm[:,None];K=gram(L,R)/norm[:,None]/norm[None,:];C=C*norm;target=C@A;B=target@A.T
  assert float((K-A@A.T).abs().max())<1e-12
  ids,gains=greedy(K,B,8)
  for k in [3,len(ids)]:
   take=ids[:k];D,gain,normal=refit(K,B,take);direct=torch.linalg.lstsq(A[take].T,target.T,driver='gelsd').solution.T
   delta=float(((D-direct)@A[take]).norm()/target.norm());gap=float((gain-gains[k-1]).abs().max()/target.square().sum(1).max());assert max(delta,gap,normal)<1e-10
   rows.append(dict(kind=kind,k=k,dense_replay=delta,gain_identity=gap,normal_residual=normal))
 Path(__file__).with_name('SHARED_CHANNEL_GREEDY_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print('10 dense projection controls passed')
if __name__=='__main__':main()
