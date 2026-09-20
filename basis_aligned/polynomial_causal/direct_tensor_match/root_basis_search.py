"""Change quadratic feature basis, preserving shared leaf computations."""
import itertools,json,time
from pathlib import Path
import torch
from shared_quadratic_bank import bank_gram
from audit_learned_bank_stability import qgram
P=Path(__file__).resolve().parent

def symmetric_square(S):
 i,j=torch.triu_indices(len(S),len(S));a,b=i[:,None],j[:,None];T=S[a,i]*S[b,j];return T+S[a,j]*S[b,i]*(i!=j)

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1833)
 source=torch.load(P/'NATIVE_LEARNED_SHARED_BANK_V1.pt',weights_only=True)['students'];supports=torch.tensor(list(itertools.combinations(range(10),8)));rotations=[torch.linalg.qr(torch.randn(4,4))[0] for _ in range(64)];start=time.perf_counter();rows=[]
 S=torch.randn(4,4);q=torch.randn(100,4);i,j=torch.triu_indices(4,4);r=q@S.T;replay=float(((q[:,i]*q[:,j])@symmetric_square(S).T-r[:,i]*r[:,j]).abs().max());assert replay<1e-10
 for key,s in source.items():
  U,V,C=[s[n].double() for n in ['U','V','C']];G=bank_gram(U,V);norm=((C.T@C)*G).sum();Q=qgram(U,V,U,V);ev,B=torch.linalg.eigh(Q);assert float(ev.min())>0;white=(B*ev.rsqrt())@B.T;bases=[('identity',torch.eye(4)),('whitening',white)]+[(f'orthogonal_{n}',R) for n,R in enumerate(rotations)]+[(f'whitened_{n}',R@white) for n,R in enumerate(rotations)];best=-float('inf');identity=None
  for name,S in bases:
   T=symmetric_square(S);newG=T@G@T.T;cross=C@G@T.T;K=cross.T@cross;blocks=newG[supports[:,:,None],supports[:,None,:]];rhs=K[supports[:,:,None],supports[:,None,:]];gains=torch.linalg.solve(blocks,rhs).diagonal(dim1=-2,dim2=-1).sum(1);n=int(gains.argmax());gain=float(gains[n]);error2=max(0,1-gain/float(norm))
   if name=='identity':identity=error2
   if gain>best:best=gain;bestS=S;bestT=T;bestname=name;chosen=supports[n];besterror=error2;condition=float(torch.linalg.cond(blocks[n]));writer=torch.linalg.solve(blocks[n],cross[:,chosen].T).T
  delta=C-writer@bestT[chosen];direct=float(((delta.T@delta)*G).sum()/norm);assert abs(direct-besterror)<1e-8
  fullwriter=C@torch.linalg.inv(bestT);fullreplay=float((fullwriter@bestT-C).norm()/C.norm());assert fullreplay<1e-10
  row=dict(optimizer=key[0],lr=key[1],seed=key[2],identity_relative_error=identity**.5,best_relative_error=direct**.5,squared_error_ratio=direct/max(identity,1e-30),basis=bestname,basis_matrix=bestS.tolist(),basis_condition=float(torch.linalg.cond(bestS)),selected_gram_condition=condition,full_basis_replay=fullreplay,support=chosen.tolist(),reduced_float_parameters=46096,support_integers=16);rows.append(row);print(key,row['identity_relative_error'],row['best_relative_error'],bestname,flush=True)
 high=[r for r in rows if r['optimizer']=='adam' and r['lr']==.005];predictions=dict(pred_a_replay=replay<1e-10 and all(r['full_basis_replay']<1e-10 for r in rows),pred_b_improvement=all(r['squared_error_ratio']<.5 for r in high),pred_c_compact=all(r['best_relative_error']<.05 for r in high))
 (P/'ROOT_BASIS_SEARCH_V1.json').write_text(json.dumps(dict(records=rows,polynomial_basis_replay=replay,predictions=predictions,seconds=time.perf_counter()-start,scope='Exact sparse approximation of exported students under130invertible quadratic bases. Not native teacher refitting or semantic identity; includes16mixing scalars and16support integers.'),indent=2)+'\n');print(predictions)
if __name__=='__main__':main()
