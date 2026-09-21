"""Common/private quadratic completion with a jointly regressed cross map.

For Q_o=[[A_o,B_o],[B_o.T,C_o]], choose E minimizing
sum_o||B_o-E C_o||_F^2. Then approximate Q_o by a shared form
A_o-E C_o E.T and private (v+E.T s).T C_o (v+E.T s).
Only cross blocks change. This is not the unrestricted best low-rank fit.
"""
import torch
from quadratic_pair_blocks import compile_pair

def compile_common_private(targets,shared,private,refine_steps=0):
 P,R=torch.linalg.qr(shared,mode='reduced')
 V=torch.linalg.qr(private-P@(P.T@private),mode='reduced').Q
 U=torch.cat([P,V],1);T=U.T@targets@U;r=P.shape[1]
 A=T[:,:r,:r];B=T[:,:r,r:];C=T[:,r:,r:]
 # Stacked rectangular least squares avoids squaring the condition number.
 lhs=torch.cat(list(C),dim=1).T;rhs=torch.cat(list(B),dim=1).T
 solution=torch.linalg.lstsq(lhs,rhs,driver='gelsd');E=solution.solution.T
 C0=C.clone();refinement=None
 if refine_steps:
  from common_private_als import fit
  E,C,refinement=fit(B,C0,steps=refine_steps)
 common=A-E@C@E.T
 cs=compile_pair(common[0],common[1]);cp=compile_pair(C[0],C[1])
 mapping=torch.linalg.solve(R,cs['input_transform'])
 reader=(V+P@E)@cp['input_transform']
 indices=torch.cat([cs['product_indices'],cp['product_indices']+torch.tensor([[r],[r],[0]],dtype=torch.int64)],1)
 weights=torch.cat([cs['product_weights'],cp['product_weights']],0)
 constructed=P@common@P.T+(V+P@E)@C@(V+P@E).T
 projection=U@T@U.T
 cross_error=2*(B-E@C).square().sum();private_error=(C-C0).square().sum()
 measured=(constructed-projection).square().sum()
 assert abs(float(measured-cross_error-private_error))<1e-8*max(1,float(targets.square().sum()))
 return dict(shared_map=mapping,private_reader=reader,product_indices=indices,product_weights=weights,shared_indices=torch.arange(r),private_indices=torch.arange(r,U.shape[1])),constructed,dict(shared_width=r,private_width=V.shape[1],least_squares_rank=int(solution.rank),cross_squared_error=float(cross_error),private_squared_error=float(private_error),refinement=refinement,coupling_norm=float(E.norm()),coupling_spectral_norm=float(torch.linalg.matrix_norm(E,ord=2)),private_norm_ratio=float(C.norm()/C0.norm()),shared_core_norm_ratio=float(common.norm()/T.norm()),projection_squared_error=float((targets-projection).square().sum()),constructed_relative_error=float((targets-constructed).norm()/targets.norm()),common_compiler=cs['diagnostics'],private_compiler=cp['diagnostics'])
