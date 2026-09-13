"""Exact one-input tensor metric: joint versus independent folded-term compression."""
import json,time,signal
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    signal.alarm(240);torch.set_num_threads(2);torch.manual_seed(352);tic=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    J=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')['mixed_map'].double()
    L,R,D=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    A,B=L@J,R@J;d=J.shape[0];m=L.shape[0]
    # G12[i,j] contracts output and uncompressed y indices of T1_i and T2_j.
    gd=D.T@D
    G1=A.T@(gd*(R@R.T))@A
    G2=B.T@(gd*(L@L.T))@B
    G12=A.T@(gd*(R@L.T))@B
    G=G1+G2+G12+G12.T
    def eig(g):
        e,u=torch.linalg.eigh((g+g.T)/2)
        assert float(e.min())>=-1e-10*float(e.max())
        return e.flip(0),u.flip(1)
    e1,u1=eig(G1);e2,u2=eig(G2);e,u=eig(G)
    total=float(torch.trace(G));assert total>0
    checks=[]
    for _ in range(3):
        x=torch.randn(d,dtype=torch.float64)
        direct=(D*(A@x)[None,:])@R+(D*(B@x)[None,:])@L
        actual=float(x@G@x);expected=float(direct.square().sum())
        checks.append(abs(actual-expected)/expected)
    assert max(checks)<1e-10
    cells=[]
    for rank in (4,8,16,32,64,128):
        budget=2*rank*(d+m);joint_rank=budget//(d+2*m)
        U,V=u1[:,:rank],u2[:,:rank]
        cross_residual=(torch.trace(G12)-torch.trace(U.T@G12@U)
                        -torch.trace(V.T@G12@V)
                        +torch.trace((U.T@G12@V)@(V.T@U)))
        separate_error=float(e1[rank:].sum()+e2[rank:].sum()+2*cross_residual)
        joint_error=float(e[joint_rank:].sum())
        assert min(separate_error,joint_error)>-1e-9*total
        cells.append(dict(separate_rank_each=rank,joint_rank=joint_rank,
                          separate_extra_scalars=budget,joint_extra_scalars=joint_rank*(d+2*m),
                          separate_relative_error=(max(0,separate_error)/total)**.5,
                          joint_relative_error=(max(0,joint_error)/total)**.5,
                          separate_storage_over_original_J=budget/J.numel(),
                          joint_storage_over_original_J=joint_rank*(d+2*m)/J.numel()))
    result=dict(cells=cells,gram_checks=checks,total_tensor_squared_norm=total,
                cross_term_fraction=float(2*torch.trace(G12))/total,
                original_J_scalars=J.numel(),common_L_R_D_scalars=L.numel()+R.numel()+D.numel(),
                folded_A_B_scalars=A.numel()+B.numel(),
                joint_ranks_for_relative_error={str(t):next((i for i in range(d+1)
                    if float(e[i:].sum())<=t*t*total),d) for t in (.01,.02,.05,.1,.2)},
                seconds=time.perf_counter()-tic,
                scope='Weights-only exact coefficient Frobenius metric for D[(LJz)*(Ry)+(RJz)*(Ly)]. '
                'Independent z,y on one folded Jw leg; joint spectral input projector is optimal only '
                'within shared orthogonal-input projection class. Separate term eigensolves optimize each '
                'term, not globally optimal joint loss for two distinct projectors. Common L/R/D weights '
                'and native normalizers/generators remain external. No native behavioral or global '
                'compression claim; storage comparison uses original J, not inflated folded matrices.')
    (P/'INTERACTION_INPUT_MODE_COMPRESSION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
