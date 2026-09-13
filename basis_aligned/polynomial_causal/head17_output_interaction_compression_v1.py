"""Setting2: selected output readers folded into MLP17 and the head17.2 write."""
from pathlib import Path
import json,time,signal
import torch
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    signal.alarm(180);torch.set_num_threads(2);torch.manual_seed(354);tic=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']
    pairs=sorted(set((r['uk_id'],r['us_id']) for r in rows));assert len(pairs)==6
    U=sd['lm_head.weight'].double();O=torch.stack([U[a]-U[b] for a,b in pairs])
    W=torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt',weights_only=True)['output_matrix'].double()
    L,R,D=[sd['transformer.h.17.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    LW,RW=L@W,R@W;C=O@D
    T=torch.stack([L.T@(c[:,None]*RW)+R.T@(c[:,None]*LW) for c in C])
    S=torch.stack([LW.T@(c[:,None]*RW) for c in C]);S=(S+S.transpose(1,2))/2
    linear=O@W
    z=torch.randn(12,1152,dtype=torch.float64);h=torch.randn(12,128,dtype=torch.float64);v=h@W.T
    actual=torch.einsum('bi,oia,ba->bo',z,T,h)+torch.einsum('ba,oac,bc->bo',h,S,h)
    expected=(((z@L.T)*(v@R.T)+(v@L.T)*(z@R.T)+(v@L.T)*(v@R.T))@D.T)@O.T
    check=float((actual-expected).norm()/expected.norm());assert check<1e-10
    output_basis=torch.linalg.eigh(T.flatten(1)@T.flatten(1).T)[1].flip(1)
    rotated=torch.einsum('oj,oia->jia',output_basis,T)
    spectra=[torch.linalg.svdvals(x).square() for x in T]
    rotated_spectra=[torch.linalg.svdvals(x).square() for x in rotated]
    head_energy=torch.linalg.svdvals(T.flatten(0,1)).square()
    input_energy=torch.linalg.svdvals(T.permute(1,0,2).reshape(1152,-1)).square()
    total=float(T.square().sum());cells=[]
    for rank in (1,2,4,8,16,32,64):
        budget=6*rank*(1152+128)
        headrank=min(128,budget//(128+6*1152))
        inputrank=min(768,budget//(1152+6*128))
        cells.append(dict(per_output_rank=rank,per_output_scalars=budget,
                          per_output_error=(sum(float(e[rank:].sum()) for e in spectra)/total)**.5,
                          output_rotated_error=(sum(float(e[rank:].sum()) for e in rotated_spectra)/total)**.5,
                          rotated_adapter_extra_scalars=36,
                          shared_head_rank=headrank,shared_head_scalars=headrank*(128+6*1152),
                          shared_head_error=(float(head_energy[headrank:].sum())/total)**.5,
                          shared_input_rank=inputrank,shared_input_scalars=inputrank*(1152+6*128),
                          shared_input_error=(float(input_energy[inputrank:].sum())/total)**.5))
    result=dict(pairs=pairs,tensor_shape=list(T.shape),mixed_scalars=T.numel(),
                quadratic_scalars=S.numel(),linear_scalars=linear.numel(),formula_check=check,
                cells=cells,seconds=time.perf_counter()-tic,
                scope='Weight-derived six fixed UK-US pre-readout contrast numerators for MLP17 change '
                'under arbitrary residual z and head17.2 write W h. T mixed tensor compressed; S quadratic '
                'and linear output terms retained, not counted as compressed. Per-output rank matrices '
                'are an LL1 baseline with fixed output directions; output rotation is a spectral candidate '
                'not optimized LL1. Shared input/head projections are exact one-mode optima. '
                'Input RMS, final RMS and separate token softcaps remain external; contrasts alone do '
                'not reconstruct saturated token-logit differences. No native effect or whole-program claim.')
    (P/'HEAD17_OUTPUT_INTERACTION_COMPRESSION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
