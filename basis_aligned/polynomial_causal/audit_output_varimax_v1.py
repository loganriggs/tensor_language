"""Exact sparsity curve and selected function complexity after output rotation."""
from pathlib import Path
import json,torch
from orthogonal_output_varimax_v1 import criterion
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def main():
    torch.set_num_threads(2)
    s=torch.load(P/'OUTPUT_VARIMAX_V1_CHECKPOINT.pt',weights_only=False,map_location='cpu')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double();centered=u-s['unembedding_mean'];initial=centered@s['writer'];a=initial@s['rotation']
    h=s['rotation'].T@s['core'];sq=a.square();ordered=sq.sort(dim=1,descending=True).values;energy=ordered.sum()
    cumulative=ordered.cumsum(1)/ordered.sum(1,keepdim=True)
    needed=(cumulative<.9).sum(1)+1
    curve={str(k):float(ordered[:,:k].sum()/energy) for k in [1,2,4,8,16,32,64,96,128]}
    rownorm=initial.square().sum(1).sqrt();normalized=initial/rownorm[:,None]
    _,g=criterion(normalized,s['rotation'],True);normalized_score=criterion(normalized,s['rotation'])
    l,r=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right']]
    selected=sq.sum(0).topk(4).indices;functions=[]
    for j in selected.tolist():
        raw=(l.T*h[j])@r;q=(raw+raw.T)/2;ev=torch.linalg.eigvalsh(q);eigen_energy=ev.square().sort(descending=True).values;total=eigen_energy.sum()
        best_product=(ev.clamp_min(0).max().square()+ev.clamp_max(0).min().square())/total
        functions.append(dict(factor=j,loading_energy_fraction=float(sq[:,j].sum()/energy),quadratic_norm_squared=float(total),best_single_real_product_capture=float(best_product),best_rank16_square_capture=float(eigen_energy[:16].sum()/total),signed_square_rank90=int(torch.searchsorted(eigen_energy.cumsum(0)/total,.9))+1))
    reference=json.loads((P/'OUTPUT_VARIMAX_V1_RESULT.json').read_text())
    replay=abs(curve['4']-reference['after']['token_top4_energy_fraction'])
    valid=replay<=1e-10 and max(abs(x['quadratic_norm_squared']-1) for x in functions)<=1e-10
    result=dict(instrument_passed=valid,topk_loading_energy=curve,median_factors_for_90percent_per_token=float(needed.double().median()),q10_factors_for_90percent=float(torch.quantile(needed.double(),.1)),q90_factors_for_90percent=float(torch.quantile(needed.double(),.9)),four_sparse_loading_centered_tensor_capture=curve['4']*reference['centered_rank128_capture'],leading_functions=functions,equal_token_row_normalized_relative_stationarity=float(g.norm()*128**.5/normalized_score.abs()),top4_replay_error=replay,redteam=dict(narrow_failure='Converged raw-varimax rotation fails50%top4-loading-energy bar within fixed128-dimensional output subspace',structural_negative=False,orthogonal_basis_restriction=True,one_rotation_initialization=True,raw_loadings_prioritizes_large_norm_tokens=True,fixed_output_subspace=True,scope='Normalized-gradient probe changes the criterion, not evidence that its optimized structure is better. Spectral complexity is exact for these four selected scalar quadratic forms, not a lower bound on joint input reuse.'))
    (P/'OUTPUT_VARIMAX_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert valid

if __name__=='__main__':main()
