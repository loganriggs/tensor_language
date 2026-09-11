"""Compare the shared-square candidate with already recorded weight factors."""
import hashlib
import json
from pathlib import Path
import torch
from ll1_group_matching_v1 import parent_cp
from ll1_shared_parent_v1 import shared_parent
from ll1_distributed_function_v1 import component_cross
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    saved=torch.load('/dev/shm/bilin18_matched_shared_groups_v1_ll1_native.pt',weights_only=True,map_location='cpu')
    parts=tuple(x.double() for x in saved['parts']);wh=saved['output_whitener'].double()
    pair=json.loads((root/'LL1_GROUP_MATCHING_V1_AUDIT.json').read_text())['native_pair']
    move=shared_parent(tuple(x[pair] for x in parts));target=parent_cp(move);u=move['parent']
    path=root/'WEIGHT_SQUARE_POLISH_V1_FINAL.pt'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    expected=json.loads((root/'WEIGHT_SQUARE_POLISH_V1_RESULT.json').read_text())['checkpoint_sha256']
    assert digest==expected
    old=torch.load(path,weights_only=True,map_location='cpu')
    a=old['model']['a'].double();w=wh@old['writer'].double();bank=(a,a,w)
    cross=component_cross(bank,target).sum(1)
    norms=(a.square().sum(1).square()*w.square().sum(0)).sqrt()
    cos=cross/norms/inner(target,target).sqrt();idx=int(cos.argmax())
    one=(a[[idx]],a[[idx]],w[:,[idx]])
    actual=inner(one,target)/(inner(one,one)*inner(target,target)).sqrt()
    input_cos=(a@u).abs()/a.norm(dim=1)
    shared=torch.load('/dev/shm/bilin18_joint_shared_reader_rank16_v1.pt',weights_only=True,map_location='cpu')
    shared_cos=[float(abs(g['reader'].double()@u)/g['reader'].double().norm()) for g in shared['programs']]
    earlier=torch.load('/dev/shm/bilin18_shared_input_factor_native_v1.pt',weights_only=True,map_location='cpu')
    prior_reader={k:float(((v['readers'].double()@u).abs()/v['readers'].double().norm(dim=1)).max()) for k,v in earlier.items()}
    pronoun=torch.load('/dev/shm/bilin18_shared_native_function_products_v1.pt',weights_only=True,map_location='cpu')['native_form'].double()
    pronoun_cos=float(u@pronoun@u/pronoun.norm())
    writes=[g['alpha']*g['writer'] for g in move['groups']]
    split=float(sum(w.square().sum() for w in writes)/sum(writes).square().sum())
    result=dict(pred_a=abs(float(actual)-float(cos[idx]))<=1e-9,pred_b=float(cos[idx])>=.95,pred_c=float(input_cos.max())>=.99,
                old_square_index=idx,old_square_whole_function_cosine=float(cos[idx]),
                old_square_input_abs_cosine=float(input_cos[idx]),max_old_square_input_abs_cosine=float(input_cos.max()),
                max_input_index=int(input_cos.argmax()),independent_cp_replay=abs(float(actual)-float(cos[idx])),
                prior_shared_rank16_reader_cosines=shared_cos,prior_full_centered_reader_cosines=prior_reader,
                square_vs_prior_pronoun_quadratic_cosine=pronoun_cos,
                split_square_writer_energy_over_combined=split,old_square_artifact_sha256=digest,
                scope='Prior weight-atom alias screen. No calibration/gender naming or new circuit count; input and full-function agreement reported separately. Splitting an arbitrary term into k equal copies lowers a squared group-energy penalty by1/k, but this descriptive pair ratio does not establish why the optimizer split it.')
    (root/'LL1_SHARED_SQUARE_ALIAS_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
