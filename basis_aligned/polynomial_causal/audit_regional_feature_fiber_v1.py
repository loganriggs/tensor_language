"""Weights-only same-feature/same-norm witness for existing regional MLP16 readers."""
import hashlib,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    binding=json.loads((P/'REGIONAL_RECURSIVE_KEY_V1_BINDING.json').read_text())['files']
    ck=next(k for k in binding if k.endswith('pytorch_model.bin'))
    sd=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].double()
    readers=torch.cat([atoms[-2,:2,:1152],atoms[-2:,2,:1152]],0)
    assert torch.linalg.matrix_rank(readers)==4
    b=torch.linalg.qr(readers.T,mode='reduced')[0]
    fold=torch.load(P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    left,right=[sd[f'transformer.h.16.mlp.{n}.weight'].double() for n in ('Left','Right')]
    d=1152;eps=torch.finfo(torch.float32).eps;cells=[];witnesses=[]
    for i,coeff in enumerate(fold['folded_down'].double()):
        sb=.5*(left.T@(coeff[:,None]*(right@b))+right.T@(coeff[:,None]*(left@b)))
        cross=sb-b@(b.T@sb);u,s,vh=torch.linalg.svd(cross,full_matrices=False)
        z=b@vh[0];w=u[:,0];pair=(d/2)**.5*torch.stack([z+w,z-w])
        feature_error=float(((pair[0]-pair[1])@b).norm()/((pair@b).norm()))
        norm_error=float(abs(pair[0].square().sum()-pair[1].square().sum())/d)
        normed=pair/(pair.square().mean(-1,keepdim=True)+eps).sqrt()
        output=((normed@left.T)*(normed@right.T))@coeff+fold['folded_bias'][i]
        actual=output[0]-output[1];predicted=2*d*s[0]/(1+eps)
        replay=float(abs(actual-predicted)/predicted.abs())
        cells.append(dict(reader=i,cross_singular_value=float(s[0]),cross_relative_to_sb=float(s[0]/sb.norm()),feature_equality_error=feature_error,norm_equality_error=norm_error,outputs=output.tolist(),observed_gap=float(actual),predicted_gap=float(predicted),relative_gap_replay_error=replay))
        witnesses.append(pair)
    a=all(c['feature_equality_error']<=1e-12 and c['norm_equality_error']<=1e-12 and c['relative_gap_replay_error']<=1e-10 for c in cells)
    result={'pred_a':a,'pred_b':a and all(c['cross_relative_to_sb']<=1e-10 for c in cells)}
    result.update(cells=cells,scope='Exact failure certificate only for the fixed4linear-current-features plus norm, over arbitrary real MLP16 inputs. Witnesses need not be token-reachable. Does not reject approximate/reachable-domain closure or alternative nonlinear shared features.',source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_sha=binding[ck])
    torch.save(dict(basis=b,witnesses=torch.stack(witnesses)),P/'REGIONAL_FEATURE_FIBER_V1_ARTIFACT.pt')
    (P/'REGIONAL_FEATURE_FIBER_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
