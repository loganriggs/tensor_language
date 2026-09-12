"""Descriptive full-vocabulary readout of frozen self plus leading eight SVD branches.
No token selection for fitting, no semantic labels, no behavior/promotion verdict.
"""
import json,hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    binding=json.loads((P/'MATCHED_PARTNER_SUBSPACE_V1_BINDING.json').read_text())['files']
    checkpoint=next(k for k in binding if k.endswith('/pytorch_model.bin'))
    artifact=P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt'
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==json.loads((P/'MATCHED_PARTNER_SUBSPACE_V1_RESULT.json').read_text())['artifact_sha']
    sd=torch.load(checkpoint,weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double();uc=u-u.mean(0)
    program=torch.load(artifact,weights_only=True);w=program['output_writers'][:,:9].double()
    raw=uc@w;cosine=raw/(uc.norm(dim=1)[:,None]*w.norm(dim=0)[None,:]).clamp_min(1e-30)
    tokenizer_path=Path('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    tok=Tokenizer.from_file(str(tokenizer_path));vocab=tok.get_vocab_size()
    def extremes(values):
        result={}
        for side,sign in [('positive',1),('negative',-1)]:
            ids=torch.topk(sign*values,12).indices.tolist()
            result[side]=[dict(id=i,token=tok.decode([i]) if i<vocab else '<padded output row>',value=float(values[i])) for i in ids]
        return result
    branches=[]
    for j in range(9):
        branches.append(dict(branch=j,kind='exact_self' if j==0 else 'SVD_partner',
                             centered_output_energy=float(raw[:,j].square().sum()),
                             raw=extremes(raw[:,j]),cosine=extremes(cosine[:,j]),
                             top12_abs_energy_fraction=float(raw[:,j].abs().topk(12).values.square().sum()/raw[:,j].square().sum())))
    g=raw[:,1:].T@raw[:,1:];normalized=g/(g.diag().sqrt()[:,None]*g.diag().sqrt()[None,:])
    result=dict(branches=branches,partner_output_orthogonality_error=float((normalized-torch.eye(8)).abs().max()),
                vocabulary_rows=u.shape[0],tokenizer_rows=vocab,program_sha=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                tokenizer_sha=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='All output rows retained. Raw coefficient loadings and residual-direction cosine are descriptive; antipodal extremes do not identify semantic tasks. Partner branch signs depend on SVD gauge.')
    (P/'MATCHED_PARTNER_TOKEN_READOUT_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(orthogonality=result['partner_output_orthogonality_error'],branches=[dict(branch=b['branch'],fraction=b['top12_abs_energy_fraction'],positive=[x['token'] for x in b['cosine']['positive'][:6]],negative=[x['token'] for x in b['cosine']['negative'][:6]]) for b in branches]),indent=2))

if __name__=='__main__':main()
