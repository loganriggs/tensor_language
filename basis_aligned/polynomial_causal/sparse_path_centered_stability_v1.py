"""A centered energy identity<=1e-8 B >=8 original joint pairs cosine>=.9 C centered totalcos>=.9.
Output-common red-team without refitting or revising original correspondence.
"""
import json,hashlib
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from sparse_path_coefficient_gram_v1 import gram


def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);p=Path(__file__).parent;out=p/'SPARSE_PATH_CENTERED_STABILITY_V1.json';assert not out.exists()
    source=p/'SPARSE_PATH_STABILITY_ATLAS_V1.json';old=json.loads(source.read_text());assert old['pred_a']
    ap=p/'COUPLED_SPARSE_PATH_CONTINUE_V1_PROGRAMS.pt';assert digest(ap)==old['source_program_sha256']
    programs=torch.load(ap,weights_only=True,map_location='cpu')['programs']
    binding=json.loads((p/'COUPLED_SPARSE_PATH_CONTINUE_V1_BINDING.json').read_text())['files']
    ck=next(k for k in binding if k.endswith('/pytorch_model.bin'))
    # Check checkpoint bytes by streaming rather than allocating them.
    h=hashlib.sha256()
    with open(ck,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    assert h.hexdigest()==binding[ck]
    state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');u=state['lm_head.weight'].double();mean=u.mean(0)
    full=u.T@u;common=len(u)*torch.outer(mean,mean);center=full-common;errors=[];norms=[];energies=[]
    for a in programs:
        g=gram(a,a,center)[0];norms.append(g.sum());energies.append(g.diag())
        direct=u@a['physical_writer'];expected=(direct-direct.mean(0)).square().sum(0)
        errors.append(float((g.diag()-expected).norm()/expected.norm()));assert g.diag().min()>0
    rows=[]
    for c in old['comparisons']:
        ia=next(i for i,a in enumerate(programs) if a['mode']==c['first']['mode'] and a['seed']==c['first']['seed'])
        ib=next(i for i,a in enumerate(programs) if a['mode']==c['second']['mode'] and a['seed']==c['second']['seed'])
        g=gram(programs[ia],programs[ib],center)[0];cos=g/(energies[ia][:,None]*energies[ib][None,:]).sqrt();assert torch.isfinite(cos).all()
        errors.append(max(0.,float(cos.abs().max())-1));i,j=torch.tensor(c['matching']).T;inherited=cos[i,j]
        ri,rj=linear_sum_assignment(-cos.numpy());total=float(g.sum()/(norms[ia]*norms[ib]).sqrt())
        rows.append(dict(first=c['first'],second=c['second'],total_function_cosine=total,inherited_matches_at_point9=int((inherited>=.9).sum()),
            rematched_at_point9=int((cos[ri,rj]>=.9).sum()),inherited_cosines=inherited.tolist()))
    joint=next(c for c in rows if c['first']['mode']=='joint' and c['second']['mode']=='joint')
    result=dict(pred_a=max(errors)<=1e-8,pred_b=joint['inherited_matches_at_point9']>=8,pred_c=joint['total_function_cosine']>=.9,
        maximum_error=max(errors),comparisons=rows,source_atlas_sha256=digest(source),
        scope='Centered coefficient stability with frozen correspondence and unchanged factors. Common pre-tanh vocabulary writing is not assumed behaviorally irrelevant; this tests a structural stability confound only.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='inherited_cosines'} for r in rows],indent=2));assert result['pred_a']

if __name__=='__main__':main()
