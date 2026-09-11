"""Frozen native path program identification screen.
A dense Gram control, native norm replay/gauge identities<=1e-8 and finite cos<=1+1e-8.
B >=8one-to-one joint edge matches with coefficient-function cosine>=.9.
C joint total coefficient-function cosine>=.9. Other comparisons/atlas descriptive.
No text fit or semantic promotion; weights determine every component.
"""
import json,hashlib
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from tokenizers import Tokenizer
from sparse_path_coefficient_gram_v1 import gram
from sparse_path_program_v1 import edges


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);p=Path(__file__).parent;out=p/'SPARSE_PATH_STABILITY_ATLAS_V1.json';assert not out.exists()
    assert json.loads((p/'SPARSE_PATH_COEFFICIENT_GRAM_V1_CONTROL.json').read_text())['passed']
    prior=json.loads((p/'COUPLED_SPARSE_PATH_CONTINUE_V1_RESULT.json').read_text());assert prior['pred_a'] and prior['pred_b']
    ap=p/'COUPLED_SPARSE_PATH_CONTINUE_V1_PROGRAMS.pt';assert digest(ap)==prior['artifact_sha256']
    artifact=torch.load(ap,weights_only=True,map_location='cpu');assert artifact['complete'] and len(artifact['programs'])==4
    binding=json.loads((p/'COUPLED_SPARSE_PATH_CONTINUE_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');u=state['lm_head.weight'].double();metric=u.T@u
    programs=artifact['programs'];norms=[];errors=[];energies=[];gauges=[]
    total=json.loads((p/'COUPLED_SPARSE_PATH_PILOT_V2_RESULT.json').read_text())['total_coefficient_norm2']
    for program in programs:
        gg=gram(program,program,metric)[0];norm=gg.sum();norms.append(norm);energy=gg.diag();energies.append(energy)
        r=next(r for r in prior['reports'] if r['mode']==program['mode'] and r['seed']==program['seed'])
        errors.append(abs(float(norm)/(total*r['capture'])-1));assert torch.isfinite(gg).all()
        # Compensated reader-sign gauge: each incident writer acquires the same sign.
        changed=dict(program);bank=program['bank'].clone();bank[0,:,0]*=-1;changed['bank']=bank
        pair=edges(program);sign=torch.where((pair==0).sum(0)%2==1,-1.,1.);changed['physical_writer']=program['physical_writer']*sign
        replay=gram(program,changed,metric)[0];gauges.append(float((gg-replay).norm()/gg.norm()))
    comparisons=[]
    for i,j in [(0,2),(1,3),(0,1),(2,3)]:
        a,b=programs[i],programs[j];g=gram(a,b,metric)[0];cos=g/(energies[i][:,None]*energies[j][None,:]).sqrt()
        assert torch.isfinite(cos).all();errors.append(max(0.,float(cos.abs().max())-1))
        ii,jj=linear_sum_assignment(-cos.numpy());scores=cos[ii,jj];totalcos=float(g.sum()/(norms[i]*norms[j]).sqrt())
        comparisons.append(dict(first=dict(mode=a['mode'],seed=a['seed']),second=dict(mode=b['mode'],seed=b['seed']),total_function_cosine=totalcos,
            matches_at_point9=int((scores>=.9).sum()),matches_at_point8=int((scores>=.8).sum()),matched_cosines=scores.tolist(),matching=list(zip(ii.tolist(),jj.tolist()))))
    tok=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json');vocab=tok.get_vocab_size();atlas=[]
    for idx,program in enumerate(programs):
        if program['mode']!='joint':continue
        logits=u@program['physical_writer'];mean=logits.mean(0);centered=logits[:vocab]-mean
        pair=edges(program);rows=[]
        for e in range(logits.shape[1]):
            top=centered[:,e].topk(6).indices;bottom=(-centered[:,e]).topk(6).indices
            rows.append(dict(edge=e,feature_ids=pair[:,e].tolist(),coefficient_energy=float(energies[idx][e]),
                common_output_energy_fraction=float(len(u)*mean[e]**2/logits[:,e].square().sum()),
                positive=[dict(id=int(t),token=tok.decode([int(t)]),weight=float(centered[t,e])) for t in top],
                negative=[dict(id=int(t),token=tok.decode([int(t)]),weight=float(centered[t,e])) for t in bottom]))
        atlas.append(dict(seed=program['seed'],mode=program['mode'],common_output_energy_fraction=float((len(u)*mean.square()).sum()/energies[idx].sum()),
            edges=sorted(rows,key=lambda e:e['coefficient_energy'],reverse=True)))
    joint=next(c for c in comparisons if c['first']['mode']=='joint' and c['second']['mode']=='joint')
    result=dict(pred_a=max(errors+gauges)<=1e-8,pred_b=joint['matches_at_point9']>=8,pred_c=joint['total_function_cosine']>=.9,
        maximum_norm_cosine_error=max(errors),gauge_replay_errors=gauges,comparisons=comparisons,atlas=atlas,
        source_program_sha256=digest(ap),source_result_sha256=digest(p/'COUPLED_SPARSE_PATH_CONTINUE_V1_RESULT.json'),
        scope='Frozen coefficient-function stability and weight-only signed output atlas; positive/negative writer orientation is gauge-dependent, complete edge tensor comparison is invariant. No text, semantic category claim or behavioral validation.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('atlas','comparisons')},indent=2))
    print(json.dumps([{k:v for k,v in c.items() if k not in ('matched_cosines','matching')} for c in comparisons],indent=2))
    print(json.dumps([dict(seed=a['seed'],common_fraction=a['common_output_energy_fraction'],top_edges=a['edges'][:3]) for a in atlas],indent=2));assert result['pred_a']

if __name__=='__main__':main()
