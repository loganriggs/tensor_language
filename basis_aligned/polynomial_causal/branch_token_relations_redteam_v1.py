"""Red-team token relations: marginal effects, small branches, output aliases.
A permutation-mean and analytic independent-pair identities <=1e-12.
B >=80% of qualifying atlas cells have >=5% of their parent coefficient energy.
C parent1branch0 matched difference energy <=80% of random-pair expectation
  in each add_s/add_es/y_to_ies relation. Failure means the claimed matched
  lexical organization needs narrowing, not absence of systematic token writes.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;out=root/'BRANCH_TOKEN_RELATIONS_REDTEAM_V1.json'
    assert not out.exists()
    source=root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    atlas_path=root/'BRANCH_TOKEN_RELATIONS_V1.json'
    atlas=json.loads(atlas_path.read_text());assert atlas['pred_a']
    saved=torch.load(source,weights_only=True,map_location='cpu')
    binding=json.loads((root/'FROZEN_BRANCH_TENSE_V6_BINDING.json').read_text())['files']
    checkpoint=next(p for p in binding if p.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    labels=[];columns=[];energies=[]
    for parent,node in enumerate(saved['nodes']):
        for branch in range(node['writers'].shape[1]):
            w=node['writers'][:,branch]
            columns.append(w/w.norm());labels.append((parent,branch))
            energies.append(float(node['singular_values'][branch].square()/node['singular_values'].square().sum()))
    cols=torch.stack(columns,1)
    physical=torch.linalg.solve_triangular(saved['output_whitener'],cols,upper=True)
    f=weights['lm_head.weight'].double()@physical
    cells=[];errors=[]
    generator=torch.Generator().manual_seed(6101)
    for relation,pairs in atlas['relations'].items():
        ids=torch.tensor(pairs,dtype=torch.long)
        a,b=f[ids[:,0]],f[ids[:,1]]
        delta=b-a
        shuffled=b[torch.randperm(len(b),generator=generator)]-a
        errors.append(float((shuffled.mean(0)-delta.mean(0)).abs().max()))
        independent=(a.square()+b.square()).mean(0)-2*a.mean(0)*b.mean(0)
        # Independent full cartesian pairs on a small subset verify the identity.
        aa,bb=a[:19],b[:19]
        predicted=(aa.square()+bb.square()).mean(0)-2*aa.mean(0)*bb.mean(0)
        actual=(bb[None]-aa[:,None]).square().mean((0,1))
        errors.append(float((predicted-actual).abs().max()))
        for j,(parent,branch) in enumerate(labels):
            old=next(c for c in atlas['cells'] if c['relation']==relation and c['parent']==parent and c['branch']==branch)
            cells.append(dict(parent=parent,branch=branch,relation=relation,parent_energy_fraction=energies[j],
                original_qualifies=old['qualifies'],nonnegligible_qualifies=old['qualifies'] and energies[j]>=.05,
                matched_to_independent_difference_energy=float(delta[:,j].square().mean()/independent[j]),
                matched_coherence=old['coherence'],independent_coherence=float(delta[:,j].mean().abs()/independent[j].sqrt())))
    cosine=cols.T@cols
    aliases=[]
    for j,(p,b) in enumerate(labels):
        for k,(q,c) in enumerate(labels[:j]):
            if p==q:continue
            aliases.append(dict(left=[p,b],right=[q,c],writer_cosine=float(cosine[j,k]),
                left_parent_energy=energies[j],right_parent_energy=energies[k]))
    aliases.sort(key=lambda r:-abs(r['writer_cosine']))
    passing=[c for c in cells if c['original_qualifies']]
    survival=sum(c['nonnegligible_qualifies'] for c in passing)/len(passing)
    chosen=[c for c in cells if c['parent']==1 and c['branch']==0 and c['relation'] in ('add_s','add_es','y_to_ies')]
    a=max(errors)<=1e-12
    result=dict(pred_a=a,pred_b=a and survival>=.8,pred_c=a and all(c['matched_to_independent_difference_energy']<=.8 for c in chosen),
        replay_errors=errors,nonnegligible_survival=survival,cells=cells,top_crossparent_writer_cosines=aliases[:10],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,atlas_path)},
        checkpoint_sha256=binding[checkpoint],scope='Exact marginal-pair accounting on weights. Random-pair expectation uses all cartesian pairs '
        'including identity index. Strong means can reflect suffix or capitalization categories without lexical pairing. '
        'Parent fractions describe standalone node functions and must not be summed as native whole-layer energy. '
        'Writer alignment does not establish identical input computation or causally reusable variables.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','sources','top_crossparent_writer_cosines')},indent=2))
    print(json.dumps(dict(parent1_s_relations=chosen,writer_aliases=aliases[:4]),indent=2))


if __name__=='__main__':main()
