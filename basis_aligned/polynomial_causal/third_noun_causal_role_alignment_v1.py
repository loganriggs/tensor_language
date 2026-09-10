"""Audit role alignment against counterfactual information availability."""
import json
import hashlib
from pathlib import Path
from third_noun_write_structure_rows_v1 import positions


def main():
    p=Path(__file__).resolve().parent
    old_path=p/'THIRD_NOUN_VALUE_TRANSFER_V1_ROWS.json'
    new_path=p/'THIRD_NOUN_WRITE_STRUCTURE_V1_ROWS.json'
    old=json.loads(old_path.read_text());new=json.loads(new_path.read_text())
    roles_old=['subject_determiner','subject','verb','object_determiner','object',
               'preposition','attractor_determiner','attractor','to','action']
    roles_new=['preposition','attractor_determiner','attractor','comma','subject_determiner',
               'subject','verb','object_determiner','object','to','action']
    worlds=[];audits=[]
    fronts={w['world_id'].split(':',1)[1]:w for w in new['worlds'] if w['layout']=='fronted_pp'}
    for w in old['worlds']:
        base=positions(dict(w,layout='original'));front=fronts[w['world_id']]
        shared=[role for role in roles_old[base['interaction_start']:] if role in roles_new[front['interaction_start']:]]
        assert shared==['to','action']
        violations=[]
        for source,target,rs,rt in [(base,front,roles_old,roles_new),(front,base,roles_new,roles_old)]:
            for pos in range(source['interaction_start'],source['length']):
                role=rs[pos]
                if role in rt and rt.index(role)<target['interaction_start']:
                    violations.append({'from_layout':source['layout'],'role':role,'source_position':pos,
                                       'target_position':rt.index(role),'target_information_available_at':target['interaction_start']})
        assert len(violations)==2
        audits.append({'world_id':w['world_id'],'common_mature_roles':shared,'noncausal_role_maps':violations})
        for world in (base,front):
            worlds.append(dict(world,common_positions=[world['length']-2,world['length']-1]))
    result={key:old[key] for key in ('corners','reader_names','reader_ids')}
    result.update(worlds=worlds,audits=audits,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (old_path,new_path)},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Causal possibility audit, not evidence of nonzero native writes. Swapping full role-aligned tables can introduce information unavailable at the recipient position. Only to/action are mature in both layouts.')
    with (p/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'worlds':len(worlds),'audited_pairs':len(audits),'sample':audits[0]},indent=2))


if __name__=='__main__':main()
