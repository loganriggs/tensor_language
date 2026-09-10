"""Frozen new reporter/noun combinations; no inference-based row selection."""
import json
from pathlib import Path
import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidate_correlative_pair as old
import circuit_fast_screen_candidate_correlative_disjoint_either_not as disjoint
import circuit_fast_screen_candidates as lex

SEED=9111280


def build():
    agent=lambda i:lex._REPORTERS[i][0]
    alternate=lambda i:lex._REPORTERS[i][1]
    noun1=lambda i:lex._REPORTERS[(i+5)%32][0]
    noun2=lambda i:lex._REPORTERS[(i+13)%32][0]
    obj=lambda i:lex._OBJECTS[(i+3)%32]
    a=bs.BehaviourSpec(
        task_id='correlative_recombined_v1.both_vs_neither',vocabulary=(' and',' nor'),
        generator_role='fixed_reporter_noun_recombination',answer_role='and_nor_next_token',
        a1=bs.Family('bare_recombined','bare_cue_swap',lambda i,pos:old._bare(agent(i),noun1(i),pos)),
        a2=bs.Family('report_recombined','report_cue_swap',lambda i,pos:old._report(agent(i),noun2(i),pos)),
        p_donor=lambda i,pos:old._bare(alternate(i),noun1(i),pos),
        a1_suffix=lambda i:' the '+noun1(i),a2_suffix=lambda i:' the '+noun2(i),
        directions=('both_to_neither','neither_to_both'),kinds=('both','neither'))
    c=bs.BehaviourSpec(
        task_id='correlative_recombined_v1.either_vs_not',vocabulary=(' or',' but'),
        generator_role='fixed_disjoint_reporter_object_recombination',answer_role='or_but_next_token',
        a1=bs.Family('praise_recombined','praise_cue_swap',lambda i,pos:f"The {agent(i)} praised {'either' if pos else 'not'} the {obj(i)}"),
        a2=bs.Family('report_recombined','report_cue_swap',lambda i,pos:f"In the notes the {agent(i)} named {'either' if pos else 'not'} the {obj(i)}"),
        p_donor=lambda i,pos:f"The {alternate(i)} praised {'either' if pos else 'not'} the {obj(i)}",
        a1_suffix=lambda i:' the '+obj(i),a2_suffix=lambda i:' the '+obj(i),
        directions=('either_to_not','not_to_either'),kinds=('either','not'))
    rows_a=a.api()[0](groups=16,seed=SEED);rows_c=c.api()[0](groups=16,seed=SEED)
    panels={name:[r for r in rows_a if r['family']==name] for name in ('A1','A2','P')}
    panels['C']=[r for r in rows_c if r['family']=='A1']
    old_rows=old.build_rows()+disjoint.build_rows()
    old_tokens={tuple(row[side+'_ids']) for row in old_rows for side in ('base','donor')}
    new_tokens={tuple(row[side+'_ids']) for rows in panels.values() for row in rows for side in ('base','donor')}
    overlap=old_tokens&new_tokens
    assert not overlap, 'Frozen recombinations overlap old source prompts; do not select another offset silently'
    for rows in panels.values():
        assert len(rows)==16
        assert all(all(row['construction_checks'].values()) for row in rows)
        assert all(len(row['base_ids'])==len(row['donor_ids']) for row in rows)
    return {'seed':SEED,'panels':panels,'provenance':{'new_unique_tokenized_prompts':len(new_tokens),
             'old_unique_tokenized_prompts':len(old_tokens),'exact_overlap':len(overlap),
             'scope':'New reporter/noun combinations relative to complete original source banks; vocabulary/templates reused, not training OOD or global benchmark novelty'}}


if __name__=='__main__':
    result=build();out=Path(__file__).with_name('CORRELATIVE_RECOMBINED_ROWS_V1.json')
    with out.open('x') as stream:json.dump(result,stream,sort_keys=True);stream.write('\n')
    print(json.dumps(result['provenance']))
