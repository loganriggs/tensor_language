"""Norm bounds from saved mediation metrics and a sequential group-clamp control."""
import json,hashlib
from pathlib import Path


def mediator_norm_bounds(remaining_error,interaction_norm):
    # total = mediator + remaining + interaction; triangle/reverse triangle.
    return max(0.,remaining_error-interaction_norm),remaining_error+interaction_norm


def group_control():
    # Native x=1, attention a=x=1, MLP m=(x+a)^2=4, output x+a+m=6.
    # Under source x=0, recompute live groups; clamped groups use native writes.
    def run(x,attention_live,mlp_live):
        a=x if attention_live else 1.
        m=(x+a)**2 if mlp_live else 4.
        return x+a+m
    cube=[[run(0.,a,m) for m in (False,True)] for a in (False,True)]
    assert cube==[[5.,2.],[4.,0.]]
    frozen=cube[0][0];total=frozen-cube[1][1]
    attention=frozen-cube[1][0];mlp=frozen-cube[0][1];interaction=total-attention-mlp
    assert (total,attention,mlp,interaction)==(5.,1.,3.,1.)
    # Full-source-edit MLP write is zero; replaying it under native attention gives 1,
    # whereas actually recomputing that MLP under the clamp gives 2.
    wrong_replay=0.+1.+0.;assert wrong_replay!=cube[0][1]
    assert run(1.,False,False)==run(1.,True,True)==6.
    return {'passed':True,'edited_output_cube':cube,'native_output':6.,'direct_carry_effect':1.,
        'late_response_effects':{'total':total,'attention':attention,'mlp':mlp,'interaction':interaction},
        'wrong_full_edit_write_replay':wrong_replay,
        'scope':'Synthetic sequential group interventions only; broad groups are not assumed semantic circuits.'}


def main():
    p=Path(__file__).resolve().parent;source=p/'MLP8_MLP9_BYPASS_FACTORIAL_V1_RESULT.json';data=json.loads(source.read_text());assert data['predictions']['pred_a_instrument']
    reports=[]
    for row in data['reports']:
        error=row['dominance_errors']['remaining']['vocabulary'];interaction=row['interaction_errors']['mixed']['vocabulary']
        lo,hi=mediator_norm_bounds(error,interaction)
        reports.append({'world_id':row['world_id'],'layout':row['layout'],
            'remaining_task_passed':row['task_remaining_passed'],
            'mediator_vocabulary_norm_lower':lo,'mediator_vocabulary_norm_upper':hi,
            'vocabulary_interaction_norm':interaction})
    result={'reports':reports,'group_control':group_control(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':'Rigorous triangle bounds on opened centered mixed-vocabulary effects; norm is not signed explained fraction. No adjacent-module promotion.'}
    with (p/'MLP9_CONSUMER_SCOPE_BOUNDS_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for layout in ('original','fronted_pp'):
        rows=[r for r in reports if r['layout']==layout]
        print(layout,'bounds_envelope',min(r['mediator_vocabulary_norm_lower'] for r in rows),max(r['mediator_vocabulary_norm_upper'] for r in rows),'task_remaining_passes',sum(r['remaining_task_passed'] for r in rows))
    print(result['group_control'])

if __name__=='__main__':main()
